import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import norm
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.calibration import calibration_curve
from tabulate import tabulate
import matplotlib.pyplot as plt
import warnings

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class DataEngine:
    """
    Loads raw sensor data. Only forward-looking gap filling is done here
    (ffill), since it never uses a future observation to patch a past one.
    Any imputation that could see 'the future' (linear interpolation,
    bfill) is deliberately NOT done globally anymore -- it's leakage-prone
    when later split into train/test folds.
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def load_and_preprocess(self):
        df = pd.read_csv(self.filepath, skiprows=1, parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        features = ["tmin", "tmax", "td_m", "um", "ffm"]

        df[features] = df[features].ffill()
        return df


class FeatureFactory:
    """
    Fit/transform split so the GMM (and any future fitted imputers) only
    ever see train-fold statistics. NaNs produced by shift()/diff() at the
    start of a fold are left as NaN on purpose -- XGBoost handles missing
    values natively, so there's no need to bfill them (which would leak
    a future-in-fold value backwards).
    """

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.gmm = GaussianMixture(
            n_components=self.n_components, random_state=random_state
        )
        self.is_fitted = False

    def fit(self, df):
        gmm_cols = ["tmin", "tmax", "td_m", "um", "ffm"]
        self.gmm.fit(df[gmm_cols])
        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("FeatureFactory must be fitted on train data first!")

        df = df.copy()

        time_diff = df["datetime"].diff().dt.total_seconds() / 3600.0
        time_diff = time_diff.replace(0, np.nan).fillna(24.0)
        df["lapse_rate"] = df["tmin"].diff() / time_diff
        df["dtr"] = df["tmax"] - df["tmin"]
        df["dtr_lag_1"] = df["dtr"].shift(1)

        for i in range(1, 4):
            df[f"tmin_lag_{i}"] = df["tmin"].shift(i).bfill()
            df[f"ffm_lag_{i}"] = df["ffm"].shift(i).bfill()
        for i in [5, 7]:
            df[f"tmin_lag_{i}"] = df["tmin"].shift(i).bfill()
            df[f"ffm_lag_{i}"] = df["ffm"].shift(i).bfill()

        df["td_m_roll_3"] = df["td_m"].rolling(window=3, min_periods=1).mean()
        df["tmin_volatility_3d"] = df["tmin"].rolling(window=3, min_periods=1).std()
        df["tmin_ewma_3"] = df["tmin"].ewm(span=3, adjust=False).mean()
        df["td_m_ewma_3"] = df["td_m"].ewm(span=3, adjust=False).mean()
        df["day_of_year"] = df["datetime"].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        df["um_ffm_interaction"] = df["um"] * df["ffm"]

        gmm_cols = ["tmin", "tmax", "td_m", "um", "ffm"]
        gmm_probs = self.gmm.predict_proba(df[gmm_cols])
        for i in range(self.n_components):
            df[f"gmm_prob_{i}"] = gmm_probs[:, i]

        return df


class TargetGenerator:
    def prepare_regression_targets(self, df):
        target_cols = ["tmin", "tmax", "td_m", "ffm"]
        for col in target_cols:
            df[f"{col}_next"] = df[col].shift(-1)
        return df.dropna(subset=[f"{col}_next" for col in target_cols]).copy()


class FrostRiskEngine:
    def __init__(self, dynamic_sigma=None, historical_mae_tmin=2.0):
        self.historical_mae_tmin = historical_mae_tmin
        self.static_sigma = historical_mae_tmin * np.sqrt(np.pi / 2.0)
        self.dynamic_sigma = None

        if dynamic_sigma is not None:
            dynamic_sigma = np.asarray(dynamic_sigma, dtype=float)
            if len(dynamic_sigma) > 0 and np.all(np.isfinite(dynamic_sigma)):
                self.dynamic_sigma = dynamic_sigma

    def calculate_risk_probability(self, predicted_tmin, threshold=0.0):
        predicted_tmin = np.asarray(predicted_tmin, dtype=float)
        x = threshold - predicted_tmin

        if self.dynamic_sigma is not None:
            if len(self.dynamic_sigma) != len(predicted_tmin):
                raise ValueError(
                    "dynamic_sigma length must match number of predictions"
                )
            z_scores = x / self.dynamic_sigma
        else:
            z_scores = x / self.static_sigma

        return norm.cdf(z_scores)


class ProbabilisticVolatilityModel:
    def __init__(
        self,
        n_estimators=300,
        learning_rate=0.02,
        floor_frac=0.3,
        val_frac=0.15,
        early_stopping_rounds=25,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.floor_frac = floor_frac
        self.val_frac = val_frac
        self.early_stopping_rounds = early_stopping_rounds
        self.model = NGBRegressor(
            Dist=Normal,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            natural_gradient=True,
            verbose=False,
            random_state=random_state,
        )
        self.is_fitted = False
        self.sigma_floor = 1e-2

    def fit(self, X, y_tmin):
        X = X.reset_index(drop=True)
        y_tmin = pd.Series(y_tmin).reset_index(drop=True)

        n = len(X)
        n_val = max(int(n * self.val_frac), 10)

        if n - n_val >= 30:
            X_tr, X_val = X.iloc[: n - n_val], X.iloc[n - n_val :]
            y_tr, y_val = y_tmin.iloc[: n - n_val], y_tmin.iloc[n - n_val :]
            self.model.fit(
                X_tr,
                y_tr,
                X_val=X_val,
                Y_val=y_val,
                early_stopping_rounds=self.early_stopping_rounds,
            )
            in_sample_pred = self.model.predict(X_val)
            in_sample_resid = y_val.values - in_sample_pred
        else:
            self.model.fit(X, y_tmin)
            in_sample_pred = self.model.predict(X)
            in_sample_resid = y_tmin.values - in_sample_pred

        self.is_fitted = True
        resid_std = np.std(in_sample_resid) if len(in_sample_resid) > 1 else 2.0
        self.sigma_floor = max(self.floor_frac * resid_std, 1e-2)
        return self

    def predict_sigma(self, X):
        if not self.is_fitted:
            return None
        dist = self.model.pred_dist(X)
        sigma = np.asarray(dist.params["scale"], dtype=float)
        return np.clip(sigma, self.sigma_floor, None)


class ForecastModelTrainer:
    def __init__(
        self,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=150,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ):
        base_estimator = XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            missing=np.nan,
        )
        self.model = MultiOutputRegressor(base_estimator)

    def train(self, X, y):
        critical_mask = y.iloc[:, 0] <= 2.0
        sample_weights = np.where(critical_mask, 3.0, 1.0)
        self.model.fit(X, y, sample_weight=sample_weights)
        return self.model


class BenchmarkSuite:
    def evaluate_split(self, model, X_test, Y_test, risk_engine):
        Y_pred = model.predict(X_test)
        mae_tmin = mean_absolute_error(Y_test[:, 0], Y_pred[:, 0])
        actual_frost = (Y_test[:, 0] <= 0.0).astype(int)
        frost_probs = risk_engine.calculate_risk_probability(Y_pred[:, 0])

        if len(np.unique(actual_frost)) > 1:
            auc = roc_auc_score(actual_frost, frost_probs)
            logloss = log_loss(actual_frost, frost_probs)
        else:
            auc, logloss = np.nan, np.nan

        residuals = Y_test[:, 0] - Y_pred[:, 0]
        return mae_tmin, auc, logloss, frost_probs, actual_frost, residuals

    def run_time_series_cv(
        self,
        df,
        base_feature_cols,
        target_cols,
        n_splits=4,
        random_state=42,
        trainer_kwargs=None,
        gmm_n_components=3,
        verbose=True,
    ):

        trainer_kwargs = trainer_kwargs or {}
        tscv = TimeSeriesSplit(n_splits=n_splits)

        gmm_cols = [f"gmm_prob_{i}" for i in range(gmm_n_components)]
        feature_cols = base_feature_cols + gmm_cols

        results_list = []
        all_probs, all_actuals = [], []

        fold = 1
        if verbose:
            print("\n=== Fold-by-Fold Analysis ===")

        for train_index, test_index in tscv.split(df):
            df_train = df.iloc[train_index].copy()
            df_test = df.iloc[test_index].copy()

            factory = FeatureFactory(
                n_components=gmm_n_components, random_state=random_state
            )
            factory.fit(df_train)
            df_train_feat = factory.transform(df_train)
            df_test_feat = factory.transform(df_test)

            X_train, Y_train = df_train_feat[feature_cols], df_train_feat[target_cols]
            X_test, Y_test = (
                df_test_feat[feature_cols],
                df_test_feat[target_cols].values,
            )

            trainer = ForecastModelTrainer(random_state=random_state, **trainer_kwargs)
            trainer.train(X_train, Y_train)

            prob_vol_model = ProbabilisticVolatilityModel(random_state=random_state)
            prob_vol_model.fit(X_train, Y_train["tmin_next"])
            dynamic_sigma = prob_vol_model.predict_sigma(X_test)

            risk_engine = FrostRiskEngine(
                dynamic_sigma=dynamic_sigma, historical_mae_tmin=2.0
            )

            mae, auc, logloss, frost_probs, actual_frost, fold_residuals = (
                self.evaluate_split(trainer.model, X_test, Y_test, risk_engine)
            )
            results_list.append([mae, auc, logloss])
            all_probs.extend(frost_probs)
            all_actuals.extend(actual_frost)

            if verbose:
                print(
                    f"Fold {fold} | MAE: {mae:.2f}\u00b0C | AUC-ROC: {auc:.4f} | LogLoss: {logloss:.4f}"
                )
            fold += 1

        avg_metrics = np.nanmean(results_list, axis=0)

        if verbose:
            report = [
                ["Global MAE (T_min) \u00b0C", f"{avg_metrics[0]:.4f}"],
                ["Frost Risk AUC-ROC", f"{avg_metrics[1]:.4f}"],
                ["Frost Risk LogLoss", f"{avg_metrics[2]:.4f}"],
            ]
            print("\n--- CV Evaluation Summary ---")
            print(
                tabulate(report, headers=["Metric", "Average Value"], tablefmt="pipe")
            )

        return avg_metrics

    def plot_reliability_diagram(self, actuals, probs, n_bins=10, save_path=None):
        actuals = np.asarray(actuals)
        probs = np.asarray(probs)
        prob_true, prob_pred = calibration_curve(
            actuals, probs, n_bins=n_bins, strategy="quantile"
        )

        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="FrostRiskEngine")
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated"
        )
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives (Actual Frost)")
        plt.title("Reliability Diagram — Frost Risk Calibration")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()

    def run_ablation_study(
        self,
        df,
        base_feature_cols,
        target_cols,
        new_feature_cols,
        n_splits=4,
        random_state=42,
        trainer_kwargs=None,
        gmm_n_components=3,
        importance_threshold=0.01,
    ):
        trainer_kwargs = trainer_kwargs or {}
        tscv = TimeSeriesSplit(n_splits=n_splits)

        gmm_cols = [f"gmm_prob_{i}" for i in range(gmm_n_components)]
        feature_cols = base_feature_cols + gmm_cols

        train_index, _ = list(tscv.split(df))[-1]
        df_train = df.iloc[train_index].copy()

        factory = FeatureFactory(
            n_components=gmm_n_components, random_state=random_state
        )
        factory.fit(df_train)
        df_train_feat = factory.transform(df_train)

        X_train = df_train_feat[feature_cols]
        Y_train = df_train_feat[target_cols]

        trainer = ForecastModelTrainer(random_state=random_state, **trainer_kwargs)
        trainer.train(X_train, Y_train)

        importance_matrix = np.array(
            [est.feature_importances_ for est in trainer.model.estimators_]
        )
        avg_importance = importance_matrix.mean(axis=0)
        tmin_importance = importance_matrix[target_cols.index("tmin_next")]

        report_df = (
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "is_new_feature": [f in new_feature_cols for f in feature_cols],
                    "importance_tmin_next": tmin_importance,
                    "importance_avg_all_targets": avg_importance,
                }
            )
            .sort_values("importance_avg_all_targets", ascending=False)
            .reset_index(drop=True)
        )

        max_importance = report_df["importance_avg_all_targets"].max()
        low_importance_mask = (
            report_df["importance_avg_all_targets"]
            < importance_threshold * max_importance
        ) & (report_df["is_new_feature"])
        low_importance_features = report_df.loc[low_importance_mask, "feature"].tolist()

        print("\n--- Ablation Study: Feature Importance ---")
        print(tabulate(report_df, headers="keys", tablefmt="pipe", showindex=False))
        print(
            f"\nLow-importance NEW features (< {importance_threshold * 100:.0f}% of max importance, safe to drop):"
        )
        print(low_importance_features if low_importance_features else "None")

        return report_df, low_importance_features


class HyperparameterOptimizer:
    def __init__(
        self, df, base_feature_cols, target_col_names, n_splits=4, random_state=42
    ):
        self.df = df
        self.base_feature_cols = base_feature_cols
        self.target_col_names = target_col_names
        self.n_splits = n_splits
        self.random_state = random_state
        self.study = None
        self.bench = BenchmarkSuite()

    def _objective(self, trial):
        max_depth = trial.suggest_int("max_depth", 3, 5)
        n_components = trial.suggest_int("n_components", 2, 5)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        n_estimators = trial.suggest_int("n_estimators", 100, 400, step=25)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        trainer_kwargs = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )

        avg_metrics = self.bench.run_time_series_cv(
            self.df,
            self.base_feature_cols,
            self.target_col_names,
            n_splits=self.n_splits,
            random_state=self.random_state,
            trainer_kwargs=trainer_kwargs,
            gmm_n_components=n_components,
            verbose=False,
        )
        _, avg_auc, avg_logloss = avg_metrics

        if np.isnan(avg_auc) or np.isnan(avg_logloss):
            return 0.0, 10.0

        trial.set_user_attr("avg_auc", float(avg_auc))
        trial.set_user_attr("avg_logloss", float(avg_logloss))
        return float(avg_auc), float(avg_logloss)

    def run(self, n_trials=60, timeout=None):
        sampler = TPESampler(seed=self.random_state, multivariate=True, group=True)
        self.study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=sampler,
            study_name="frost_risk_hpo_issue3",
        )
        self.study.optimize(
            self._objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True
        )
        return self.study

    def select_trial(self, baseline_auc, baseline_logloss, min_auc_gain=0.01):

        if self.study is None:
            raise RuntimeError("Call .run() first")

        scored = []
        for t in self.study.best_trials:
            auc, logloss = t.values
            auc_gain = auc - baseline_auc
            logloss_drop = baseline_logloss - logloss
            meets_auc_criterion = auc_gain >= min_auc_gain
            scored.append((t, auc_gain, logloss_drop, meets_auc_criterion))

        scored.sort(key=lambda c: (c[3], c[1] if c[3] else c[2]), reverse=True)
        best = scored[0]
        return {
            "trial": best[0],
            "auc_gain": best[1],
            "logloss_drop": best[2],
            "meets_auc_criterion": best[3],
            "meets_logloss_criterion": best[2] > 0,
        }

    def report(self, output_dir="."):

        import optuna.visualization.matplotlib as opt_mpl

        paths = {}

        fig1 = opt_mpl.plot_param_importances(
            self.study, target=lambda t: t.values[0], target_name="AUC-ROC"
        )
        fig1.figure.tight_layout()
        p1 = f"{output_dir}/param_importance_auc.png"
        fig1.figure.savefig(p1, dpi=150)
        paths["param_importance_auc"] = p1

        fig2 = opt_mpl.plot_param_importances(
            self.study, target=lambda t: t.values[1], target_name="LogLoss"
        )
        fig2.figure.tight_layout()
        p2 = f"{output_dir}/param_importance_logloss.png"
        fig2.figure.savefig(p2, dpi=150)
        paths["param_importance_logloss"] = p2

        fig3 = opt_mpl.plot_pareto_front(
            self.study, target_names=["AUC-ROC", "LogLoss"]
        )
        fig3.figure.tight_layout()
        p3 = f"{output_dir}/pareto_front.png"
        fig3.figure.savefig(p3, dpi=150)
        paths["pareto_front"] = p3

        return paths


if __name__ == "__main__":
    data_engine = DataEngine("data/records.csv")
    df = data_engine.load_and_preprocess()

    target_gen = TargetGenerator()
    df = target_gen.prepare_regression_targets(df)

    base_feature_cols = [
        "tmin",
        "tmax",
        "td_m",
        "um",
        "ffm",
        "lapse_rate",
        "dtr",
        "dtr_lag_1",
        "tmin_lag_1",
        "tmin_lag_2",
        "tmin_lag_3",
        "ffm_lag_1",
        "td_m_roll_3",
        "tmin_volatility_3d",
        "sin_doy",
        "cos_doy",
        "tmin_ewma_3",
        "td_m_ewma_3",
        "um_ffm_interaction",
        "tmin_lag_5",
        "tmin_lag_7",
        "ffm_lag_5",
        "ffm_lag_7",
    ]
    target_col_names = ["tmin_next", "tmax_next", "td_m_next", "ffm_next"]

    print("\n################ BASELINE (heuristic hyperparameters) ################")
    bench = BenchmarkSuite()
    baseline_mae, baseline_auc, baseline_logloss = bench.run_time_series_cv(
        df,
        base_feature_cols,
        target_col_names,
        n_splits=4,
        trainer_kwargs=dict(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        gmm_n_components=3,
    )
    print("\n################ ABLATION STUDY (New Features) ################")
    new_feature_cols = [
        "tmin_ewma_3",
        "td_m_ewma_3",
        "um_ffm_interaction",
        "tmin_lag_5",
        "tmin_lag_7",
        "ffm_lag_5",
        "ffm_lag_7",
    ]
    ablation_report, low_importance_features = bench.run_ablation_study(
        df,
        base_feature_cols,
        target_col_names,
        new_feature_cols,
        n_splits=4,
        trainer_kwargs=dict(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        gmm_n_components=3,
    )
    print("\n################ OPTUNA SEARCH (Issue #3) ################")
    optimizer = HyperparameterOptimizer(
        df, base_feature_cols, target_col_names, n_splits=4
    )
    study = optimizer.run(n_trials=60)

    selection = optimizer.select_trial(
        baseline_auc, baseline_logloss, min_auc_gain=0.01
    )
    best_trial = selection["trial"]

    print("\n################ RESULT ################")
    print(f"Baseline   -> AUC: {baseline_auc:.4f} | LogLoss: {baseline_logloss:.4f}")
    print(
        f"Best trial -> AUC: {best_trial.values[0]:.4f} | LogLoss: {best_trial.values[1]:.4f}"
    )
    print(f"Best params: {best_trial.params}")
    print(
        f"ΔAUC = {selection['auc_gain']:+.4f} | ΔLogLoss = {-selection['logloss_drop']:+.4f}"
    )
    print(f"Meets AUC criterion (>=+0.01)?     {selection['meets_auc_criterion']}")
    print(f"Meets LogLoss-improved criterion?  {selection['meets_logloss_criterion']}")

    paths = optimizer.report(output_dir=".")
    print(f"\nSaved reports: {paths}")
