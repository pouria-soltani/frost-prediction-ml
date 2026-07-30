import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import norm
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")


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
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=42)
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
            df[f"tmin_lag_{i}"] = df["tmin"].shift(i)
            df[f"ffm_lag_{i}"] = df["ffm"].shift(i)

        df["td_m_roll_3"] = df["td_m"].rolling(window=3, min_periods=1).mean()
        df["tmin_volatility_3d"] = (
            df["tmin"].rolling(window=3, min_periods=1).std()
        )
        

        df["day_of_year"] = df["datetime"].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

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
    def __init__(self, historical_mae_tmin=2.0):
        self.sigma = historical_mae_tmin * np.sqrt(np.pi / 2.0)

    def calculate_risk_probability(self, predicted_tmin, threshold=0.0):
        z_scores = (threshold - predicted_tmin) / self.sigma
        return norm.cdf(z_scores)


class ForecastModelTrainer:
    def __init__(self):
        base_estimator = XGBRegressor(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
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

        return mae_tmin, auc, logloss

    def run_time_series_cv(self, df, feature_cols, target_cols, n_splits=4):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        risk_engine = FrostRiskEngine(historical_mae_tmin=2.0)

        results_list = []
        fold = 1

        print("\n=== Probabilistic Risk Engine Fold-by-Fold Analysis ===")
        for train_index, test_index in tscv.split(df):
            df_train = df.iloc[train_index].copy()
            df_test = df.iloc[test_index].copy()

            factory = FeatureFactory(n_components=3)
            factory.fit(df_train)
            df_train_feat = factory.transform(df_train)
            df_test_feat = factory.transform(df_test)

            X_train, Y_train = df_train_feat[feature_cols], df_train_feat[target_cols]
            X_test, Y_test = (
                df_test_feat[feature_cols],
                df_test_feat[target_cols].values,
            )

            trainer = ForecastModelTrainer()
            trainer.train(X_train, Y_train)

            mae, auc, logloss = self.evaluate_split(
                trainer.model, X_test, Y_test, risk_engine
            )
            results_list.append([mae, auc, logloss])

            print(
                f"Fold {fold} | MAE: {mae:.2f}\u00b0C | AUC-ROC: {auc:.4f} | LogLoss: {logloss:.4f}"
            )
            fold += 1

        avg_metrics = np.nanmean(results_list, axis=0)

        report = [
            ["Global MAE (T_min) \u00b0C", f"{avg_metrics[0]:.4f}"],
            ["Frost Risk AUC-ROC", f"{avg_metrics[1]:.4f}"],
            ["Frost Risk LogLoss", f"{avg_metrics[2]:.4f}"],
        ]
        print("\n--- Final Production Pipeline Evaluation ---")
        print(tabulate(report, headers=["Metric", "Average Value"], tablefmt="pipe"))


if __name__ == "__main__":
    data_engine = DataEngine("data/records.csv")
    df = data_engine.load_and_preprocess()

    target_gen = TargetGenerator()
    df = target_gen.prepare_regression_targets(df)

    feature_cols = [
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
        "gmm_prob_0",
        "gmm_prob_1",
        "gmm_prob_2",
    ]
    target_col_names = ["tmin_next", "tmax_next", "td_m_next", "ffm_next"]

    benchmark = BenchmarkSuite()
    benchmark.run_time_series_cv(df, feature_cols, target_col_names, n_splits=4)