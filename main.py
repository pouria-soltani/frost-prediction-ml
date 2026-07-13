import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, f1_score
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")


class DataEngine:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_and_preprocess(self):
        df = pd.read_csv(self.filepath, skiprows=1, parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        features = ["tmin", "tmax", "td_m", "um", "ffm"]
        df[features] = df[features].interpolate(method="linear").bfill().ffill()
        return df


class FeatureFactory:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=42)

    def engineer_features(self, df):
        df = df.copy()

        time_diff = df["datetime"].diff().dt.total_seconds() / 3600.0
        time_diff = time_diff.replace(0, np.nan).fillna(24.0)
        df["lapse_rate"] = (df["tmin"].diff() / time_diff).bfill()

        df["dtr"] = df["tmax"] - df["tmin"]
        df["dtr_lag_1"] = df["dtr"].shift(1).bfill()

        for i in range(1, 4):
            df[f"tmin_lag_{i}"] = df["tmin"].shift(i).bfill()
            df[f"ffm_lag_{i}"] = df["ffm"].shift(i).bfill()

        df["td_m_roll_3"] = df["td_m"].rolling(window=3, min_periods=1).mean()
        df["tmin_volatility_3d"] = (
            df["tmin"].rolling(window=3, min_periods=1).std().fillna(0)
        )

        df["day_of_year"] = df["datetime"].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

        gmm_cols = ["tmin", "tmax", "td_m", "um", "ffm"]
        if not hasattr(self.gmm, "means_"):
            self.gmm.fit(df[gmm_cols])

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


class FrostRuleEngine:
    @staticmethod
    def apply_rules(tmin, tmax, td_m, ffm):
        c1 = (tmin <= 0.0) & (ffm <= 2.0) & ((tmax - tmin) > 10.0)
        c2 = (tmin <= 0.0) & (ffm > 2.0)
        c3 = (tmin <= 0.0) & (td_m < tmin)

        return np.select([c1, c2, c3], [1, 2, 3], default=0)


class ForecastModelTrainer:
    def __init__(self):
        base_estimator = XGBRegressor(
            max_depth=5,
            learning_rate=0.03,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:pseudohubererror",
            random_state=42,
            n_jobs=-1,
        )
        self.model = MultiOutputRegressor(base_estimator)

    def train(self, X, y):
        critical_mask = y.iloc[:, 0] <= 2.0
        sample_weights = np.where(critical_mask, 5.0, 1.0)

        self.model.fit(X, y, sample_weight=sample_weights)
        return self.model


class BenchmarkSuite:
    def evaluate(self, df, feature_cols, target_cols, model):
        X = df[feature_cols]
        Y_true = df[target_cols].values

        Y_pred = model.predict(X)

        mae_tmin = mean_absolute_error(Y_true[:, 0], Y_pred[:, 0])
        mae_td = mean_absolute_error(Y_true[:, 2], Y_pred[:, 2])

        critical_idx = Y_true[:, 0] <= 0.0
        if np.any(critical_idx):
            mae_critical_tmin = mean_absolute_error(
                Y_true[critical_idx, 0], Y_pred[critical_idx, 0]
            )
        else:
            mae_critical_tmin = 0.0

        true_classes = FrostRuleEngine.apply_rules(
            Y_true[:, 0], Y_true[:, 1], Y_true[:, 2], Y_true[:, 3]
        )
        pred_classes = FrostRuleEngine.apply_rules(
            Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], Y_pred[:, 3]
        )

        acc = np.mean(true_classes == pred_classes)
        f1_macro = f1_score(
            true_classes, pred_classes, average="macro", zero_division=0
        )

        results = [
            ["Global MAE (T_min) °C", f"{mae_tmin:.4f}"],
            ["Critical MAE (T_min <= 0) °C", f"{mae_critical_tmin:.4f}"],
            ["Global MAE (T_dew) °C", f"{mae_td:.4f}"],
            ["Frost Rule Accuracy", f"{acc:.4f}"],
            ["Frost Rule F1-Macro", f"{f1_macro:.4f}"],
        ]

        print(tabulate(results, headers=["Metric", "Value"], tablefmt="pipe"))


if __name__ == "__main__":
    data_engine = DataEngine("data/records.csv")
    df = data_engine.load_and_preprocess()

    feature_factory = FeatureFactory(n_components=3)
    df = feature_factory.engineer_features(df)

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
    target_cols = ["tmin_next", "tmax_next", "td_m_next", "ffm_next"]

    model_trainer = ForecastModelTrainer()
    model_trainer.train(df[feature_cols], df[target_cols])

    benchmark = BenchmarkSuite()
    benchmark.evaluate(df, feature_cols, target_cols, model_trainer.model)
