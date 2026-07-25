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
    def __init__(self, n_components=3, random_state=42):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=random_state)
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
    """
    احتمال ریسک یخ‌زدگی را با فرض گاوسی محاسبه می‌کند، اما σ دیگر نه یک
    عدد ثابت است و نه یک پروکسی post-hoc — مستقیماً از
    ProbabilisticVolatilityModel (که بر پایه‌ی NGBoost ساخته شده) گرفته
    می‌شود: یک σ اختصاصی هر نمونه که از دل بهینه‌سازی مشترک μ/σ روی
    Log-Likelihood توزیع نرمال به دست آمده، نه یک حدس جداگانه‌ی بعد از
    واقعه (مثل VolatilityModel قبلی) یا یک اصلاح shape بعد از آن (مثل
    KDE قبلی). چون این σ در همان فولد و از داده‌ی آموزشی همان فولد
    یاد گرفته می‌شود، دیگر نیازی به walk-forward history یا cold-start
    fallback برای فولد اول هم نیست.

    اگر dynamic_sigma به هر دلیلی در دسترس نباشد (مثلاً مدل fit نشده)،
    به فرمول ثابت قدیمی (sigma = MAE * sqrt(pi/2)) برمی‌گردد، صرفاً
    به‌عنوان یک safety net.
    """

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
    """
    جایگزین VolatilityModel + error_kde قبلی.

    این دقیقاً "Dual Regression" پیشنهادی issue است: به‌جای این‌که یک
    مدل کمکی جدا روی |residual|ی فولدهای *قبلی* فیت بشه (walk-forward،
    با تأخیر و نیازمند حجم تاریخچه‌ی کافی برای شروع)، از NGBoost با
    توزیع Normal استفاده می‌کنیم که μ و σ را با هم و مستقیماً از طریق
    natural gradient boosting روی Log-Likelihood یاد می‌گیرد — یعنی σ
    از اول برای calibration بهینه شده، نه یک حدس جداگانه‌ی بعد از مدل
    اصلی.

    چون این مدل مثل ForecastModelTrainer روی داده‌ی آموزشی *همان* فولد
    فیت می‌شود (نه تاریخچه‌ی فولدهای قبلی)، مشکل cold-start فولد اول و
    نیاز به نگه‌داشتن تاریخچه‌ی طولانی هم به‌طور کامل حذف می‌شود.
    """

    def __init__(self, n_estimators=300, learning_rate=0.02, floor_frac=0.3,
                 val_frac=0.15, early_stopping_rounds=25, random_state=42):
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

        # اعتبارسنجی زمانی (نه تصادفی): آخرین بخش داده‌ی آموزشی به‌عنوان
        # validation نگه داشته می‌شود تا early stopping از overfitting
        # روی فولدهای کوچک اولیه جلوگیری کند — همان چیزی که باعث
        # LogLoss=1.12 در فولد اول شد (سیگمای بیش‌ازحد کوچک و
        # بیش‌اطمینان روی داده‌ی کم).
        if n - n_val >= 30:
            X_tr, X_val = X.iloc[: n - n_val], X.iloc[n - n_val :]
            y_tr, y_val = y_tmin.iloc[: n - n_val], y_tmin.iloc[n - n_val :]
            self.model.fit(
                X_tr, y_tr,
                X_val=X_val, Y_val=y_val,
                early_stopping_rounds=self.early_stopping_rounds,
            )
            in_sample_pred = self.model.predict(X_val)
            in_sample_resid = y_val.values - in_sample_pred
        else:
            self.model.fit(X, y_tmin)
            in_sample_pred = self.model.predict(X)
            in_sample_resid = y_tmin.values - in_sample_pred

        self.is_fitted = True

        # floor برای جلوگیری از سیگمای بیش‌ازحد کوچک: حتی با early
        # stopping، ممکن است مدل برای چند نمونه‌ی خاص بیش‌اعتماد بماند؛
        # این floor یک سقف پایین امن بر اساس پراکندگی واقعی خطای
        # out-of-sample (نه in-sample خام) می‌گذارد.
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
    def __init__(self, random_state=42):
        base_estimator = XGBRegressor(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
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

        # خطای واقعی out-of-sample این فولد، که قراره به فولد بعدی
        # داده بشه تا KDE بدون leakage فیت بشه (walk-forward)
        residuals = Y_test[:, 0] - Y_pred[:, 0]

        return mae_tmin, auc, logloss, frost_probs, actual_frost, residuals

    def run_time_series_cv(self, df, feature_cols, target_cols, n_splits=4, random_state=42):
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results_list = []
        all_probs, all_actuals = [], []

        fold = 1
        print("\n=== Probabilistic Risk Engine Fold-by-Fold Analysis ===")
        for train_index, test_index in tscv.split(df):
            df_train = df.iloc[train_index].copy()
            df_test = df.iloc[test_index].copy()

            factory = FeatureFactory(n_components=3, random_state=random_state)
            factory.fit(df_train)
            df_train_feat = factory.transform(df_train)
            df_test_feat = factory.transform(df_test)

            X_train, Y_train = df_train_feat[feature_cols], df_train_feat[target_cols]
            X_test, Y_test = (
                df_test_feat[feature_cols],
                df_test_feat[target_cols].values,
            )

            trainer = ForecastModelTrainer(random_state=random_state)
            trainer.train(X_train, Y_train)

            # NGBoost مستقیماً روی داده‌ی آموزشی همین فولد فیت می‌شود —
            # برخلاف VolatilityModel قبلی، نیازی به تاریخچه‌ی فولدهای
            # قبلی یا cold-start fallback نیست؛ از فولد اول هم σ
            # اختصاصی هر نمونه در دسترس است.
            prob_vol_model = ProbabilisticVolatilityModel(random_state=random_state)
            prob_vol_model.fit(X_train, Y_train["tmin_next"])
            dynamic_sigma = prob_vol_model.predict_sigma(X_test)

            risk_engine = FrostRiskEngine(
                dynamic_sigma=dynamic_sigma,
                historical_mae_tmin=2.0,
            )

            mae, auc, logloss, frost_probs, actual_frost, fold_residuals = (
                self.evaluate_split(trainer.model, X_test, Y_test, risk_engine)
            )
            results_list.append([mae, auc, logloss])
            all_probs.extend(frost_probs)
            all_actuals.extend(actual_frost)

            engine_mode = "NGBoost per-sample sigma (dual regression)" if dynamic_sigma is not None else "Static Gaussian (fallback)"

            diagnostic = ""
            if dynamic_sigma is not None:
                actual_abs_error = np.abs(fold_residuals)
                corr = np.corrcoef(dynamic_sigma, actual_abs_error)[0, 1]
                diagnostic = f" | corr(predicted_sigma, actual|error|)={corr:.3f}"

            print(
                f"Fold {fold} | MAE: {mae:.2f}°C | AUC-ROC: {auc:.4f} | "
                f"LogLoss: {logloss:.4f} | RiskEngine: {engine_mode}{diagnostic}"
            )
            fold += 1

        avg_metrics = np.nanmean(results_list, axis=0)

        report = [
            ["Global MAE (T_min) °C", f"{avg_metrics[0]:.4f}"],
            ["Frost Risk AUC-ROC", f"{avg_metrics[1]:.4f}"],
            ["Frost Risk LogLoss", f"{avg_metrics[2]:.4f}"],
        ]
        print(f"\n--- Final Production Pipeline Evaluation ---")
        print(tabulate(report, headers=["Metric", "Average Value"], tablefmt="pipe"))

        self.plot_reliability_diagram(all_actuals, all_probs)

        return avg_metrics


    def plot_reliability_diagram(self, actuals, probs, n_bins=10):
        actuals = np.asarray(actuals)
        probs = np.asarray(probs)

        prob_true, prob_pred = calibration_curve(
            actuals, probs, n_bins=n_bins, strategy="quantile"
        )

        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2,
                 label="FrostRiskEngine (NGBoost dual regression)")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray",
                 label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives (Actual Frost)")
        plt.title("Reliability Diagram — Frost Risk Calibration")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


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







    # این اسکریپت رو به‌جای بلوک "if __name__" در frost_risk_ngboost.py
# (یا notebook) اجرا کنید. تمام کلاس‌های آن فایل باید از قبل import/
# تعریف شده باشند (یعنی همان df, feature_cols, target_col_names که در
# main.py/notebook آماده کرده‌اید).
#
# هدف: مطمئن شویم بهبود ~2.2% LogLoss که با seed=42 دیدیم، یک اتفاق
# (artifact) وابسته به آن seed خاص نیست. این نسخه واقعاً seed را به
# FeatureFactory (GMM)، ForecastModelTrainer (XGB) و
# ProbabilisticVolatilityModel (NGBoost) پاس می‌دهد.

import numpy as np

BASELINE_LOGLOSS = 0.1890  # میانگین بیس‌لاین قبلی (فقط سیگمای ثابت)
BASELINE_AUC = 0.9724      # میانگین AUC بیس‌لاین قبلی

def run_multi_seed_check(df, feature_cols, target_col_names, seeds=(42, 7, 123, 2024, 99)):
    results = {}
    benchmark = BenchmarkSuite()  # noqa: F821  (از frost_risk_ngboost.py)

    for seed in seeds:
        print(f"\n########## seed={seed} ##########")
        avg_metrics = benchmark.run_time_series_cv(
            df, feature_cols, target_col_names, n_splits=4, random_state=seed
        )
        mae, auc, logloss = avg_metrics
        results[seed] = {"mae": mae, "auc": auc, "logloss": logloss}

    loglosses = [v["logloss"] for v in results.values()]
    aucs = [v["auc"] for v in results.values()]

    print("\n=== خلاصه‌ی پایداری روی seedهای مختلف ===")
    for seed, v in results.items():
        flag_ll = "✅" if v["logloss"] < BASELINE_LOGLOSS else "⚠️"
        flag_auc = "✅" if v["auc"] >= BASELINE_AUC - 0.005 else "⚠️"
        print(f"seed={seed:>5} | LogLoss={v['logloss']:.4f} {flag_ll} | "
              f"AUC={v['auc']:.4f} {flag_auc}")

    print(f"\nLogLoss: میانگین={np.mean(loglosses):.4f} | "
          f"انحراف‌معیار={np.std(loglosses):.4f} | "
          f"بازه=[{min(loglosses):.4f}, {max(loglosses):.4f}]")
    print(f"AUC-ROC: میانگین={np.mean(aucs):.4f} | "
          f"انحراف‌معیار={np.std(aucs):.4f} | "
          f"بازه=[{min(aucs):.4f}, {max(aucs):.4f}]")

    n_pass_ll = sum(1 for ll in loglosses if ll < BASELINE_LOGLOSS)
    print(f"\n{n_pass_ll}/{len(seeds)} seed از {len(seeds)} تا زیر بیس‌لاین LogLoss "
          f"({BASELINE_LOGLOSS}) ماندند.")
    print("معیار پیشنهادی برای بستن issue: حداقل 4 از 5 seed باید زیر بیس‌لاین بمانند "
          "(نه فقط میانگین کلی) تا مطمئن شویم بهبود به یک seed خاص وابسته نیست.")

    return results


# نمونه‌ی استفاده (بعد از آماده‌سازی df, feature_cols, target_col_names):
run_multi_seed_check(df, feature_cols, target_col_names)