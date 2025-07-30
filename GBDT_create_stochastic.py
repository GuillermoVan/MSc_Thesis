# ─────────────────────────────────────────────────────────────────────────────
#  Stochastic NGBoost model with LogNormal output
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Standard library & deps
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    make_scorer,
)
from ngboost import NGBRegressor
from ngboost.distns import LogNormal          # positive-only target
import shap
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from scipy.stats import ttest_rel

# ───────────────────────────── helper functions ──────────────────────────────
def _mape_ignore_zero(y_true, y_pred):
    """Mean Absolute Percentage Error ignoring zero targets (fraction)."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def crps_lognormal_cf(y_true, mu, sigma):
    """
    Closed-form CRPS for a LogNormal(mu, sigma²) forecast.
    Parameters
    ----------
    y_true : array-like, shape (n,)
    mu, sigma : array-like, same shape
        Latent Normal parameters.
    Returns
    -------
    crps : ndarray, shape (n,)
        CRPS per observation (lower = better).
    """
    y_true = np.asarray(y_true, dtype=float)
    mu     = np.asarray(mu,     dtype=float)
    sigma  = np.asarray(sigma,  dtype=float)

    if np.any(y_true <= 0):
        raise ValueError("CRPS formula requires y_true > 0 (LogNormal support).")

    kappa  = (np.log(y_true) - mu) / sigma
    term1  = y_true * (2 * norm.cdf(kappa) - 1)
    term2  = 2 * np.exp(mu + 0.5 * sigma**2) * (norm.cdf(kappa - sigma) - 0.5)
    return term1 - term2

# ────────────────────────────── main class ───────────────────────────────────
class GBDT_Model:
    """NGBoost model producing a LogNormal predictive distribution."""

    def __init__(
        self,
        big_csv_path: str,
        *,
        model_path_active: str = "stochastic_gbdt_active.pkl",
        model_path_inactive: str = "stochastic_gbdt_inactive.pkl",
        cv_folds: int = 5,
        random_state: int = 42,
        remaining_time_cap: float = 600.0,
        eval_start: str | pd.Timestamp = "2025-05-02",
        eval_end:   str | pd.Timestamp = "2025-05-30",
        param: dict | None = None,
        use_gridsearchcv: bool = False,
        use_kfold_insights: bool = False,
        param_grid: dict | None = None,
    ):
        self.big_csv_path       = big_csv_path
        self.eval_start         = pd.Timestamp(eval_start)
        self.eval_end           = pd.Timestamp(eval_end)
        self.model_path_active  = model_path_active
        self.model_path_inactive= model_path_inactive
        self.cv_folds           = cv_folds
        self.random_state       = random_state
        self.remaining_time_cap = remaining_time_cap
        self.param              = param or {
            "regressor__n_estimators": 200,
            "regressor__learning_rate": 0.05,
        }
        self.use_gridsearchcv   = use_gridsearchcv
        self.use_kfold_insights = use_kfold_insights
        self.param_grid         = param_grid or {
            "regressor__n_estimators": [400, 600, 800],
            "regressor__learning_rate": [0.03, 0.05, 0.07],
        }
        self.best_params: dict[str, dict | None] = {"Active": None, "Inactive": None}

        print(f"Loading full dataset from {self.big_csv_path}…")
        self._load_and_split_all_data()

    # ───────────────────────── data split ────────────────────────────────────
    def _load_and_split_all_data(self):
        df_all = pd.read_csv(self.big_csv_path, parse_dates=["START_PICKING_TS"])

        train_start, train_end = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-05-01")
        mask_train = (df_all["START_PICKING_TS"].between(train_start, train_end))
        mask_eval  = (df_all["START_PICKING_TS"].between(self.eval_start, self.eval_end))

        self.df_train_active   = df_all.loc[mask_train].copy()
        self.df_train_inactive = self.df_train_active[self.df_train_active["PLANNED_TOTE_ID"] == "START"].copy()
        df_eval_active    = df_all.loc[mask_eval].copy()
        self.df_eval_active = df_eval_active[df_eval_active["REMAINING_PICKING_TIME"] > 0]
        self.df_eval_inactive  = self.df_eval_active [self.df_eval_active ["PLANNED_TOTE_ID"] == "START"].copy()

        print("Data split complete:")
        print(f"  Training Active   : {self.df_train_active.shape[0]} rows")
        print(f"  Training Inactive : {self.df_train_inactive.shape[0]} rows")
        print(f"  Eval Active       : {self.df_eval_active.shape[0]} rows ({self.eval_start.date()} → {self.eval_end.date()})")
        print(f"  Eval Inactive     : {self.df_eval_inactive.shape[0]} rows")

    # ───────────────────────── utilities ─────────────────────────────────────
    @staticmethod
    def _get_feature_targets(df: pd.DataFrame, drop_cols: list[str]):
        X = df.drop(columns=drop_cols + ["REMAINING_PICKING_TIME"], errors="ignore")
        y = df["REMAINING_PICKING_TIME"].values
        return X, y

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        categorical = X.select_dtypes(include=["object"]).columns.tolist()
        numerical   = X.select_dtypes(include=["number"]).columns.tolist()

        preproc = ColumnTransformer(
            [("num", StandardScaler(), numerical),
             ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]
        )

        ngb = NGBRegressor(
            Dist          = LogNormal,
            n_estimators  = self.param.get("regressor__n_estimators", 200),
            learning_rate = self.param.get("regressor__learning_rate", 0.05),
            random_state  = self.random_state,
            verbose       = True,
        )

        return Pipeline([("preprocessor", preproc), ("regressor", ngb)])

    # ───────────────────────── metrics ───────────────────────────────────────
    @staticmethod
    def _evaluate_metrics(y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0.0
        return rmse, mae, mape

    # ───────────────────────── training / eval ───────────────────────────────
    def _train_and_evaluate(self, df, drop_cols, model_path, label):
        print(f"\n--- Training {label} model ---")
        df_filt = df[
            (df["REMAINING_PICKING_TIME"] > 0) &
            (df["REMAINING_PICKING_TIME"] <= self.remaining_time_cap) &
            (df["REMAINING_PICKING_TIME"].notna())
        ].copy()

        print(f"{label}: removed {df.shape[0] - df_filt.shape[0]} rows with non-positive or NaN targets.")
        X, y = self._get_feature_targets(df_filt, drop_cols)
        pipe = self._build_pipeline(X)

        t0 = time.perf_counter()

        if self.use_gridsearchcv:
            print("GridSearchCV in progress…")
            grid = GridSearchCV(
                pipe, self.param_grid,
                cv=KFold(self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring="neg_mean_squared_error",
                n_jobs=-1, verbose=1,
            )
            grid.fit(X, y)
            self.best_params[label] = grid.best_params_
            best_model = grid.best_estimator_
            print(f"Best params for {label}: {grid.best_params_}")
        else:
            best_model = pipe.fit(X, y)

        fit_seconds = time.perf_counter() - t0
        print(f"Training time for {label} model: {fit_seconds:.2f} s")

        # Optional CV diagnostics
        if self.use_kfold_insights:
            def _crps_scorer(estimator, X_val, y_val):
                pre  = estimator.named_steps["preprocessor"]
                reg  = estimator.named_steps["regressor"]
                dist = reg.pred_dist(pre.transform(X_val))
                return -crps_lognormal_cf(y_val, dist.loc, dist.scale).mean()

            scoring = {
                "mse" : "neg_mean_squared_error",
                "mae" : "neg_mean_absolute_error",
                "mape": make_scorer(_mape_ignore_zero, greater_is_better=False),
                "crps": _crps_scorer,
            }

            cv_res = cross_validate(
                best_model, X, y,
                cv=KFold(self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring=scoring,
            )
            rmse = np.sqrt(-cv_res["test_mse"])
            crps = -cv_res["test_crps"]                    # flip sign back (lower = better)
            mape = -cv_res["test_mape"] * 100              # convert from fraction to %

            print(
                f"CV {label}: "
                f"RMSE {rmse.mean():.2f}±{rmse.std(ddof=1):.2f}, "
                f"MAE {-cv_res['test_mae'].mean():.2f}, "
                f"MAPE {mape.mean():.2f}%±{mape.std(ddof=1):.2f}%, "  # <─ new
                f"CRPS {crps.mean():.3f}±{crps.std(ddof=1):.3f}"
            )

        joblib.dump(best_model, model_path)
        print(f"{label} model saved → {model_path}")

    def train_models(self):
        base_drop = [
            "PLANNED_FRAME_STACK_ID", "PLANNED_TOTE_ID", "START_PICKING_TS",
            "END_PICKING_TS", "TRUCK_TRIP_ID", "PICK_DATE"
        ]
        self._train_and_evaluate(self.df_train_active,   base_drop.copy(), self.model_path_active,   "Active")
        self._train_and_evaluate(self.df_train_inactive, base_drop + ["TIME_OF_DAY_MINS", "NR_OF_PICKERS", "NR_OF_PICKS"],
                                 self.model_path_inactive, "Inactive")

    # ───────────────────────── evaluation ────────────────────────────────────
    def _evaluate_loaded_model(self, df, drop_cols, model_path, label):
        print(f"\n--- Evaluating {label} model ---")

        df = df[df["REMAINING_PICKING_TIME"].notna()]
        X_raw, y_true = self._get_feature_targets(df, drop_cols)

        model = joblib.load(model_path)
        if not isinstance(model, Pipeline):
            raise ValueError("Loaded object is not a sklearn Pipeline.")

        preproc   = model.named_steps["preprocessor"]
        ngb_model = model.named_steps["regressor"]

        # Predict full distribution and point estimates
        dist   = ngb_model.pred_dist(preproc.transform(X_raw))
        y_pred = dist.mean()

        # ── Per-row errors ───────────────────────────────────────────────────
        err_abs = np.abs(y_pred - y_true)
        err_sq  = (y_pred - y_true) ** 2

        # MAE / RMSE (mean ± std)
        mae_mean  = err_abs.mean()
        mae_std   = err_abs.std(ddof=1)

        rmse_mean = np.sqrt(err_sq.mean())
        rmse_std  = np.sqrt(err_sq.std(ddof=1))

        # MAPE (skip zeros)
        mask = y_true != 0
        mape_vals = np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]) * 100
        mape_mean = mape_vals.mean() if mask.any() else 0.0
        mape_std  = mape_vals.std(ddof=1) if mask.any() else 0.0

        # CRPS (per observation → mean ± std)
        crps_vals = crps_lognormal_cf(y_true, dist.loc, dist.scale)
        crps_mean = crps_vals.mean()
        crps_std  = crps_vals.std(ddof=1)

        # ── Residual variance / std ───────────────────────────────────────
        residuals     = y_true - y_pred
        resid_variance = residuals.var(ddof=1)
        resid_std      = residuals.std(ddof=1)

        # ── Predicted distribution spread (mean over rows) ────────────────
        pred_std_mean  = dist.std().mean()
        pred_var_mean  = dist.var().mean()

        print(
            f"{label} Evaluation:"
            f"\n  RMSE            : {rmse_mean:.2f} ± {rmse_std:.2f}"
            f"\n  MAE             : {mae_mean:.2f} ± {mae_std:.2f}"
            f"\n  MAPE            : {mape_mean:.2f}% ± {mape_std:.2f}%"
            f"\n  CRPS            : {crps_mean:.3f} ± {crps_std:.3f}"
            f"\n  Residual Std Dev: {resid_std:.2f} min    Variance: {resid_variance:.2f} min²"
            f"\n  Pred  Std Dev   : {pred_std_mean:.2f} min    Variance: {pred_var_mean:.2f} min²"
        )

    def evaluate(self):
        base_drop = [
            "PLANNED_FRAME_STACK_ID", "PLANNED_TOTE_ID", "START_PICKING_TS",
            "END_PICKING_TS", "TRUCK_TRIP_ID", "PICK_DATE"
        ]

        df_active_eval = self.df_eval_active.copy()
        df_active_eval = df_active_eval[
            (df_active_eval['PLANNED_TOTE_ID'] == 'START') &
            (df_active_eval['REMAINING_PICKING_TIME'] <= self.remaining_time_cap)
        ]

        self._evaluate_loaded_model(
            df_active_eval[df_active_eval["REMAINING_PICKING_TIME"] <= self.remaining_time_cap],
            base_drop.copy(), self.model_path_active, "Active"
        )
        self._evaluate_loaded_model(
            self.df_eval_inactive[self.df_eval_inactive["REMAINING_PICKING_TIME"] <= self.remaining_time_cap],
            base_drop + ["TIME_OF_DAY_MINS", "NR_OF_PICKERS", "NR_OF_PICKS"],
            self.model_path_inactive, "Inactive"
        )
        
        print("\n--- Baseline: LogNormal Fit on Inactive Training Set ---")

        # Fit log-normal to training targets (requires > 0 values)
        y_train_inactive = self.df_train_inactive["REMAINING_PICKING_TIME"]
        y_train_inactive = y_train_inactive[(y_train_inactive > 0) & (y_train_inactive <= self.remaining_time_cap)]

        log_y = np.log(y_train_inactive)
        mū_train  = log_y.mean()
        σ̄_train   = log_y.std(ddof=1)

        # Compute actual mean and std (i.e. of the true target distribution)
        mean_true = y_train_inactive.mean()
        std_true  = y_train_inactive.std(ddof=1)

        # Evaluation set
        df_eval = self.df_eval_inactive[self.df_eval_inactive["REMAINING_PICKING_TIME"] <= self.remaining_time_cap]
        y_true  = df_eval["REMAINING_PICKING_TIME"].values

        # Predict mean and std of same LogNormal(μ̄, σ̄) for all rows
        dist_mean = np.exp(mū_train + 0.5 * σ̄_train**2)
        dist_std  = np.sqrt((np.exp(σ̄_train**2) - 1) * np.exp(2 * mū_train + σ̄_train**2))

        y_pred = np.full_like(y_true, dist_mean)

        err_abs = np.abs(y_pred - y_true)
        err_sq  = (y_pred - y_true) ** 2

        mae_mean = err_abs.mean()
        mae_std  = err_abs.std(ddof=1)

        rmse_mean = np.sqrt(err_sq.mean())
        rmse_std  = np.sqrt(err_sq.std(ddof=1))

        mape_vals = np.abs((y_pred - y_true) / y_true) * 100
        mape_mean = mape_vals.mean()
        mape_std  = mape_vals.std(ddof=1)

        crps_vals = crps_lognormal_cf(
            y_true,
            np.full_like(y_true, mū_train),
            np.full_like(y_true, σ̄_train)
        )
        crps_mean = crps_vals.mean()
        crps_std  = crps_vals.std(ddof=1)

        # residual variance / std for baseline
        residuals_bl = y_true - y_pred
        resid_var_bl = residuals_bl.var(ddof=1)
        resid_std_bl = residuals_bl.std(ddof=1)

        print(
            f"Inactive Baseline (μ̄={mean_true:.2f}, σ̄={std_true:.2f}):"
            f"\n  RMSE            : {rmse_mean:.2f} ± {rmse_std:.2f}"
            f"\n  MAE             : {mae_mean:.2f} ± {mae_std:.2f}"
            f"\n  MAPE            : {mape_mean:.2f}% ± {mape_std:.2f}%"
            f"\n  CRPS            : {crps_mean:.3f} ± {crps_std:.3f}"
            f"\n  Residual Std Dev: {resid_std_bl:.2f} min    Variance: {resid_var_bl:.2f} min²"
            f"\n  Pred  Std Dev   : {dist_std:.2f} min    Variance: {(dist_std**2):.2f} min²"
        )

        # ToDo: perform statistical significance test (Delta and p-value) to see if the expected (MAPE, MAE, RMSE, CRPS) of the baseline is actually worse than the model's expected (MAPE, MAE, RMSE, CRPS)
        print("\n--- Statistical significance vs. LogNormal Inactive Baseline ---")

        def paired_ttest_and_delta(metric_name, baseline_vals, model_vals, is_rmse=False):
            # mean values for reporting
            base_mean = np.mean(baseline_vals)
            model_mean = np.mean(model_vals)

            # If comparing RMSE, apply sqrt to MSE mean for both
            if is_rmse:
                base_val = np.sqrt(base_mean)
                model_val = np.sqrt(model_mean)
                delta = base_val - model_val
            else:
                delta = base_mean - model_mean

            # paired t-test (one-sided: baseline > model)
            t_stat, p_val = ttest_rel(baseline_vals, model_vals, alternative='greater')

            print(f"Δ {metric_name:5s}: {delta:>7.3f}   (p = {p_val:.2e})")

        # Gather per-row values from baseline evaluation above
        y_true_eval = y_true
        y_pred_baseline = y_pred
        mu_bl = np.full_like(y_true_eval, mū_train)
        sigma_bl = np.full_like(y_true_eval, σ̄_train)

        # Errors from baseline (already computed):
        abs_errors_baseline = np.abs(y_pred_baseline - y_true_eval)
        sq_errors_baseline  = (y_pred_baseline - y_true_eval) ** 2
        mape_vals_baseline  = np.abs((y_pred_baseline - y_true_eval) / y_true_eval) * 100
        crps_vals_baseline  = crps_lognormal_cf(y_true_eval, mu_bl, sigma_bl)

        # Errors from inactive model:
        df_eval_inact = self.df_eval_inactive[self.df_eval_inactive["REMAINING_PICKING_TIME"] <= self.remaining_time_cap]
        df_eval_inact = df_eval_inact[df_eval_inact["REMAINING_PICKING_TIME"] > 0]
        drop_cols_inactive = [
            "PLANNED_FRAME_STACK_ID", "PLANNED_TOTE_ID", "START_PICKING_TS",
            "END_PICKING_TS", "TRUCK_TRIP_ID", "PICK_DATE",
            "TIME_OF_DAY_MINS", "NR_OF_PICKERS", "NR_OF_PICKS"
        ]
        X_raw_inact, y_model = self._get_feature_targets(df_eval_inact, drop_cols_inactive)
        pipe_inact = joblib.load(self.model_path_inactive)
        pre_inact = pipe_inact.named_steps["preprocessor"]
        reg_inact = pipe_inact.named_steps["regressor"]
        dist_inact = reg_inact.pred_dist(pre_inact.transform(X_raw_inact))
        y_pred_model = dist_inact.mean()

        abs_errors_model = np.abs(y_pred_model - y_model)
        sq_errors_model  = (y_pred_model - y_model) ** 2
        mape_vals_model  = np.abs((y_pred_model - y_model) / y_model) * 100
        crps_vals_model  = crps_lognormal_cf(y_model, dist_inact.loc, dist_inact.scale)

        # Ensure lengths match for fair paired testing
        assert len(abs_errors_baseline) == len(abs_errors_model), "Mismatched evaluation sizes."

        # Report Δ and p-values
        paired_ttest_and_delta("MAE",   abs_errors_baseline, abs_errors_model)
        paired_ttest_and_delta("MAPE",  mape_vals_baseline,  mape_vals_model)
        paired_ttest_and_delta("RMSE",  sq_errors_baseline,  sq_errors_model, is_rmse=True)
        paired_ttest_and_delta("CRPS",  crps_vals_baseline,  crps_vals_model)


    def plot_active_vs_running_completion(self, n_bins: int = 20):
        """
        For the ACTIVE model, plot in a single figure:
        (a) Mean CRPS with shaded ±1 std (68%) and ±2 std (95%) intervals
        (b) Mean predicted Std-Dev with shaded ±1 std (68%) and ±2 std (95%) intervals
        Transparent dots are excluded.

        Parameters
        ----------
        n_bins : int, default=20
            Number of equal-width bins on PICKING_RUNNING_COMPLETION ∈ [0, 1),
            with the first bin capturing exactly 0.0.
        """
        import matplotlib.colors as mcolors

        df = self.df_eval_active.copy()
        df = df[df["REMAINING_PICKING_TIME"] <= self.remaining_time_cap]

        if "PICKING_RUNNING_COMPLETION" not in df.columns:
            raise ValueError("Column 'PICKING_RUNNING_COMPLETION' not found.")

        # ── Predict LogNormal(μ, σ) distributions ─────────────────────────────
        pipe = joblib.load(self.model_path_active)
        pre = pipe.named_steps["preprocessor"]
        ngb = pipe.named_steps["regressor"]

        X_raw = df.drop(columns=["REMAINING_PICKING_TIME"], errors="ignore")
        dist = ngb.pred_dist(pre.transform(X_raw))

        μ, σ = dist.loc, dist.scale
        df["crps"] = crps_lognormal_cf(df["REMAINING_PICKING_TIME"].values, μ, σ)
        df["pred_std_minutes"] = np.sqrt((np.exp(σ**2) - 1) * np.exp(2 * μ + σ**2))

        # ── Custom binning with a separate bin for 0.0 ────────────────────────
        eps = 1e-8
        df_zero = df[df["PICKING_RUNNING_COMPLETION"] == 0.0].copy()
        df_nonzero = df[df["PICKING_RUNNING_COMPLETION"] > 0.0].copy()

        bins_nonzero = np.linspace(0.0, 1.0, n_bins + 1)[1:]  # exclude 0.0
        bins_nonzero = np.insert(bins_nonzero, 0, 0.0 + eps)
        df_zero["bin"] = pd.IntervalIndex.from_tuples([(0.0, 0.0)], closed="both")[0]
        df_nonzero["bin"] = pd.cut(df_nonzero["PICKING_RUNNING_COMPLETION"], bins=bins_nonzero, include_lowest=True)

        df_plot = pd.concat([df_zero, df_nonzero], axis=0)
        df_plot["bin"] = pd.Categorical(df_plot["bin"])
        bin_left_edges = df_plot["bin"].cat.categories.map(lambda iv: iv.left)

        # Compute bin width (assumes uniform bin width)
        if len(bin_left_edges) >= 2:
            bin_width = bin_left_edges[1] - bin_left_edges[0]
        else:
            bin_width = 1.0 / n_bins  # fallback

        x_max = bin_left_edges[-1] + bin_width

        # ── Aggregate stats per bin ───────────────────────────────────────────
        def agg_stats(series):
            return pd.DataFrame({
                "mean": series.groupby(df_plot["bin"]).mean(),
                "std": series.groupby(df_plot["bin"]).std(),
                "low": series.groupby(df_plot["bin"]).apply(lambda x: np.percentile(x, 2.5)),
                "high": series.groupby(df_plot["bin"]).apply(lambda x: np.percentile(x, 97.5)),
            })

        crps_stats = agg_stats(df_plot["crps"])
        std_stats = agg_stats(df_plot["pred_std_minutes"])

        # ── Plot ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        base_color = mcolors.to_rgb("C0")
        std_color = [c * 0.6 for c in base_color]
        ci_color = base_color

        # ── CRPS Plot ─────────────────────────────────────────────────────────
        axes[0].plot(bin_left_edges, crps_stats["mean"], color="C0", marker="o", linewidth=2, label="Mean CRPS")
        axes[0].fill_between(bin_left_edges,
                            crps_stats["mean"] - crps_stats["std"],
                            crps_stats["mean"] + crps_stats["std"],
                            color=std_color, alpha=0.2, label="±1 Std Dev (68%)")
        axes[0].fill_between(bin_left_edges,
                            crps_stats["low"], crps_stats["high"],
                            color=ci_color, alpha=0.35, label="±2 Std Dev (95%)")
        axes[0].set_xlabel("Running Completion")
        axes[0].set_xlim(left=0.0, right=x_max)
        axes[0].set_ylabel("CRPS [min]")
        axes[0].set_title("CRPS vs. Running Completion")
        axes[0].grid(True)
        axes[0].legend()

        # ── Std Dev Plot ──────────────────────────────────────────────────────
        axes[1].plot(bin_left_edges, std_stats["mean"], color="C0", marker="o", linewidth=2, label="Mean σ")
        axes[1].fill_between(bin_left_edges,
                            std_stats["mean"] - std_stats["std"],
                            std_stats["mean"] + std_stats["std"],
                            color=std_color, alpha=0.2, label="±1 Std Dev (68%)")
        axes[1].fill_between(bin_left_edges,
                            std_stats["low"], std_stats["high"],
                            color=ci_color, alpha=0.35, label="±2 Std Dev (95%)")
        axes[1].set_xlabel("Running Completion")
        axes[1].set_xlim(left=0.0, right=x_max)
        axes[1].set_ylabel("Predicted Std-Dev [min]")
        axes[1].set_title("σ vs. Running Completion")
        axes[1].grid(True)
        axes[1].legend()

        fig.suptitle("Active NGBoost Model – Performance vs. Running Completion", y=1.02)
        fig.tight_layout()
        plt.show()
    
    # ToDo: maybe add feature importance on std results as well

        

# ───────────────────────── single-row inference ─────────────────────────────
def stochastic_gbdt_inference(row_idx: int = 0, use_active_model: bool = True):
    "Inference performance and check on stochastic GBDT model"

    model_path = "stochastic_gbdt_active.pkl" if use_active_model else "stochastic_gbdt_inactive.pkl"
    df_path = "Data/Training/Features_all_dates.csv"
    label = "Active" if use_active_model else "Inactive"

    pipe = joblib.load(model_path)
    df = pd.read_csv(df_path, parse_dates=["START_PICKING_TS"])

    # Filter to evaluation period
    eval_start = pd.Timestamp("2025-05-21")
    eval_end   = pd.Timestamp("2025-05-27")
    df = df[(df["START_PICKING_TS"] >= eval_start) & (df["START_PICKING_TS"] <= eval_end)]

    if use_active_model:
        df_eval = df[df["REMAINING_PICKING_TIME"] > 0].copy()
    else:
        df_eval = df[(df["REMAINING_PICKING_TIME"] > 0) & (df["PLANNED_TOTE_ID"] == "START")].copy()
        df_eval = df_eval.drop(columns=["TIME_OF_DAY_MINS", "NR_OF_PICKERS", "NR_OF_PICKS"], errors="ignore")

    if not 0 <= row_idx < len(df_eval):
        raise IndexError(f"Row index {row_idx} out of bounds for {label} eval set with {len(df_eval)} rows.")

    row = df_eval.iloc[row_idx]
    y_true = row["REMAINING_PICKING_TIME"]
    X = pd.DataFrame([row.drop("REMAINING_PICKING_TIME")])

    preproc = pipe.named_steps["preprocessor"]
    ngb     = pipe.named_steps["regressor"]

    # ── Measure prediction time ──
    t0 = time.perf_counter()
    X_trans = preproc.transform(X)
    dist    = ngb.pred_dist(X_trans)
    prediction_time = time.perf_counter() - t0

    mu, sigma = dist.loc[0], dist.scale[0]
    mean_, std_ = dist.mean()[0], dist.std()[0]
    crps_val   = crps_lognormal_cf(np.array([y_true]), np.array([mu]), np.array([sigma]))[0]

    print(
        f"{label} NGBoost inference (row {row_idx}):"
        f"\n  ▸ true remaining time     : {y_true:.2f}"
        f"\n  ▸ predicted μ, σ (log-space): {mu:.4f}, {sigma:.4f}"
        f"\n  ▸ moments (mean, std)       : {mean_:.2f}, {std_:.2f}"
        f"\n  ▸ CRPS                      : {crps_val:.3f}"
        f"\n  ▸ Prediction time           : {prediction_time:.4f} seconds"
    )

# ───────────────────────── script entrypoint ────────────────────────────────
if __name__ == "__main__":
    model = GBDT_Model(
        big_csv_path="Data/Training/Features_all_dates.csv",
        use_gridsearchcv=False,
        use_kfold_insights=True,
        eval_start="2025-05-21",
        eval_end  ="2025-05-27",
        param_grid={
            "regressor__n_estimators": [100, 200, 300],
            "regressor__learning_rate": [0.05, 0.1, 0.3],
        },
    )

    # Uncomment if you need to train
    #model.train_models()

    model.evaluate()

    model.plot_active_vs_running_completion()

    stochastic_gbdt_inference(row_idx=100, use_active_model=False)  # change index as needed
