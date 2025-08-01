import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    make_scorer
)
import shap
import matplotlib.pyplot as plt
import time
from scipy.stats import permutation_test
from scipy.stats import ttest_rel
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def _mape_ignore_zero(y_true, y_pred):
    """
    Mean Absolute Percentage Error ignoring zero targets.
    Returns fraction (not percentage).
    """
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


class GBDT_Model:
    def __init__(
        self,
        big_csv_path: str,
        model_path_active: str = 'gbdt_active.pkl',
        model_path_inactive: str = 'gbdt_inactive.pkl',
        cv_folds: int = 5,
        random_state: int = 42,
        remaining_time_cap: float = 600.0,
        param: dict = None,
        use_gridsearchcv: bool = False,
        use_kfold_insights: bool = False,
        param_grid: dict = None
    ):
        """
        big_csv_path: path to the single CSV containing all data
        The data will be split into:
          - Training active:    2025-01-01 <= START_PICKING_TS <= 2025-05-01
          - Training inactive:  same as active, but PLANNED_TOTE_ID == 'START'
          - Evaluation active:  2025-05-12 <= START_PICKING_TS <= 2025-05-18
          - Evaluation inactive: same as eval active, but PLANNED_TOTE_ID == 'START'
        use_gridsearchcv: if True, performs hyperparameter tuning via GridSearchCV.
        use_kfold_insights: if True, performs k-fold cross-validation to report average metrics (RMSE, MAE, MAPE) with std.
        param: default regressor parameters when not tuning.
        param_grid: grid for tuning; ignored when use_gridsearchcv=False.
        """
        self.big_csv_path = big_csv_path
        self.model_path_active = model_path_active
        self.model_path_inactive = model_path_inactive
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.remaining_time_cap = remaining_time_cap
        # default parameters when not using GridSearchCV
        self.param = param or {'regressor__max_iter': 200, 'regressor__learning_rate': 0.1}
        self.use_gridsearchcv = use_gridsearchcv
        self.use_kfold_insights = use_kfold_insights
        # grid for hyperparameter tuning
        self.param_grid = param_grid or {
            'regressor__max_iter': [100, 200],
            'regressor__learning_rate': [0.05, 0.1]
        }
        self.best_params = {'Active': None, 'Inactive': None}

        # Load the big CSV once and split into four DataFrames
        print(f"Loading full dataset from {self.big_csv_path}...")
        self._load_and_split_all_data()

    def _load_and_split_all_data(self):
        """
        Load the single CSV and split into:
          - df_train_active
          - df_train_inactive
          - df_eval_active
          - df_eval_inactive
        Assumes 'START_PICKING_TS' column is present and in a parseable datetime format,
        and 'PLANNED_TOTE_ID' is present.
        """
        df_all = pd.read_csv(self.big_csv_path, parse_dates=['START_PICKING_TS'])
        # Ensure timestamp column is datetime
        if not np.issubdtype(df_all['START_PICKING_TS'].dtype, np.datetime64):
            df_all['START_PICKING_TS'] = pd.to_datetime(df_all['START_PICKING_TS'])

        # Define date ranges
        train_start = pd.Timestamp('2025-01-01')
        train_end = pd.Timestamp('2025-05-01')
        eval_start = pd.Timestamp('2025-05-21')
        eval_end = pd.Timestamp('2025-05-27')

        # Training Active: 2025-01-01 <= START_PICKING_TS <= 2025-05-01
        mask_train_active = (df_all['START_PICKING_TS'] >= train_start) & \
                            (df_all['START_PICKING_TS'] <= train_end)
        self.df_train_active = df_all.loc[mask_train_active].copy()

        # Training Inactive: same rows as train_active, but PLANNED_TOTE_ID == 'START'
        self.df_train_inactive = self.df_train_active[
            self.df_train_active['PLANNED_TOTE_ID'] == 'START'
        ].copy()

        # Evaluation Active: 2025-05-12 <= START_PICKING_TS <= 2025-05-18
        mask_eval_active = (df_all['START_PICKING_TS'] >= eval_start) & \
                           (df_all['START_PICKING_TS'] <= eval_end)
        self.df_eval_active = df_all.loc[mask_eval_active].copy()

        # Evaluation Inactive: same rows as eval_active, but PLANNED_TOTE_ID == 'START'
        self.df_eval_inactive = self.df_eval_active[
            self.df_eval_active['PLANNED_TOTE_ID'] == 'START'
        ].copy()

        print(f"Data split complete:")
        print(f"  Training Active   : {self.df_train_active.shape[0]} rows")
        print(f"  Training Inactive : {self.df_train_inactive.shape[0]} rows")
        print(f"  Eval Active       : {self.df_eval_active.shape[0]} rows")
        print(f"  Eval Inactive     : {self.df_eval_inactive.shape[0]} rows")

    def _get_feature_targets(self, df: pd.DataFrame, drop_cols: list):
        X = df.drop(columns=drop_cols + ['REMAINING_PICKING_TIME'], errors='ignore')
        y = df['REMAINING_PICKING_TIME'].values
        return X, y

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        categorical = X.select_dtypes(include=['object']).columns.tolist()
        numerical = X.select_dtypes(include=['number']).columns.tolist()
        preproc = ColumnTransformer([
            ('num', StandardScaler(), numerical),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
        ])
        booster = HistGradientBoostingRegressor(
            max_iter=self.param['regressor__max_iter'],
            learning_rate=self.param['regressor__learning_rate'],
            max_depth=5,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state
        )
        pipe = Pipeline([
            ('preprocessor', preproc),
            ('regressor', booster)
        ])
        return pipe

    def _evaluate_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mask = y_true != 0
        mape = (np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        ) if mask.any() else 0.0
        return rmse, mae, mape

    def _train_and_evaluate(
        self,
        df: pd.DataFrame,
        drop_cols: list,
        model_path: str,
        label: str
    ):
        print(f"\n--- Training {label} model ---")
        # Apply remaining_time_cap filter
        df_filtered = df[df['REMAINING_PICKING_TIME'] <= self.remaining_time_cap].copy()
        print(f"{label} data shape after remaining_time_cap: {df_filtered.shape}")

        X, y = self._get_feature_targets(df_filtered, drop_cols)
        pipe = self._build_pipeline(X)

        start_time = time.time()

        if self.use_gridsearchcv:
            print("Starting GridSearchCV for tuning...")
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=self.param_grid,
                cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X, y)
            self.best_params[label] = grid.best_params_
            best_model = grid.best_estimator_
        else:
            print("Fitting histogram GBDT...")
            best_model = pipe.fit(X, y)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Training time for {label} model: {duration:.2f} seconds")

        # Cross-validation insights
        if self.use_kfold_insights:
            print(f"Performing {self.cv_folds}-fold cross-validation for insights...")
            scoring = {
                'mse': 'neg_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'mape': make_scorer(_mape_ignore_zero, greater_is_better=False)
            }
            cv_results = cross_validate(
                best_model,
                X,
                y,
                cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                return_train_score=False
            )
            # compute per-fold metrics
            rmse_folds = np.sqrt(-cv_results['test_mse'])
            mae_folds = -cv_results['test_mae']
            mape_folds = -cv_results['test_mape']

            avg_rmse = rmse_folds.mean()
            std_rmse = rmse_folds.std(ddof=1)
            avg_mae  = mae_folds.mean()
            std_mae  = mae_folds.std(ddof=1)
            avg_mape = mape_folds.mean() * 100
            std_mape = mape_folds.std(ddof=1) * 100

            print(f"Cross-validation results for {label}:")
            print(f"  Avg RMSE : {avg_rmse:.2f} ± {std_rmse:.2f}")
            print(f"  Avg MAE  : {avg_mae:.2f} ± {std_mae:.2f}")
            print(f"  Avg MAPE : {avg_mape:.2f}% ± {std_mape:.2f}%")

        print(f"Saving {label} model to {model_path}...")
        joblib.dump(best_model, model_path)
        print(f"{label} model saved.")

    def train_models(self):
        # Drop columns for training
        drop_cols_active = ['PLANNED_FRAME_STACK_ID', 'PLANNED_TOTE_ID', 'START_PICKING_TS', 'END_PICKING_TS', 'TRUCK_TRIP_ID', 'PICK_DATE']
        drop_cols_inactive = drop_cols_active + ['TIME_OF_DAY_MINS', 'NR_OF_PICKERS', 'NR_OF_PICKS']

        # Train active model on training-active DataFrame
        self._train_and_evaluate(
            self.df_train_active,
            drop_cols_active,
            self.model_path_active,
            'Active'
        )

        # Train inactive model on training-inactive DataFrame
        self._train_and_evaluate(
            self.df_train_inactive,
            drop_cols_inactive,
            self.model_path_inactive,
            'Inactive'
        )

        if self.use_gridsearchcv:
            print("\nOptimal parameters found:")
            for label, params in self.best_params.items():
                print(f"  {label}: {params}")

    def _evaluate_loaded_model(self, df: pd.DataFrame, model_path: str, label: str, return_errors=False):
        X, y_true = self._get_feature_targets(df, [])
        model = joblib.load(model_path)
        y_pred = model.predict(X)

        residuals = y_true - y_pred
        abs_errors = np.abs(residuals)
        if return_errors:
            return abs_errors
        sq_errors = residuals ** 2

        # Metrics
        mae = abs_errors.mean()
        mae_std = abs_errors.std(ddof=1)

        rmse = np.sqrt(sq_errors.mean())
        rmse_std = np.sqrt(sq_errors.std(ddof=1))  # std of squared error → std of RMSE approx.

        mape_mask = y_true != 0
        mape_vals = np.abs((y_pred[mape_mask] - y_true[mape_mask]) / y_true[mape_mask]) * 100
        mape = mape_vals.mean()
        mape_std = mape_vals.std(ddof=1)

        resid_variance = residuals.var(ddof=1)
        resid_std = residuals.std(ddof=1)

        print(f"{label:8s} RMSE : {rmse:.2f} ± {rmse_std:.2f}")
        print(f"{label:8s} MAE  : {mae:.2f} ± {mae_std:.2f}")
        print(f"{label:8s} MAPE : {mape:.2f}% ± {mape_std:.2f}%")
        print(f"{label:8s} Residual Std Dev : {resid_std:.2f} min")
        print(f"{label:8s} Residual Variance: {resid_variance:.2f} min²")

    def evaluate(self):
        """
        Evaluate on the pre-split evaluation DataFrames.
        Computes:
          - Mean baseline (only on inactive eval)
          - Inactive GBDT Model
          - Active GBDT Model
        """
        print("\n--- Evaluating Mean Baseline (only on inactive) ---")
        df_mb = self.df_eval_inactive.copy()
        df_mb = df_mb[df_mb['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]
        _, y = self._get_feature_targets(df_mb, [])

        mean_val = self.df_train_inactive[
            self.df_train_inactive['REMAINING_PICKING_TIME'] <= self.remaining_time_cap
        ]['REMAINING_PICKING_TIME'].mean()

        y_pred_mb = np.full_like(y, fill_value=mean_val)

        # Errors
        residuals = y - y_pred_mb
        abs_errors = np.abs(residuals)
        sq_errors = residuals ** 2
        mape_mask = y != 0
        mape_vals = np.abs((y[mape_mask] - mean_val) / y[mape_mask]) * 100

        # Metrics
        rmse_mean = np.sqrt(sq_errors.mean())
        rmse_std  = np.sqrt(sq_errors.std(ddof=1))

        mae_mean  = abs_errors.mean()
        mae_std   = abs_errors.std(ddof=1)

        mape_mean = mape_vals.mean()
        mape_std  = mape_vals.std(ddof=1)

        resid_var = residuals.var(ddof=1)
        resid_std = residuals.std(ddof=1)

        # Report
        print(f"Mean    RMSE      : {rmse_mean:.2f} ± {rmse_std:.2f}")
        print(f"Mean    MAE       : {mae_mean:.2f} ± {mae_std:.2f}")
        print(f"Mean    MAPE      : {mape_mean:.2f}% ± {mape_std:.2f}%")
        print(f"Mean    Residual Std Dev: {resid_std:.2f} min")
        print(f"Mean    Residual Variance: {resid_var:.2f} min²")

        print("\n--- Evaluating Inactive GBDT Model ---")
        df_inactive_eval = self.df_eval_inactive.copy()
        df_inactive_eval = df_inactive_eval[df_inactive_eval['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]
        # Drop TIME_OF_DAY_MINS before feature extraction
        df_inactive_eval = df_inactive_eval.drop(columns=['TIME_OF_DAY_MINS'], errors='ignore')
        self._evaluate_loaded_model(df_inactive_eval, self.model_path_inactive, "Inactive")
        abs_errors_inactive = self._evaluate_loaded_model(df_inactive_eval, self.model_path_inactive, "Inactive", return_errors=True)

        print("\n--- Evaluating Active GBDT Model (on START-only from active dataset) ---")
        df_active_eval = self.df_eval_active.copy()
        df_active_eval = df_active_eval[
            (df_active_eval['PLANNED_TOTE_ID'] == 'START') &
            (df_active_eval['REMAINING_PICKING_TIME'] <= self.remaining_time_cap)
        ]
        self._evaluate_loaded_model(df_active_eval, self.model_path_active, "Active")
        abs_errors_active = self._evaluate_loaded_model(df_active_eval, self.model_path_active, "Active", return_errors=True)


        # --- Statistical significance testing vs. baseline ---

        # Compute mape_vals_inactive
        df_inactive_eval_full = self.df_eval_inactive.copy()
        df_inactive_eval_full = df_inactive_eval_full[df_inactive_eval_full['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]
        df_inactive_eval_full = df_inactive_eval_full.drop(columns=['TIME_OF_DAY_MINS'], errors='ignore')
        _, y_inact = self._get_feature_targets(df_inactive_eval_full, [])
        y_pred_inact = joblib.load(self.model_path_inactive).predict(_)
        mape_mask_inact = y_inact != 0
        mape_vals_inactive = np.abs((y_pred_inact[mape_mask_inact] - y_inact[mape_mask_inact]) / y_inact[mape_mask_inact]) * 100
        rmse_vals_inactive = np.sqrt((y_inact - y_pred_inact) ** 2)

        # Compute mape_vals_active
        df_active_eval_full = self.df_eval_active.copy()
        df_active_eval_full = df_active_eval_full[
            (df_active_eval_full['PLANNED_TOTE_ID'] == 'START') &
            (df_active_eval_full['REMAINING_PICKING_TIME'] <= self.remaining_time_cap)
        ]
        _, y_act = self._get_feature_targets(df_active_eval_full, [])
        y_pred_act = joblib.load(self.model_path_active).predict(_)
        mape_mask_act = y_act != 0
        mape_vals_active = np.abs((y_pred_act[mape_mask_act] - y_act[mape_mask_act]) / y_act[mape_mask_act]) * 100
        rmse_vals_active = np.sqrt((y_act - y_pred_act) ** 2)

        print("\n--- Δ and p-values compared to Mean Baseline ---")

        def paired_t_test(a, b, alternative='greater'):
            # Compute differences
            diffs = np.array(a) - np.array(b)
            t_stat, p = ttest_rel(a, b, alternative=alternative)
            return p

        def report_delta_and_p(label, base_vals, model_vals, metric_name, is_rmse=False):
            base_mean = np.mean(base_vals)
            model_mean = np.mean(model_vals)
            delta = base_mean - model_mean
            p_val = paired_t_test(base_vals, model_vals)
            
            if is_rmse:
                base_mean = np.sqrt(base_mean)
                model_mean = np.sqrt(model_mean)
                delta = base_mean - model_mean  # now in RMSE units
            print(f"{label:8s} Δ {metric_name:7s} : {delta:>7.3f}   (p = {p_val:.2e})")

        # Errors: Mean Baseline
        residuals_mb = y - y_pred_mb
        abs_errors_mb = np.abs(residuals_mb)
        sq_errors_mb = residuals_mb ** 2
        mape_vals_mb = np.abs((y[mape_mask] - mean_val) / y[mape_mask]) * 100

        # Inactive
        report_delta_and_p("Inactive", abs_errors_mb, abs_errors_inactive, "MAE")
        report_delta_and_p("Inactive", mape_vals_mb, mape_vals_inactive, "MAPE")
        report_delta_and_p("Inactive", sq_errors_mb, (y_inact - y_pred_inact) ** 2, "RMSE", is_rmse=True)

        # Active
        report_delta_and_p("Active", abs_errors_mb, abs_errors_active, "MAE")
        report_delta_and_p("Active", mape_vals_mb, mape_vals_active, "MAPE")
        report_delta_and_p("Active", sq_errors_mb, (y_act - y_pred_act) ** 2, "RMSE", is_rmse=True)



    def plot_active_vs_running_completion(self, n_bins: int = 10):
        """
        Plot MAE, RMSE, MAPE, and residual variance vs. running completion
        with both ±1 Std Dev (68%) and ±2 Std Dev (95%) intervals.

        The first bin includes 0.0 and extends up to the first non-zero interval edge.
        """

        df = self.df_eval_active.copy()
        df = df[df['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]

        X, y_true = self._get_feature_targets(df, drop_cols=[])
        y_pred = joblib.load(self.model_path_active).predict(X)

        df_plot = df.assign(
            pred=y_pred,
            abs_error=np.abs(y_pred - y_true),
            sq_error=(y_pred - y_true) ** 2,
            abs_perc_error=np.where(y_true != 0, np.abs((y_pred - y_true) / y_true) * 100, np.nan),
            residual=y_pred - y_true
        )

        if 'PICKING_RUNNING_COMPLETION' not in df_plot:
            raise KeyError("Column 'PICKING_RUNNING_COMPLETION' is missing.")

        # Binning: include 0.0 in first bin
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        df_plot['bin'] = pd.cut(df_plot['PICKING_RUNNING_COMPLETION'], bins=bins, include_lowest=True, right=False)

        # Use sorted bin centers
        bin_intervals = df_plot['bin'].cat.categories
        bin_edges = [iv.left for iv in bin_intervals]

        def bin_stats(series):
            grouped = series.groupby(df_plot['bin'])
            return pd.DataFrame({
                'mean': grouped.mean(),
                'std': grouped.std(),
                'low': grouped.apply(lambda x: np.nanpercentile(x, 2.5)),
                'high': grouped.apply(lambda x: np.nanpercentile(x, 97.5)),
            }).reindex(bin_intervals)
        
        def rmse_bin_stats():
            grouped = df_plot.groupby('bin')['sq_error']
            mean_sq = grouped.mean()
            std_sq  = grouped.std()
            low     = grouped.apply(lambda x: np.nanpercentile(x, 2.5))
            high    = grouped.apply(lambda x: np.nanpercentile(x, 97.5))
            return pd.DataFrame({
                'mean': np.sqrt(mean_sq),
                'std' : std_sq / (2 * np.sqrt(mean_sq)),  # approx std of RMSE via delta method
                'low' : np.sqrt(low),
                'high': np.sqrt(high),
            }).reindex(bin_intervals)

        mae_stats  = bin_stats(df_plot['abs_error'])
        rmse_stats = rmse_bin_stats()
        mape_stats = bin_stats(df_plot['abs_perc_error'])  # avoid explosion
        var_stats  = bin_stats(df_plot['residual'] ** 2)

        # Colors
        base_color = mcolors.to_rgb("C0")
        std_color = [c * 0.6 for c in base_color]
        ci_color = base_color

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
        axes = axes.flatten()

        def plot_with_bands(ax, stats, label, ylabel):
            mean = stats['mean']
            std = stats['std']
            low = stats['low']
            high = stats['high']

            lower_1std = np.maximum(mean - std, 0)
            lower_95ci = np.maximum(low, 0)

            ax.plot(bin_edges, mean, marker='o', color='C0', linewidth=2)
            ax.fill_between(bin_edges, lower_1std, mean + std, color=std_color, alpha=0.2)
            ax.fill_between(bin_edges, lower_95ci, high, color=ci_color, alpha=0.35)

            ax.set_ylabel(ylabel)
            ax.set_title(f'{label} vs. Running Completion')
            ax.grid(True)

            legend_elements = [
                Line2D([0], [0], color='C0', marker='o', linewidth=2, label=f'Mean {label}'),
                Patch(facecolor=std_color, alpha=0.2, label='±1 Std Dev (68%)'),
                Patch(facecolor=ci_color, alpha=0.35, label='±2 Std Dev (95%)'),
            ]
            ax.legend(handles=legend_elements)

        plot_with_bands(axes[0], mae_stats, 'MAE', 'MAE [min]')
        plot_with_bands(axes[1], rmse_stats, 'RMSE', 'RMSE [min]')
        axes[2].set_xlabel('Running Completion')
        plot_with_bands(axes[2], mape_stats, 'MAPE', 'MAPE [%]')
        axes[3].set_xlabel('Running Completion')
        plot_with_bands(axes[3], var_stats, 'Residual Variance', 'Variance [min²]')

        for ax in axes:
            ax.set_xlim(left=min(bin_edges) - 0.01, right=max(bin_edges))

        fig.suptitle('Active GBDT Model – Performance vs. Running Completion\n(±1 Std Dev = 68%, ±2 Std Dev = 95%)', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()



    def plot_residuals(self):
        """
        Plot residuals vs. true remaining time for both active and inactive models
        in a single scatter plot with different colors.
        """
        # Active set
        df_act = self.df_eval_active.copy()
        df_act = df_act[df_act['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]
        X_act, y_act = self._get_feature_targets(df_act, [])
        y_pred_act = joblib.load(self.model_path_active).predict(X_act)
        residual_act = y_pred_act - y_act

        # Inactive set (drop TIME_OF_DAY_MINS)
        df_inact = self.df_eval_inactive.copy()
        df_inact = df_inact[df_inact['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]
        df_inact = df_inact.drop(columns=['TIME_OF_DAY_MINS'], errors='ignore')
        X_inact, y_inact = self._get_feature_targets(df_inact, [])
        y_pred_inact = joblib.load(self.model_path_inactive).predict(X_inact)
        residual_inact = y_pred_inact - y_inact

        # Single combined scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_act, residual_act, alpha=0.3, label='Active Model', color='blue')
        plt.scatter(y_inact, residual_inact, alpha=0.3, label='Inactive Model', color='orange')
        plt.axhline(0, linestyle='--', linewidth=1, color='gray')
        plt.xlabel('True Remaining Time')
        plt.ylabel('Residual (pred - true)')
        plt.title('Residuals vs. True Remaining Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def explain_with_shap(self, n_samples=2000, return_tables=False):
        """
        Compute and plot SHAP summary explanations for both Active and Inactive models.
        Additionally, compute numerical SHAP summary tables with mean(|SHAP value|) per feature.
        Returns tables if return_tables=True.
        """
        def _get_transformed_feature_names(preprocessor):
            numeric_features = preprocessor.named_transformers_['num'].feature_names_in_
            cat_transformer = preprocessor.named_transformers_['cat']
            cat_features = preprocessor.transformers_[1][2]  # This gives original column names

            try:
                cat_feature_names = cat_transformer.get_feature_names_out(cat_features)
            except:
                cat_feature_names = cat_transformer.get_feature_names()  # fallback for older sklearn

            return list(numeric_features) + list(cat_feature_names)

        shap_tables = {}

        for label, df in [
            ('Active', self.df_train_active),
            ('Inactive', self.df_train_inactive)
        ]:
            print(f"\n--- SHAP Explanation for {label} Model ---")

            # Apply remaining_time_cap filter
            df_filtered = df[df['REMAINING_PICKING_TIME'] <= self.remaining_time_cap].copy()
            X, _ = self._get_feature_targets(df_filtered, drop_cols=[])
            X_sample = X.iloc[:n_samples]

            # Load model and extract components
            model_path = self.model_path_active if label == 'Active' else self.model_path_inactive
            pipeline = joblib.load(model_path)
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['regressor']

            # Transform data and get feature names
            X_trans = preprocessor.transform(X_sample)
            feature_names = _get_transformed_feature_names(preprocessor)

            # SHAP values
            explainer = shap.Explainer(model)
            shap_values = explainer(X_trans)

            # Plot summary
            shap.summary_plot(
                shap_values.values,
                features=X_trans,
                feature_names=feature_names,
                plot_type="dot",
                show=True
            )

            # Numerical SHAP summary table
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            std_abs_shap = np.abs(shap_values.values).std(axis=0)

            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean(|SHAP value|)': mean_abs_shap,
                'Std(|SHAP value|)': std_abs_shap
            }).sort_values(by='Mean(|SHAP value|)', ascending=False).reset_index(drop=True)

            shap_tables[label] = shap_df

            # Print top 10
            print(f"\nSHAP feature summary for {label} model (all features):")
            print(shap_df.to_string(index=False))

        if return_tables:
            return shap_tables

        
    def infer_row(self, row_idx: int = 0, use_active_model: bool = True):
        """
        Perform single-row inference and print prediction time, true vs. predicted, and residual.
        """
        df = self.df_eval_active if use_active_model else self.df_eval_inactive
        model_path = self.model_path_active if use_active_model else self.model_path_inactive
        label = "Active" if use_active_model else "Inactive"

        if not 0 <= row_idx < len(df):
            raise IndexError(f"Row index {row_idx} out of bounds for {label} evaluation set with {len(df)} rows.")

        row = df.iloc[row_idx]
        y_true = row["REMAINING_PICKING_TIME"]
        X_row = pd.DataFrame([row.drop("REMAINING_PICKING_TIME")])

        # Drop inactive-only cols if needed
        if not use_active_model:
            X_row = X_row.drop(columns=["TIME_OF_DAY_MINS", "NR_OF_PICKERS", "NR_OF_PICKS"], errors="ignore")

        pipe = joblib.load(model_path)
        preproc = pipe.named_steps["preprocessor"]
        model = pipe.named_steps["regressor"]

        # Measure prediction time
        t0 = time.perf_counter()
        X_trans = preproc.transform(X_row)
        y_pred = model.predict(X_trans)[0]
        elapsed = time.perf_counter() - t0

        print(
            f"{label} model inference (row {row_idx}):"
            f"\n  ▸ true remaining time  : {y_true:.2f} min"
            f"\n  ▸ predicted time       : {y_pred:.2f} min"
            f"\n  ▸ residual             : {y_pred - y_true:+.2f} min"
            f"\n  ▸ inference time       : {elapsed:.4f} seconds"
        )

# Example usage
if __name__ == '__main__':
    gbdt = GBDT_Model(
        big_csv_path='Data/Training/Features_all_dates.csv',
        use_gridsearchcv=False,
        use_kfold_insights=True,
        param_grid={
            'regressor__max_iter': [100, 200, 300],
            'regressor__learning_rate': [0.05, 0.1, 0.3]
        },
        param={
            'regressor__max_iter': 200,
            'regressor__learning_rate': 0.1
        }
    )

    # Train and save models
    #gbdt.train_models()

    # Evaluate on the held-out evaluation split
    gbdt.evaluate()

    # Plot performance metrics vs. running completion for active model
    gbdt.plot_active_vs_running_completion(n_bins=20)

    # Plot residuals for both active and inactive models
    gbdt.plot_residuals()

    # Generate SHAP explanations
    gbdt.explain_with_shap(n_samples=2000)

    gbdt.infer_row(row_idx=1234, use_active_model=False)