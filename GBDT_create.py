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

        # Hyperparameter tuning
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
            print(f"Best params for {label}: {self.best_params[label]}")
            best_model = grid.best_estimator_
        else:
            print("Fitting histogram GBDT with early stopping...")
            best_model = pipe.fit(X, y)

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

    def _evaluate_loaded_model(self, df: pd.DataFrame, model_path: str, label: str):
        X, y_true = self._get_feature_targets(df, [])
        model = joblib.load(model_path)
        y_pred = model.predict(X)

        rmse, mae, mape = self._evaluate_metrics(y_true, y_pred)

        # compute residuals and their variance/std
        residuals = y_true - y_pred
        resid_variance = np.var(residuals, ddof=1)   # sample variance
        resid_std      = np.std(residuals, ddof=1)   # sample standard deviation

        print(f"{label:8s} RMSE: {rmse:.2f}")
        print(f"{label:8s} MAE : {mae:.2f}")
        print(f"{label:8s} MAPE: {mape:.2f}%")
        print(f"{label:8s} Std Dev : {resid_std:.2f} min")
        print(f"{label:8s} Variance: {resid_variance:.2f} min²")

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
        # Apply remaining_time_cap filter
        df_mb = df_mb[df_mb['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]
        _, y = self._get_feature_targets(df_mb, [])
        # Compute mean baseline from inactive training data
        mean_val = self.df_train_inactive[self.df_train_inactive['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]['REMAINING_PICKING_TIME'].mean()
        y_pred_mb = np.full_like(y, fill_value=mean_val)

        # compute primary metrics
        rmse_mb, mae_mb, mape_mb = self._evaluate_metrics(y, y_pred_mb)

        # compute residuals, variance, std
        residuals_mb = y - y_pred_mb
        resid_var_mb = np.var(residuals_mb, ddof=1)   # sample variance
        resid_std_mb = np.std(residuals_mb, ddof=1)   # sample std deviation

        print(f"Mean    RMSE      : {rmse_mb:.2f}")
        print(f"Mean    MAE       : {mae_mb:.2f}")
        print(f"Mean    MAPE      : {mape_mb:.2f}%")
        print(f"Mean Resid Std Dev: {resid_std_mb:.2f} (min)    Variance: {resid_var_mb:.2f} (min²)")

        print("\n--- Evaluating Inactive GBDT Model ---")
        df_inactive_eval = self.df_eval_inactive.copy()
        df_inactive_eval = df_inactive_eval[df_inactive_eval['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]
        # Drop TIME_OF_DAY_MINS before feature extraction
        df_inactive_eval = df_inactive_eval.drop(columns=['TIME_OF_DAY_MINS'], errors='ignore')
        self._evaluate_loaded_model(df_inactive_eval, self.model_path_inactive, "Inactive")

        print("\n--- Evaluating Active GBDT Model ---")
        df_active_eval = self.df_eval_active.copy()
        df_active_eval = df_active_eval[df_active_eval['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]
        # Drop no columns for active (handled inside _get_feature_targets)
        self._evaluate_loaded_model(df_active_eval, self.model_path_active, "Active")

    def plot_active_vs_running_completion(self, n_bins: int = 10):
        """
        Plot MAE, RMSE, MAPE, and residual variance vs. running completion
        for the active model in a 2x2 grid, including 0 but excluding 1.0.
        Uses the evaluation-active DataFrame.
        """
        # Prepare evaluation-active DataFrame
        df = self.df_eval_active.copy()
        df = df[df['REMAINING_PICKING_TIME'] <= self.remaining_time_cap]

        X, y_true = self._get_feature_targets(df, drop_cols=[])
        y_pred = joblib.load(self.model_path_active).predict(X)

        df_plot = df.assign(
            pred=y_pred,
            abs_error=np.abs(y_pred - y_true),
            sq_error=(y_pred - y_true) ** 2,
            abs_perc_error=np.where(
                y_true != 0,
                np.abs((y_pred - y_true) / y_true) * 100,
                np.nan
            ),
            residual=y_pred - y_true
        )
        if 'PICKING_RUNNING_COMPLETION' not in df_plot:
            raise KeyError("Column 'PICKING_RUNNING_COMPLETION' is missing.")

        # define uniform bins over [0,1), exclude 1.0
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        df_plot['bin'] = pd.cut(
            df_plot['PICKING_RUNNING_COMPLETION'], bins=bins,
            include_lowest=True, right=False
        )
        df_plot = df_plot[df_plot['bin'].notna()]

        # aggregate metrics by bin
        mae_stats = df_plot.groupby('bin')['abs_error'].mean()
        rmse_stats = np.sqrt(df_plot.groupby('bin')['sq_error'].mean())
        mape_stats = df_plot.groupby('bin')['abs_perc_error'].mean()
        var_stats = df_plot.groupby('bin')['residual'].var(ddof=1)

        # compute midpoints and prep values, padding 0
        centers = mae_stats.index.map(lambda iv: iv.mid)
        x_vals = np.concatenate(([0.0], centers.values))
        mae_vals = np.concatenate(([mae_stats.iloc[0]], mae_stats.values))
        rmse_vals = np.concatenate(([rmse_stats.iloc[0]], rmse_stats.values))
        mape_vals = np.concatenate(([mape_stats.iloc[0]], mape_stats.values))
        var_vals = np.concatenate(([var_stats.iloc[0]], var_stats.values))

        # plot metrics in 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
        axes_flat = axes.flatten()

        # MAE subplot
        axes_flat[0].plot(x_vals, mae_vals, 'o-')
        axes_flat[0].set_ylabel('MAE [min]')
        axes_flat[0].set_title('MAE vs. Running Completion')
        axes_flat[0].grid(True)

        # RMSE subplot
        axes_flat[1].plot(x_vals, rmse_vals, 'o-')
        axes_flat[1].set_ylabel('RMSE [min]')
        axes_flat[1].set_title('RMSE vs. Running Completion')
        axes_flat[1].grid(True)

        # MAPE subplot
        axes_flat[2].plot(x_vals, mape_vals, 'o-')
        axes_flat[2].set_xlabel('Running Completion')
        axes_flat[2].set_ylabel('MAPE [%]')
        axes_flat[2].set_title('MAPE vs. Running Completion')
        axes_flat[2].grid(True)

        # Residual Variance subplot
        axes_flat[3].plot(x_vals, var_vals, 'o-')
        axes_flat[3].set_xlabel('Running Completion')
        axes_flat[3].set_ylabel('Residual Variance [min²]')
        axes_flat[3].set_title('Residual Variance vs. Running Completion')
        axes_flat[3].grid(True)

        # x-axis limits
        axes_flat[3].set_xlim(0.0, centers.max())

        fig.suptitle('Active Model Performance Metrics vs. Running Completion')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_residuals(self):
        """
        Plot residuals vs. true remaining time for both active and inactive models
        using the evaluation DataFrames.
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

        # create side-by-side scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        axes[0].scatter(y_act, residual_act, alpha=0.3)
        axes[0].axhline(0, linestyle='--', linewidth=1)
        axes[0].set_xlabel('True Remaining Time')
        axes[0].set_ylabel('Residual (pred - true)')
        axes[0].set_title('Active Model Residuals')
        axes[0].grid(True)

        axes[1].scatter(y_inact, residual_inact, alpha=0.3)
        axes[1].axhline(0, linestyle='--', linewidth=1)
        axes[1].set_xlabel('True Remaining Time')
        axes[1].set_title('Inactive Model Residuals')
        axes[1].grid(True)

        fig.suptitle('Residuals vs. True Remaining Time')
        fig.tight_layout()
        plt.show()

    def explain_with_shap(self, n_samples=2000):
        """
        Compute and plot SHAP summary explanations for both Active and Inactive models.
        Each plot is shown in its own figure.
        Uses the training DataFrames.
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

            # Plot
            shap.summary_plot(
                shap_values.values,
                features=X_trans,
                feature_names=feature_names,
                plot_type="dot",
                show=True
            )


# Example usage
if __name__ == '__main__':
    gbdt = GBDT_Model(
        big_csv_path='Data/Training/Features_all_dates.csv',
        use_gridsearchcv=True,
        use_kfold_insights=False,
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
