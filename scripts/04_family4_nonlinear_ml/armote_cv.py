# ARMOTE: Automated Regression workflow with Multi-Objective hyperparameter optimization using Tree-Parzen Estimator algorithm
#
# Version: 1.0.0
# Author: Shakti P. Padhy
# Date: 2026-06-30
#
# Description:
# This script provides a comprehensive, end-to-end framework for training, optimizing,
# and evaluating multiple regression models. It leverages Optuna for advanced
# multi-objective hyperparameter tuning (minimizing MSE, maximizing R²) and now
# includes computational time tracking for both optimization and final model training.
#
# Workflow Steps:
# 1.  Data Scaling: Features and targets are standardized for modeling.
# 2.  Multi-Objective Optimization: Optuna efficiently searches for the Pareto front
#     of non-dominated hyperparameter solutions.
# 3.  Automated Model Selection: The model with the highest R-squared from the Pareto
#     front is selected as the champion model.
# 4.  Comprehensive Evaluation: The final model is evaluated using R², MSE, and MAPE.
# 5.  Time Tracking: The workflow measures and reports the time spent on optimization
#     and final model training.
# 6.  Artifact Generation & Saving: All important outputs are saved, including trained
#     models, data scalers, Optuna studies, visualization plots, and a final
#     summary CSV with performance metrics and computational times.
# ---

# --- 1. Imports ---
# Core libraries for data manipulation, file operations, numerical processing, and timing.
import json
import os
import copy
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Scikit-learn modules for modeling, metrics, and data preprocessing.
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.gaussian_process.kernels import RBF

# Keras/TensorFlow are imported lazily (inside the NNR-only code paths below) so
# that this module can be imported, and every non-NNR model run, without tensorflow
# installed.

# Optuna for advanced hyperparameter optimization.
import optuna
from tqdm import tqdm


# --- 3. Core Helper Functions ---


def compute_metrics(y_true, y_pred):
    """
    Computes and returns a standard set of regression metrics.

    Args:
        y_true (array-like): The ground truth (actual) target values.
        y_pred (array-like): The values predicted by the model.

    Returns:
        tuple: A tuple containing the (R-squared, MSE, MAPE) scores.
    """
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return r2, mse, mape


def create_nn(
    hidden_layers=1,
    units=64,
    activation="relu",
    learning_rate=0.001,
    input_dim=None,
    output_dim=1,
):
    """
    Creates, configures, and compiles a Keras Sequential neural network for regression
    where the number of units is halved in each subsequent hidden layer.

    Args:
        hidden_layers (int): The number of hidden layers in the network.
        units (int): The number of neurons in each hidden layer.
        activation (str): The activation function for hidden layers.
        learning_rate (float): The learning rate for the Adam optimizer.
        input_dim (int): Number of input features.
        output_dim (int): Number of output targets.

    Returns:
        keras.Model: A compiled Keras model instance ready for training.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    # --- Model Creation ---
    model = Sequential()

    # Add the input layer and the first hidden layer
    model.add(
        Dense(
            units, activation=activation, input_dim=input_dim, name="input_hidden_layer"
        )
    )

    # Initialize a variable to track the number of units for the next layer
    current_units = units

    # Add the remaining hidden layers with halving units
    # This loop runs for the 2nd, 3rd, ... nth hidden layer
    for i in range(hidden_layers - 1):
        # Halve the number of units, ensuring it's at least 1
        current_units = max(1, current_units // 2)
        model.add(
            Dense(current_units, activation=activation, name=f"hidden_layer_{i + 1}")
        )

    # Add the output layer
    model.add(Dense(output_dim, activation="linear"))

    # --- Model Compilation ---
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    return model


def cross_val_nn(
    X,
    y,
    build_fn,
    params,
    cv=5,
    epochs=100,
    batch_size=8,
    refit_scaler_per_fold=True,
    cv_random_state=42,
):
    """
    Performs manual K-Fold cross-validation for a Keras model for multi-objective evaluation.

    Args:
        X (np.array): Feature data. Raw when refit_scaler_per_fold=True, pre-scaled when False.
        y (np.array): Target data. Raw when refit_scaler_per_fold=True, pre-scaled when False.
        build_fn (function): The function used to construct the Keras model.
        params (dict): The dictionary of hyperparameters to pass to the `build_fn`.
        cv (int): The number of folds for cross-validation.
        refit_scaler_per_fold (bool): If True, fit StandardScaler on each fold's training
            data only, eliminating intra-CV scaler leakage.
        cv_random_state (int): Random seed for KFold shuffling.

    Returns:
        tuple: Mean MSE (objective 1) and mean R-squared (objective 2) across all folds.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=cv_random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    input_dim = X.shape[1]
    output_dim = y.shape[1] if y.ndim > 1 else 1
    r2_scores, mse_scores = [], []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if refit_scaler_per_fold:
            fold_xsc = StandardScaler().fit(X_tr)
            fold_ysc = StandardScaler().fit(y_tr.reshape(-1, 1))
            X_tr = fold_xsc.transform(X_tr)
            X_val = fold_xsc.transform(X_val)
            y_tr = fold_ysc.transform(y_tr.reshape(-1, 1))
            y_val = fold_ysc.transform(y_val.reshape(-1, 1))

        model = build_fn(**params, input_dim=input_dim, output_dim=output_dim)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred_val = model.predict(X_val)
        r2_scores.append(r2_score(y_val, y_pred_val))
        mse_scores.append(mean_squared_error(y_val, y_pred_val))
    return np.mean(mse_scores), np.mean(r2_scores)


def plot_yy(
    y_train_list,
    y_pred_train_list,
    y_test_list,
    y_pred_test_list,
    model_name,
    output_name,
    train_metrics,
    test_metrics,
    cv,
    colors,
    save_folder="plots",
):
    """
    Generates and saves a Predicted vs. Actual (Y-Y) plot to visualize model performance.

    Args:
        y_train_list (list): A list of 1D arrays, where each array is the true training targets from a fold.
        y_pred_train_list (list): A list of 1D arrays, where each array is the predicted training targets from a fold.
        y_test_list (list): A list of 1D arrays, where each array is the true test targets from a fold.
        y_pred_test_list (list): A list of 1D arrays, where each array is the predicted test targets from a fold.
        model_name (str): The name of the model for the plot title.
        train_metrics, test_metrics (tuple): Tuples of (R², MSE, MAPE) for train/test sets.
        cv (int): The number of folds for cross-validation.
        colors (list): A list of colors to use for the folds.
        save_folder (str): The directory where the plot image will be saved.
    """
    with plt.style.context("default"):
        os.makedirs(save_folder, exist_ok=True)
        fig = plt.figure(figsize=(14, 6))
        train_r2, train_mse, train_mape = train_metrics
        test_r2, test_mse, test_mape = test_metrics
        n_folds = cv

        ## --- Training Data Subplot (All Folds) ---
        ax1 = plt.subplot(121)

        # Concatenate all fold data just to get the global min/max for the 'Ideal' line
        all_train_true = np.concatenate(y_train_list).ravel()
        all_train_pred = np.concatenate(y_pred_train_list).ravel()
        min_val_train = min(all_train_true.min(), all_train_pred.min())
        max_val_train = max(all_train_true.max(), all_train_pred.max())

        # Plot each fold with a different color
        for i in range(n_folds):
            ax1.scatter(
                y_train_list[i],
                y_pred_train_list[i],
                c=colors[i % len(colors)],
                alpha=0.7,
                label=f"Fold {i}",
            )

        ax1.plot(
            [min_val_train, max_val_train],
            [min_val_train, max_val_train],
            "k--",  # Changed to black dashed line
            label="Ideal",
        )
        ax1.set_xlabel("True Values", fontsize=16)
        ax1.set_ylabel("Predicted Values", fontsize=16)
        ax1.set_title(
            f"{model_name} Y-Y Plot - {output_name} - Train (All Folds)\nAvg R²: {train_r2:.3f} | Avg MSE: {train_mse:.6f} | Avg MAPE: {train_mape:.2f}%",
            fontsize=14,
        )
        ax1.legend()

        # --- Testing Data Subplot (Out-of-Fold) ---
        ax2 = plt.subplot(122)

        # Concatenate all fold data just to get the global min/max for the 'Ideal' line
        all_test_true = np.concatenate(y_test_list).ravel()
        all_test_pred = np.concatenate(y_pred_test_list).ravel()
        min_val_test = min(all_test_true.min(), all_test_pred.min())
        max_val_test = max(all_test_true.max(), all_test_pred.max())

        # Plot each fold with a different color
        for i in range(n_folds):
            ax2.scatter(
                y_test_list[i],
                y_pred_test_list[i],
                c=colors[i % len(colors)],
                alpha=0.7,
                label=f"Fold {i}",
            )

        ax2.plot(
            [min_val_test, max_val_test],
            [min_val_test, max_val_test],
            "k--",
            label="Ideal",
        )
        ax2.set_xlabel("True Values", fontsize=16)
        ax2.set_ylabel("Predicted Values", fontsize=16)
        ax2.set_title(
            f"{model_name} Y-Y Plot - {output_name} - Test (Out-of-Fold)\nAvg R²: {test_r2:.3f} | Avg MSE: {test_mse:.6f} | Avg MAPE: {test_mape:.2f}%",
            fontsize=14,
        )
        ax2.legend()

        plt.tight_layout()
        fig.savefig(
            os.path.join(
                save_folder, f"{model_name}_{output_name}_yy_plot_{cv}_fold_CV.png"
            )
        )
        plt.close(fig)


def generate_optuna_plots(
    study, model_name, cv, output_name, save_folder="plots", fold_idx=None
):
    """
    Generates and saves key Optuna visualization plots as static PNG files.

    Args:
        study (optuna.study.Study): The completed Optuna study object.
        model_name (str): The name of the model for file naming.
        cv (int): The number of folds for cross-validation.
        save_folder (str): The directory where plot images will be saved.
        fold_idx (int, optional): Outer fold index; when provided, appended to filenames.
    """
    try:
        import optuna.visualization.matplotlib as vis
    except ImportError:
        print(
            "Could not import Optuna's matplotlib visualization. Please run 'pip install matplotlib'."
        )
        return

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    fold_tag = f"_fold{fold_idx}" if fold_idx is not None else ""

    # Generate and save each plot
    # Each Optuna matplotlib plot function returns an 'Axes' object.
    # We get its 'figure' attribute to save and then close it.

    # 1. Pareto Front
    ax = vis.plot_pareto_front(study, target_names=["MSE", "R-squared"])
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            save_folder,
            f"{model_name}_{output_name}_pareto_front{fold_tag}_{cv}_fold_CV.png",
        )
    )
    plt.close(fig)

    # 2. Optimization History (MSE)
    ax = vis.plot_optimization_history(
        study, target=lambda t: t.values[0], target_name="MSE"
    )
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            save_folder,
            f"{model_name}_{output_name}_optimization_history_mse{fold_tag}_{cv}_fold_CV.png",
        )
    )
    plt.close(fig)

    # 3. Optimization History (R-squared)
    ax = vis.plot_optimization_history(
        study, target=lambda t: t.values[1], target_name="R-squared"
    )
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            save_folder,
            f"{model_name}_{output_name}_optimization_history_r2{fold_tag}_{cv}_fold_CV.png",
        )
    )
    plt.close(fig)

    # 4. Parameter Importances (Combined for MSE and R-squared)
    try:
        ax = vis.plot_param_importances(study)
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                save_folder,
                f"{model_name}_{output_name}_param_importances_combined{fold_tag}_{cv}_fold_CV.png",
            )
        )
        plt.close(fig)
    except RuntimeError as e:
        print(f"Skipping param importances (combined) for {model_name}: {e}")

    # 5. Parameter Importances (Duration)
    try:
        ax = vis.plot_param_importances(
            study, target=lambda t: t.duration.total_seconds(), target_name="duration"
        )
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                save_folder,
                f"{model_name}_{output_name}_duration_importances{fold_tag}_{cv}_fold_CV.png",
            )
        )
        plt.close(fig)
    except RuntimeError as e:
        print(f"Skipping param importances (duration) for {model_name}: {e}")

    print(
        f"Optuna plots for {model_name} for {output_name} saved as PNGs in '{save_folder}' directory."
    )


# --- 4. Main Workflow Functions ---
def find_best_hyperparameters(
    model,
    param_space,
    X,
    y,
    x_scaler,
    y_scaler,
    cv=5,
    is_nn=False,
    is_gpr=False,
    gpr_kernel_map=None,
    nn_epochs=100,
    nn_batch_size=32,
    refit_scaler_per_fold=True,
    n_trials=100,
    cv_random_state=42,
):
    """
    Performs multi-objective optimization using Optuna on the entire dataset.

    NOTE: Optimization-phase MSE values (Pareto front, printed output) are in
    SCALED space. Final evaluation MSE reported by run_workflow is in original units.
    R² is scale-invariant and comparable across both phases.

    Args:
        model: An unfitted scikit-learn model or None for neural networks.
        param_space (dict): The hyperparameter search space for Optuna.
        X, y: The full feature and target datasets.
        x_scaler, y_scaler: Already-fitted StandardScaler instances (used only when
            refit_scaler_per_fold=False).
        cv (int): The number of folds for cross-validation.
        is_nn (bool): A flag to handle neural network logic separately.
        is_gpr (bool): A flag to handle Gaussian Process Regressor logic separately.
        gpr_kernel_map (dict): A mapping of kernel names to kernel objects for GPR.
        nn_epochs, nn_batch_size (int): Number of epochs and batch size for NN training
            during cross-validation.
        refit_scaler_per_fold (bool): If True, fit a new StandardScaler per CV fold inside
            the optimization objective (sklearn: via Pipeline; NN: manually). Eliminates
            intra-CV scaler leakage. If False, uses the pre-fitted x_scaler/y_scaler.
        n_trials (int): Number of Optuna trials to run.
        cv_random_state (int): Random seed for KFold shuffling in cross-validation.

    Returns:
        tuple: Contains best parameters, the Optuna study, and the optimization time.
    """
    # --- 1. Pre-scale data (only needed when refit_scaler_per_fold=False) ---
    if not refit_scaler_per_fold:
        X_scaled = x_scaler.transform(X)
        y_scaled = y_scaler.transform(y.reshape(-1, 1))

    # Handle GPR kernel mapping requirement
    final_kernel_map = gpr_kernel_map
    if is_gpr and final_kernel_map is None and param_space and "kernel" in param_space:
        raise ValueError(
            "A 'gpr_kernel_map' must be provided to 'run_workflow' "
            "when tuning the 'kernel' parameter for 'GaussianProcess'."
        )

    # --- 2. Handle Models Without Hyperparameter Tuning ---
    if not param_space:
        print("No hyperparameters defined. Returning default settings.")
        return ({}, None, 0)

    # --- 3. Define the Multi-Objective Function for Optuna ---
    def objective(trial):
        params = {}
        for name, definition in param_space.items():
            if definition[0] == "int":
                params[name] = trial.suggest_int(name, *definition[1:])
            elif definition[0] == "float":
                params[name] = trial.suggest_float(
                    name, definition[1], definition[2], log=("log" in definition)
                )
            elif definition[0] == "categorical":
                params[name] = trial.suggest_categorical(name, definition[1])

        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=cv_random_state)

        if is_nn:
            from tensorflow.keras import backend as K
            K.clear_session()
            if refit_scaler_per_fold:
                return cross_val_nn(
                    X,
                    y,
                    create_nn,
                    params,
                    cv=cv,
                    epochs=nn_epochs,
                    batch_size=nn_batch_size,
                    refit_scaler_per_fold=True,
                    cv_random_state=cv_random_state,
                )
            else:
                return cross_val_nn(
                    X_scaled,
                    y_scaled,
                    create_nn,
                    params,
                    cv=cv,
                    epochs=nn_epochs,
                    batch_size=nn_batch_size,
                    refit_scaler_per_fold=False,
                    cv_random_state=cv_random_state,
                )
        else:
            # Resolve GPR kernel string → object without mutating trial params
            if is_gpr and "kernel" in params:
                resolved_params = {
                    **params,
                    "kernel": final_kernel_map[params["kernel"]],
                }
            else:
                resolved_params = params

            scoring = {"r2": "r2", "neg_mse": "neg_mean_squared_error"}
            if refit_scaler_per_fold:
                model_clone = copy.deepcopy(model)
                pipe = Pipeline([("scaler", StandardScaler()), ("model", model_clone)])
                pipe.set_params(
                    **{f"model__{k}": v for k, v in resolved_params.items()}
                )
                scores = cross_validate(
                    pipe, X, y.ravel(), cv=cv_splitter, scoring=scoring, n_jobs=1
                )
            else:
                model_clone = copy.deepcopy(model)
                model_clone.set_params(**resolved_params)
                scores = cross_validate(
                    model_clone,
                    X_scaled,
                    y_scaled.ravel(),
                    cv=cv_splitter,
                    scoring=scoring,
                    n_jobs=1,
                )
            return -np.mean(scores["test_neg_mse"]), np.mean(scores["test_r2"])

    # --- 4. Run and Time the Optuna Optimization Study ---
    n_parallel_jobs = 1 if is_nn or is_gpr else -1
    if is_nn:
        print("Using n_jobs=1 for Neural Network to ensure GPU stability.")
    if is_gpr:
        print(
            "Using n_jobs=1 for GaussianProcessRegressor to manage high memory usage."
        )

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, directions=["minimize", "maximize"])
    start_time = time.time()

    study.optimize(
        objective, n_trials=n_trials, n_jobs=n_parallel_jobs, show_progress_bar=True
    )
    optimization_time = time.time() - start_time
    print(f"Hyperparameter optimization completed in {optimization_time:.3f} seconds.")

    # --- 5. Select the Best Model from the Pareto Front ---
    print("Selecting best trial from the Pareto front...")
    best_trial = max(study.best_trials, key=lambda t: t.values[1])
    best_params = best_trial.params
    print(
        f"Selected Trial #{best_trial.number} with MSE={best_trial.values[0]:.4f} (scaled space), R2={best_trial.values[1]:.4f}"
    )

    return (
        best_params,
        study,
        optimization_time,
    )


def run_workflow(
    X,
    y,
    models,
    param_spaces,
    cv=5,
    output_name="Objective 1",
    output_folder_name="workflow_output",
    gpr_kernel_map=None,
    nn_epochs=100,
    nn_batch_size=32,
    colors=["#EE6677", "#228833", "#4477AA", "#CCBB44", "#66CCEE"],
    refit_scaler_per_fold=True,
    n_trials=50,
    cv_random_state=42,
    nn_model_names=("NNR",),
    gpr_model_names=("GPR",),
    splitter=None,
    inner_cv=5,
    pool_oof_metrics=False,
    groups=None,
    overwrite=False,
    append=False,
):
    """
    Executes the end-to-end multi-objective machine learning workflow using nested CV.
    For each outer fold:
      1. Runs Optuna hyperparameter search on that fold's training data only (inner CV).
      2. Retrains the final model with the fold-specific best params on that fold's train data.
      3. Evaluates on the held-out outer test fold (never seen during optimization).
    Saves the model and scalers from every fold.

    Args:
        X (pd.DataFrame or np.array): The complete feature dataset.
        y (pd.Series or np.array): The complete target dataset.
        models (dict): A dictionary of model names to their unfitted instances.
        param_spaces (dict): A dictionary mapping model names to hyperparameter search
            spaces. Each space is a dict of {param_name: definition} where definition is
            a tuple in one of these formats:
                ("int",   low, high)               — integer range
                ("float", low, high)               — float range
                ("float", low, high, "log")        — log-scale float range
                ("categorical", [val1, val2, ...]) — discrete choices
            Example::

                param_spaces = {
                    "RF": {
                        "n_estimators": ("int", 50, 500),
                        "max_depth":    ("int", 3, 20),
                        "min_samples_split": ("float", 0.01, 0.5),
                    },
                    "GPR": {
                        "kernel": ("categorical", ["RBF", "Matern"]),
                        "alpha":  ("float", 1e-6, 1e-1, "log"),
                    },
                }

        output_name (str): Name of the target variable; used in file/plot names.
        output_folder_name (str): Root directory for all saved outputs.
        gpr_kernel_map (dict, optional): Maps kernel name strings to kernel objects,
            required when "kernel" is in the GPR param space. Example::

                from sklearn.gaussian_process.kernels import RBF, Matern
                gpr_kernel_map = {"RBF": RBF(1.0), "Matern": Matern(nu=1.5)}

        nn_epochs, nn_batch_size (int): Training epochs and batch size for NNR.
        colors (list): Fold colors for Y-Y plots.
        refit_scaler_per_fold (bool): If True (default), fit a new StandardScaler on each
            fold's training data inside the optimization CV, eliminating intra-CV scaler
            leakage. If False, scalers are fit on 100% of the data (original behavior).
        n_trials (int): Number of Optuna hyperparameter search trials (default 100).
        cv_random_state (int): Random seed for all KFold splits (default 42).
        nn_model_names (tuple): Model dict keys treated as Keras NNs (default ("NNR",)).
        gpr_model_names (tuple): Model dict keys treated as GPRs (default ("GPR",)).

    Returns:
        pd.DataFrame: Summary of performance metrics and timings for all models.
    """
    # --- 0. Input Validation ---
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} samples but y has {len(y)}.")
    if cv < 2:
        raise ValueError(f"cv must be >= 2, got {cv}.")
    if len(colors) == 0:
        raise ValueError("colors list must not be empty.")

    # --- 1. Setup and Data Preparation ---
    print(f"Initializing Nested CV workflow for target: '{output_folder_name}'...")

    # Define dynamic paths based on the output_name
    base_dir = output_folder_name
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(base_dir, "plots")
    studies_dir = os.path.join(base_dir, "studies")

    print("Setting up directories...")
    # Create all new dynamic directories
    for folder in [models_dir, plots_dir, studies_dir]:
        os.makedirs(folder, exist_ok=True)

    # Ensure X and y are pandas objects for easy .iloc indexing
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        X = pd.DataFrame(X)
    if not isinstance(y, (pd.DataFrame, pd.Series)):
        y = pd.Series(y)
    # Ensure y is a 1D Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Remove any existing indices to prevent misalignment during CV
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Store original numpy arrays for y
    y_numpy = y.values

    # --- 2. Model Training and Evaluation Loop ---
    results = []
    kf = (
        splitter
        if splitter is not None
        else KFold(n_splits=cv, shuffle=True, random_state=cv_random_state)
    )
    n_outer_splits = kf.get_n_splits(X, y, groups)

    # --- Early CSV path resolution + append/overwrite guard ---
    results_csv_path = os.path.join(
        base_dir, f"{output_name}_results_{n_outer_splits}_fold_CV.csv"
    )
    existing_df = None
    if os.path.exists(results_csv_path):
        if append:
            existing_df = pd.read_csv(results_csv_path)
            completed = set(existing_df["Model"].tolist())
            models = {k: v for k, v in models.items() if k not in completed}
            if not models:
                print(f"All models already in {results_csv_path}. Nothing to do.")
                return existing_df
            print(f"Appending. Already done: {sorted(completed)}. Remaining: {list(models.keys())}")
        elif not overwrite:
            raise FileExistsError(
                f"Output CSV already exists: {results_csv_path}\n"
                "Re-run with --overwrite to replace it, or --append to add missing models."
            )

    model_pbar = tqdm(models.items(), total=len(models), desc="Models", unit="model")
    for name, model in model_pbar:
        model_pbar.set_description(f"Model: {name}")
        print(f"\n--- Starting Workflow for: {name} ---")

        # Flags for model-specific handling
        is_nn = name in nn_model_names
        is_gpr = name in gpr_model_names

        # === NESTED CV: hyperparameter search + evaluation per fold ===
        fold_train_metrics_list = []
        fold_test_metrics_list = []
        fold_retrain_times = []
        fold_opt_times = []
        fold_best_params_list = []

        oof_y_true = []
        oof_y_pred = []
        all_train_y_true = []
        all_train_y_pred = []

        fold_pbar = tqdm(
            enumerate(kf.split(X, y, groups)),
            total=n_outer_splits,
            desc=f"{name} - Nested CV folds",
            unit="fold",
            leave=False,
        )
        for fold, (train_idx, test_idx) in fold_pbar:
            fold_pbar.set_description(f"{name} - Fold {fold + 1}/{n_outer_splits}")
            print(f"\n--- {name} - Fold {fold + 1}/{n_outer_splits} ---")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            y_train_numpy, y_test_numpy = y_numpy[train_idx], y_numpy[test_idx]

            # --- Step 1: Hyperparameter search on this fold's train data only ---
            print(
                f"  Step 1: Hyperparameter search on {len(X_train)} train samples (fold {fold + 1})..."
            )

            x_scaler_opt = y_scaler_opt = None
            if not refit_scaler_per_fold:
                x_scaler_opt = StandardScaler().fit(X_train)
                y_scaler_opt = StandardScaler().fit(y_train_numpy.reshape(-1, 1))

            best_params, study, optimization_time = find_best_hyperparameters(
                model,
                param_spaces.get(name, {}),
                X_train,
                y_train_numpy,
                x_scaler_opt,
                y_scaler_opt,
                cv=inner_cv,
                is_nn=is_nn,
                is_gpr=is_gpr,
                gpr_kernel_map=gpr_kernel_map,
                nn_epochs=nn_epochs,
                nn_batch_size=nn_batch_size,
                refit_scaler_per_fold=refit_scaler_per_fold,
                n_trials=n_trials,
                cv_random_state=cv_random_state,
            )
            fold_opt_times.append(optimization_time)
            fold_best_params_list.append(best_params)

            if study:
                study_path = os.path.join(
                    studies_dir, f"{name}_{output_name}_study_fold_{fold}.pkl"
                )
                joblib.dump(study, study_path)
                print(f"  Optuna study (fold {fold + 1}) saved to: {study_path}")
                generate_optuna_plots(
                    study, name, cv, output_name, save_folder=plots_dir, fold_idx=fold
                )

            # --- Step 2: Fit final scalers on outer train data ---
            print(f"  Step 2: Fitting scalers on Fold {fold + 1} training data...")
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            x_scaler.fit(X_train)
            y_scaler.fit(y_train_numpy.reshape(-1, 1))

            X_train_scaled = x_scaler.transform(X_train)
            y_train_scaled = y_scaler.transform(y_train_numpy.reshape(-1, 1))
            X_test_scaled = x_scaler.transform(X_test)

            joblib.dump(
                x_scaler,
                os.path.join(
                    models_dir, f"{name}_{output_name}_x_scaler_fold_{fold}.pkl"
                ),
            )
            joblib.dump(
                y_scaler,
                os.path.join(
                    models_dir, f"{name}_{output_name}_y_scaler_fold_{fold}.pkl"
                ),
            )
            print(
                f"  {name}_{output_name}_x_scaler_fold_{fold}.pkl and {name}_{output_name}_y_scaler_fold_{fold}.pkl saved in '{models_dir}'."
            )

            # --- Step 3: Retrain final model with fold-specific best params ---
            print(
                f"  Step 3: Retraining final model with fold {fold + 1} best params..."
            )
            start_time = time.time()

            if is_nn:
                from tensorflow.keras import backend as K
                K.clear_session()
                final_model = create_nn(
                    **best_params,
                    input_dim=X_train_scaled.shape[1],
                    output_dim=1,
                )
                final_model.fit(
                    X_train_scaled,
                    y_train_scaled,
                    epochs=nn_epochs,
                    batch_size=nn_batch_size,
                    verbose=0,
                )
            elif is_gpr:
                final_params = best_params.copy()
                if "kernel" in final_params:
                    kernel_name = final_params.pop("kernel")
                    final_params["kernel"] = gpr_kernel_map[kernel_name]
                elif "kernel" not in final_params:
                    final_params["kernel"] = RBF(1.0)
                    final_params["alpha"] = 1e-10
                final_model = copy.deepcopy(model).set_params(**final_params)
                final_model.fit(X_train_scaled, y_train_scaled.ravel())
            else:
                final_model = copy.deepcopy(model).set_params(**best_params)
                final_model.fit(X_train_scaled, y_train_scaled.ravel())

            retraining_time = time.time() - start_time
            print(
                f"  Fold {fold + 1}/{n_outer_splits} model training complete in {retraining_time:.3f} seconds."
            )

            # --- Step 4: Evaluate on held-out outer test fold ---
            y_pred_train_scaled = final_model.predict(X_train_scaled)
            y_pred_test_scaled = final_model.predict(X_test_scaled)
            y_pred_train = y_scaler.inverse_transform(
                y_pred_train_scaled.reshape(-1, 1)
            )
            y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1))

            train_metrics = compute_metrics(y_train_numpy, y_pred_train)
            if pool_oof_metrics and len(y_test_numpy) < 2:
                fold_mse = mean_squared_error(y_test_numpy.ravel(), y_pred_test.ravel())
                fold_mape = (
                    mean_absolute_percentage_error(
                        y_test_numpy.ravel(), y_pred_test.ravel()
                    )
                    * 100
                )
                test_metrics = (np.nan, fold_mse, fold_mape)
            else:
                test_metrics = compute_metrics(y_test_numpy, y_pred_test)

            fold_train_metrics_list.append(train_metrics)
            fold_test_metrics_list.append(test_metrics)
            fold_retrain_times.append(retraining_time)

            oof_y_true.append(y_test_numpy)
            oof_y_pred.append(y_pred_test)
            all_train_y_true.append(y_train_numpy)
            all_train_y_pred.append(y_pred_train)

            # --- Step 5: Save model ---
            print(f"  Saving model for {name} from Fold {fold}...")
            model_path = os.path.join(
                models_dir,
                f"{name}_{output_name}_best_model_fold_{fold}.{'keras' if is_nn else 'pkl'}",
            )
            if is_nn:
                final_model.save(model_path)
            else:
                joblib.dump(final_model, model_path)
            print(
                f"  Best model for {name} for {output_name} (Fold {fold}) saved to: {model_path}"
            )

        # --- 9. Collate Results After All Folds ---
        print(
            f"\n--- Aggregating {n_outer_splits}-Fold CV results for: {name} for {output_name} ---"
        )

        train_metrics_df = pd.DataFrame(
            fold_train_metrics_list, columns=["R2", "MSE", "MAPE"]
        )
        test_metrics_df = pd.DataFrame(
            fold_test_metrics_list, columns=["R2", "MSE", "MAPE"]
        )
        avg_train_metrics = train_metrics_df.mean().values
        avg_retrain_time = np.mean(fold_retrain_times)

        if pool_oof_metrics:
            pooled_true = np.concatenate([a.ravel() for a in oof_y_true])
            pooled_pred = np.concatenate([a.ravel() for a in oof_y_pred])
            avg_test_metrics = compute_metrics(pooled_true, pooled_pred)
            test_r2_std = test_mse_std = test_mape_std = np.nan
        else:
            avg_test_metrics = test_metrics_df.mean().values
            test_r2_std = test_metrics_df["R2"].std()
            test_mse_std = test_metrics_df["MSE"].std()
            test_mape_std = test_metrics_df["MAPE"].std()

        # Generate combined plot
        plot_yy(
            all_train_y_true,
            all_train_y_pred,
            oof_y_true,
            oof_y_pred,
            name,
            output_name,
            avg_train_metrics,
            avg_test_metrics,
            n_outer_splits,
            colors,
            save_folder=plots_dir,
        )

        # Store results in the main list
        results.append(
            {
                "Model": name,
                "Total Optimization Time (s)": round(sum(fold_opt_times), 3),
                "Avg Optimization Time per Fold (s)": round(np.mean(fold_opt_times), 3),
                "Avg Retraining Time (s)": round(avg_retrain_time, 3),
                "Best Params Per Fold": json.dumps(fold_best_params_list),
                "Avg Train R2": avg_train_metrics[0],
                "Std Train R2": train_metrics_df["R2"].std(),
                "Avg Train MSE (original units)": avg_train_metrics[1],
                "Std Train MSE (original units)": train_metrics_df["MSE"].std(),
                "Avg Train MAPE (%)": avg_train_metrics[2],
                "Std Train MAPE (%)": train_metrics_df["MAPE"].std(),
                "Avg Test R2": avg_test_metrics[0],
                "Std Test R2": test_r2_std,
                "Avg Test MSE (original units)": avg_test_metrics[1],
                "Std Test MSE (original units)": test_mse_std,
                "Avg Test MAPE (%)": avg_test_metrics[2],
                "Std Test MAPE (%)": test_mape_std,
                "All Fold Test R2": json.dumps(
                    [None if pd.isna(v) else v for v in test_metrics_df["R2"].tolist()]
                ),
                "All Fold Test MSE (original units)": json.dumps(
                    test_metrics_df["MSE"].tolist()
                ),
                "All Fold Test MAPE (%)": json.dumps(test_metrics_df["MAPE"].tolist()),
                "All Fold Train R2": json.dumps(train_metrics_df["R2"].tolist()),
                "All Fold Train MSE (original units)": json.dumps(
                    train_metrics_df["MSE"].tolist()
                ),
                "All Fold Train MAPE (%)": json.dumps(
                    train_metrics_df["MAPE"].tolist()
                ),
                "NNR_Epochs": nn_epochs if is_nn else np.nan,
                "NNR_Batch_Size": nn_batch_size if is_nn else np.nan,
            }
        )

    # --- 10. Final Summary ---
    results_df = pd.DataFrame(results)
    if existing_df is not None:
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    results_df.to_csv(results_csv_path, index=False)
    print("\n--- Workflow Complete ---")
    print(
        f"Final {n_outer_splits}-fold CV results summary for {output_name} saved to {results_csv_path}"
    )
    return results_df
