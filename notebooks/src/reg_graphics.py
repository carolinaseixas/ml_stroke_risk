import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import EngFormatter
from sklearn.metrics import PredictionErrorDisplay

from reg_models import RANDOM_STATE

PALETTE = ["#5A189A", "#D81159", "#4C1E4F", "#DA627D","#5E548E", "#479D94", "#55396F", "#F08080","#023E7D", "#427AA1"]
SCATTER_ALPHA = 0.2

sns.set_theme(style="darkgrid", context="notebook", palette=PALETTE)


def plot_coefficients(df_coefs, title="Coefficients"):
    """
    Plots a horizontal bar chart of model coefficients.

    This function visualizes the coefficients of a regression model as a horizontal bar chart, 
    where each bar represents a coefficient's value. It helps to understand the importance and 
    influence of each feature on the model's predictions.

    Parameters:
    ----------
    df_coefs : pandas DataFrame
        A DataFrame containing the model's coefficients. The index should correspond to feature names 
        and the values should represent the magnitude of the coefficients.

    title : str, optional (default="Coefficients")
        The title to display at the top of the plot.

    Returns:
    -------
    None
        The function displays the plot without returning any value.

    Notes:
    -----
    A vertical line at `x=0` is plotted to distinguish positive and negative coefficients.
    """

    df_coefs.plot.barh()
    plt.title(title)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficients")
    plt.gca().get_legend().remove()
    plt.show()


def plot_residuals(y_true, y_pred):
    """
    Plots residuals to evaluate the fit of a regression model.

    This function visualizes the residuals (differences between true and predicted values) in 
    three subplots:
    1. A histogram of residuals with a KDE curve to analyze the distribution.
    2. A plot of residuals vs. predicted values to detect patterns.
    3. A plot of actual vs. predicted values to visualize model performance.

    Parameters:
    ----------
    y_true : array-like
        True target values.

    y_pred : array-like
        Predicted target values.

    Returns:
    -------
    None
        The function displays the plot without returning any value.
    """

    residuals = y_true - y_pred

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    sns.histplot(residuals, kde=True, ax=axs[0])

    error_display_01 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=axs[1]
    )

    error_display_02 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=axs[2]
    )

    plt.tight_layout()

    plt.show()


def plot_estimator_residuals(estimator, X, y, eng_formatter=False, sample_fraction=0.25):
    """
    Plots residuals for a regression model estimator using various visualizations.

    This function visualizes the residuals (differences between true and predicted values) in 
    three subplots:
    1. A histogram of residuals with a KDE curve to analyze the distribution.
    2. A plot of residuals vs. predicted values to detect patterns.
    3. A plot of actual vs. predicted values to visualize model performance.
    
    It also allows for formatting the axes with engineering notation if desired.

    Parameters:
    ----------
    estimator : estimator object
        A trained regression model (e.g., LinearRegression, RandomForestRegressor).

    X : array-like
        Features of the input data.

    y : array-like
        True target values.

    eng_formatter : bool, optional (default=False)
        If True, formats the axes with engineering notation (useful for large numbers).

    sample_fraction : float, optional (default=0.25)
        The fraction of the data to sample for plotting, useful for large datasets.

    Returns:
    -------
    None
        The function displays the plot without returning any value.
    """

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=sample_fraction,
    )

    error_display_02 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="actual_vs_predicted",
        ax=axs[2],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=sample_fraction,
    )

    residuals = error_display_01.y_true - error_display_01.y_pred

    sns.histplot(residuals, kde=True, ax=axs[0])

    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())

    plt.tight_layout()

    plt.show()


def plot_compare_models_metrics(df_results):
    """
    Compares model metrics across different models.

    This function visualizes and compares key performance metrics (e.g., time, R², MAE, RMSE) 
    for multiple models using boxplots. It helps in assessing how each model performs on 
    various metrics and highlights their variations.

    Parameters:
    ----------
    df_results : pandas DataFrame
        A DataFrame containing the results of multiple models, where each row corresponds to a 
        model and columns represent different performance metrics such as R², MAE, RMSE, and time.
        It expects the following column names: "time_seconds", "test_r2", "test_neg_mean_absolute_error" and
        "test_neg_root_mean_squared_error".

    Returns:
    -------
    None
        The function displays the plot without returning any value.

    Notes:
    -----
    Four metrics are visualized: 
        - Time (in seconds)
        - R² (coefficient of determination)
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
    """
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    compare_metrics = [
        "time_seconds",
        "test_r2",
        "test_neg_mean_absolute_error",
        "test_neg_root_mean_squared_error",
    ]

    metrics_names = [
        "Time (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    for ax, metric, name in zip(axs.flatten(), compare_metrics, metrics_names):
        sns.boxplot(
            x="model",
            y=metric,
            data=df_results,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(name)
        ax.set_ylabel(name)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.show()
