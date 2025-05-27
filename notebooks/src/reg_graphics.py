import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import EngFormatter
from sklearn.metrics import PredictionErrorDisplay

from reg_models import RANDOM_STATE

PALETTE = ["#5A189A", "#D81159", "#4C1E4F", "#DA627D","#5E548E", "#479D94", "#55396F", "#F08080","#023E7D", "#427AA1"]
SCATTER_ALPHA = 0.2

sns.set_theme(style="darkgrid", context="notebook", palette=PALETTE)


def plot_coefficients(df_coefs, title="Coefficients"):
    df_coefs.plot.barh()
    plt.title(title)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficients")
    plt.gca().get_legend().remove()
    plt.show()


def plot_residuals(y_true, y_pred):
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
