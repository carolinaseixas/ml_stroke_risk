import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = ["#5A189A", "#D81159", "#4C1E4F", "#DA627D","#5E548E", "#479D94", "#55396F", "#F08080","#023E7D", "#427AA1"]
SCATTER_ALPHA = 0.2

sns.set_theme(style="darkgrid", context="notebook", palette=PALETTE)


def plot_compare_models_metrics(df_resultados):
    """
    Compares model metrics across different models.

    This function visualizes and compares key performance metrics (e.g., time, recall, AUROC, AUPRC) 
    for multiple models using boxplots. It helps in assessing how each model performs on 
    various metrics and highlights their variations.

    Parameters:
    ----------
    df_results : pandas DataFrame
        A DataFrame containing the results of multiple models, where each row corresponds to a 
        model and columns represent different performance metrics such as R², MAE, RMSE, and time.
        It expects the following column names: "time_seconds", "test_accuracy", "test_balanced_accuracy",
        "test_f1", "test_precision", "test_recall", "test_roc_auc" and "test_average_precision".

    Returns:
    -------
    None
        The function displays the plot without returning any value.

    Notes:
    -----
    Four metrics are visualized: 
        - Time (in seconds)
        - Accuracy
        - Balanced_accuracy
        - F1-score
        - Precision
        - Recall
        - AUROC (area under the receiver operating characterisctic curve)
        - AUPRC (area under the precision recall curve)
    """
    
    fig, axs = plt.subplots(4, 2, figsize=(9, 9), sharex=True)

    comparar_metricas = [
        "time_seconds",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_roc_auc",
        "test_average_precision",
    ]

    nomes_metricas = [
        "Tempo (s)",
        "Acurácia",
        "Acurácia balanceada",
        "F1",
        "Precisão",
        "Recall",
        "AUROC",
        "AUPRC",
    ]

    for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        sns.boxplot(
            x="model",
            y=metrica,
            data=df_resultados,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(nome)
        ax.set_ylabel(nome)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.show()
