import pandas as pd

from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def make_pipeline_classif_model(classifier, preprocessor=None):
    """
    Creates a pipeline for a classification model with optional preprocessing.

    This function constructs a scikit-learn pipeline that includes an optional feature preprocessing 
    step followed by a classification model.

    Parameters:
    ----------
    classifier : estimator object
        A classification model instance (e.g., LogisticRegression, RandomForestClassifier) 
        implementing the scikit-learn estimator interface.

    preprocessor : transformer object, optional (default=None)
        A scikit-learn-compatible transformer or pipeline for preprocessing features 
        (e.g., StandardScaler, ColumnTransformer). If None, no preprocessing is applied.

    Returns:
    -------
    model : sklearn.pipeline.Pipeline
        A scikit-learn pipeline object that includes the preprocessing step (if provided) 
        followed by the classification model.
    """

    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("clf", classifier)])
    else:
        pipeline = Pipeline([("clf", classifier)])

    model = pipeline

    return model


def train_validate_classif_model(
    X,
    y,
    cv,
    classifier,
    preprocessor=None,
):
    """
    Trains and validates a classification model using cross-validation.

    This function sets up a pipeline for training and evaluating a classification model using 
    cross-validation. It evaluates several classification performance metrics, including accuracy, 
    balanced accuracy, F1 score, precision, recall, ROC AUC, and average precision.

    Parameters:
    ----------
    X : array-like
        The input features for training the model.

    y : array-like
        The target variable for training the model.

    cv : cross-validation generator or integer
        The cross-validation splitting strategy (e.g., KFold, StratifiedKFold) or number of folds.

    classifier : estimator object
        A classification model instance (e.g., LogisticRegression, RandomForestClassifier) 
        implementing the scikit-learn estimator interface.

    preprocessor : transformer object, optional (default=None)
        A scikit-learn-compatible transformer or pipeline for preprocessing features 
        (e.g., StandardScaler, ColumnTransformer). If None, no preprocessing is applied.

    Returns:
    -------
    scores : dict
        A dictionary with cross-validation scores for each metric, including:
        - accuracy, balanced_accuracy, f1, precision, recall, roc_auc, average_precision.
    """

    model = make_pipeline_classif_model(
        classifier,
        preprocessor,
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ],
    )

    return scores


def grid_search_cv_classifier(
    classifier,
    param_grid,
    cv,
    preprocessor=None,
    return_train_score=False,
    refit_metric="roc_auc",
):
    """
    Performs grid search with cross-validation for a classification model.

    This function sets up a classification pipeline, optionally including a preprocessing step, 
    and performs hyperparameter tuning using `GridSearchCV` with given cross-validation generator. 
    The grid search evaluates multiple classification metrics and refits the model based on a specified 
    metric (default is AUROC).

    Parameters:
    ----------
    classifier : estimator object
        A classification model instance (e.g., LogisticRegression, RandomForestClassifier) 
        implementing the scikit-learn estimator interface.

    param_grid : dict
        Dictionary with parameters names (including their pipeline prefixes, e.g., 
        "clf__C") as keys and lists of parameter settings to try as values.

    cv : cross-validation generator or integer
        The cross-validation splitting strategy (e.g., KFold, StratifiedKFold) or number of folds.

    preprocessor : transformer object, optional (default=None)
        A scikit-learn-compatible transformer or pipeline for preprocessing features 
        (e.g., StandardScaler, ColumnTransformer). If None, no preprocessing is applied.

    return_train_score : bool, optional (default=False)
        If True, training scores will be included in the results.

    refit_metric : str, optional (default="roc_auc")
        The metric to optimize for refitting the model after grid search. The possible values are
        "accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc" or "average_precision".

    Returns:
    -------
    sklearn.model_selection.GridSearchCV
        A fitted `GridSearchCV` object containing the best model, cross-validated results, and 
        access to the best hyperparameters.

    Notes:
    -----
    The scoring metrics used are:
        - Accuracy (`accuracy`)
        - Balanced Accuracy (`balanced_accuracy`)
        - F1 Score (`f1`)
        - Precision (`precision`)
        - Recall (`recall`)
        - AUROC (`roc_auc`)
        - AUPRC (`average_precision`)
    
    The model is refitted using the score that maximizes the specified `refit_metric` (default is `roc_auc`).
    """

    model = make_pipeline_classif_model(classifier, preprocessor)

    grid_search = GridSearchCV(
        model,
        cv=cv,
        param_grid=param_grid,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ],
        refit=refit_metric,
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search


def organize_results(results):
    """
    Processes the results from multiple model evaluations and prepares them for analysis.

    This function takes a dictionary of model evaluation results, where the keys are the names of the 
    models and the values are dictionaries containing performance metrics (fit time, score time, etc.),
    usually obtained from the `train_validate_classif_model` function. The function computes the total 
    time taken for each model by summing the 'fit_time' and 'score_time' fields, then organizes and 
    transforms the results into a pandas DataFrame.

    The resulting DataFrame contains the following:
    - A column for the model names.
    - Columns for each performance metric and the calculated total time.
    - A 'time_seconds' column, which sums 'fit_time' and 'score_time' for each model.

    The function also attempts to convert all columns in the DataFrame to numeric types where applicable.

    Parameters:
    ----------
    results : dict
        A dictionary where the keys are the names of the models and the values are 
        performance metrics such as 'fit_time', 'score_time', and scoring results,
        usually returned by `train_validate_classif_model` function.

    Returns:
    -------
    df_expanded_results : pandas.DataFrame
        A DataFrame containing the expanded results for each model, with one row 
        per metric and model, including calculated total time and individual 
        performance metrics.
    """

    for key in results.keys():
        results[key]["time_seconds"] = (
            results[key]["fit_time"] + results[key]["score_time"]
        )

    df_results = (
        pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    )

    df_expanded_results = df_results.explode(
        df_results.columns[1:].to_list()
    ).reset_index(drop=True)

    try:
        df_expanded_results = df_expanded_results.apply(pd.to_numeric)
    except ValueError:
        pass

    return df_expanded_results
