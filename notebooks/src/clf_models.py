import pandas as pd

from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def make_pipeline_classif_model(classifier, preprocessor=None):
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
