import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def make_pipeline_regression_model(
    regressor, preprocessor=None, target_transformer=None
):
    """
    This function creates a sklearn pipeline based on regression model.

    In
    -------
        regressor: type.
        preprocessor: Pipeline or ColumnTransformer. Optional, default = None. Preprocessing steps.
        target_transformer: type. Optional, default = None. Transformation to apply on target variable.

    Out
    -------
        model: Pipeline. Pipeline with the selected preprocessing, target transformer and regressor.
    """
    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("reg", regressor)])
    else:
        pipeline = Pipeline([("reg", regressor)])

    if target_transformer is not None:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model = pipeline
    return model


def train_validate_regression_model(
    X,
    y,
    regressor,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
):

    model = make_pipeline_regression_model(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=[
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
    )

    return scores


def grid_search_cv_regressor(
    regressor,
    param_grid,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
    return_train_score=False,
):
    model = make_pipeline_regression_model(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        model,
        cv=kf,
        param_grid=param_grid,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        refit="neg_root_mean_squared_error",
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
