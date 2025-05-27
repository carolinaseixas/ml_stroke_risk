import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def make_pipeline_regression_model(
    regressor, preprocessor=None, target_transformer=None
):
    """
    Creates a pipeline for a regression model with optional preprocessing and 
    target transformation.

    This function creates a scikit-learn pipeline for a regression task. It allows 
    optional inclusion of a preprocessing step for input features and a transformer 
    for the target variable.

    Parameters:
    ----------
    regressor : estimator object
        A regression model instance (e.g., LinearRegression, RandomForestRegressor) 
        implementing the scikit-learn fit/predict interface.

    preprocessor : transformer object, optional (default=None)
        A scikit-learn-compatible transformer or pipeline for preprocessing features 
        (e.g., StandardScaler, ColumnTransformer). If None, no preprocessing is applied.

    target_transformer : transformer object, optional (default=None)
        A transformer for the target variable (e.g., power transform), 
        used for cases where the target distribution needs to be adjusted. 
        If None, no target transformation is applied.

    Returns:
    -------
    model : estimator object
        A scikit-learn estimator or pipeline that can be fit to data. If a target 
        transformer is provided, it returns a `TransformedTargetRegressor` wrapping 
        the pipeline.
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
    """
    Trains and validates a regression model using cross-validation.

    This function sets up a pipeline with optional preprocessing and 
    target transformation steps, then evaluates the model using K-fold 
    cross-validation. It returns multiple performance metrics for each fold.

    Parameters:
    ----------
    X : array-like or pandas DataFrame of shape (n_samples, n_features)
        The input features for training the model.

    y : array-like or pandas Series of shape (n_samples,)
        The target variable for training the model.

    regressor : estimator object
        A regression model instance (e.g., LinearRegression, RandomForestRegressor) 
        implementing the scikit-learn fit/predict interface.

    preprocessor : transformer object, optional (default=None)
        A scikit-learn-compatible transformer or pipeline for preprocessing features 
        (e.g., StandardScaler, ColumnTransformer). If None, no preprocessing is applied.

    target_transformer : transformer object, optional (default=None)
        A transformer for the target variable (e.g., power transform), 
        used for cases where the target distribution needs to be adjusted. 
        If None, no target transformation is applied.

    n_splits : int, optional (default=5)
        Number of folds for K-fold cross-validation.

    random_state : int, optional (default=42)
        Controls the randomness of the cross-validation splits.

    Returns:
    -------
    scores : dict
        A dictionary containing cross-validated scores with the following keys:
        - 'test_r2'
        - 'test_neg_mean_absolute_error'
        - 'test_neg_root_mean_squared_error'

    Notes:
    -----
    The scoring metrics follow scikit-learn's convention where some metrics 
    (e.g., MAE, RMSE) are returned as negative values to maintain consistency 
    in score maximization.
    """

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
    """
    Performs grid search with cross-validation for a regression model pipeline.

    This function sets up a full regression pipeline, optionally including a feature
    preprocessor and target transformer, and performs hyperparameter tuning using
    `GridSearchCV` with K-fold cross-validation. The grid search evaluates multiple
    scoring metrics and refits the model based on the negative root mean squared error.

    Parameters:
    ----------
    regressor : estimator object
        A regression model instance (e.g., LinearRegression, RandomForestRegressor) 
        implementing the scikit-learn estimator interface.

    param_grid : dict
        Dictionary with parameters names (including their pipeline prefixes, e.g., 
        "reg__alpha") as keys and lists of parameter settings to try as values.

    preprocessor : transformer object, optional (default=None)
        A scikit-learn-compatible transformer or pipeline for preprocessing features 
        (e.g., StandardScaler, ColumnTransformer). If None, no preprocessing is applied.

    target_transformer : transformer object, optional (default=None)
        A transformer for the target variable (e.g., power transform), 
        used for cases where the target distribution needs to be adjusted. 
        If None, no target transformation is applied.

    n_splits : int, optional (default=5)
        Number of folds to use in K-fold cross-validation.

    random_state : int, optional (default=42)
        Controls the shuffling applied to the data before splitting into batches.

    return_train_score : bool, optional (default=False)
        If True, training scores will be included in the results.

    Returns:
    -------
    grid_search : GridSearchCV instance
        A fitted `GridSearchCV` object containing the best model, cross-validated 
        results, and access to the best hyperparameters.
    
    Notes:
    -----
    The scoring metrics used are:
        - R² (`r2`)
        - Negative Mean Absolute Error (`neg_mean_absolute_error`)
        - Negative Root Mean Squared Error (`neg_root_mean_squared_error`)
    
    The model is refit using the score that minimizes RMSE (i.e., maximizes 
    `neg_root_mean_squared_error`).
    """

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
    """
    Processes the results from multiple model evaluations and prepares them for analysis.

    This function takes a dictionary of model evaluation results, where the keys are the names of the 
    models and the values are dictionaries containing performance metrics (fit time, score time, etc.),
    usually obtained from the `train_validate_regression_model` function. The function computes the total 
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
        usually returned by `train_validate_regression_model` function.

    Returns:
    -------
    df_expanded_results : pandas.DataFrame
        A DataFrame containing the expanded results for each model, with one row 
        per metric and model, including calculated total time and individual 
        performance metrics (R2, MAE, RMSE).
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
