# app/timeseries_engine.py
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import numpy as np


def train_and_forecast(df: pd.DataFrame, x_col: str, y_col: str, model_type: str = "baseline"):
    """
    Entraîne un modèle de régression sur une série temporelle simple.
    Args:
        df (pd.DataFrame): Données d'entrée.
        x_col (str): Nom de la colonne des features (ex: temps).
        y_col (str): Nom de la colonne à prédire.
        model_type (str): Type de modèle ("baseline" ou "linear").

    Returns:
        forecast_df (pd.DataFrame): DataFrame contenant les vraies valeurs et les prédictions.
        metrics (dict): Dictionnaire avec MAE et MAPE.
    """
    df = df[[x_col, y_col]].dropna()

    # Transformation X: si date ou string -> ordinal / float
    if np.issubdtype(df[x_col].dtype, np.datetime64):
        X = df[x_col].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    elif np.issubdtype(df[x_col].dtype, np.number):
        X = df[[x_col]].values
    else:
        # Si texte, encoder avec position
        X = np.arange(len(df)).reshape(-1, 1)

    y = df[y_col].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Modèle
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "xgboost":
        model = XGBRegressor(n_estimators=100, random_state=42)
    elif model_type == "arima":
        # ARIMA nécessite une série temporelle univariée (juste y)
        # et un index de type datetime (ou entier croissant)
        df_arima = df[[x_col, y_col]].copy()
        if not np.issubdtype(df_arima[x_col].dtype, np.datetime64):
            df_arima[x_col] = pd.to_datetime(df_arima[x_col])
        df_arima.set_index(x_col, inplace=True)
        y_series = df_arima[y_col]

        # Split temporel
        split_index = int(len(y_series) * 0.8)
        y_train, y_test = y_series[:split_index], y_series[split_index:]

        # Entraînement ARIMA simple (1,1,0)
        model = ARIMA(y_train, order=(1, 1, 0))
        model_fit = model.fit()

        # Prédiction
        y_pred = model_fit.forecast(steps=len(y_test))

        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        forecast_df = pd.DataFrame({
            x_col: y_test.index,
            "Actual": y_test.values,
            "Predicted": y_pred.values
        })

        return forecast_df, round(mae, 2), round(mape, 2)
    else:
        model = DummyRegressor(strategy="mean")



    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Évaluation
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    forecast_df = pd.DataFrame({
        x_col: X_test.flatten(),
        "Actual": y_test,
        "Predicted": y_pred
    })

    if np.issubdtype(df[x_col].dtype, np.datetime64):
        forecast_df[x_col] = forecast_df[x_col].apply(lambda x: pd.Timestamp.fromordinal(int(x)))

    return forecast_df, round(mae, 2), round(mape, 2)
