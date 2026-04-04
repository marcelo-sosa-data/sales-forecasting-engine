"""
forecast_sales.py
Predicción de ventas con Prophet + comparación de modelos.
Ejecutar: python src/forecast_sales.py
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


def mape(y_true, y_pred):
    mask = np.array(y_true) != 0
    return np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask])) * 100


def train_prophet(df: pd.DataFrame, test_days: int = 30):
    """Entrena Prophet y evalúa en los últimos test_days días."""
    train = df.iloc[:-test_days].copy()
    test  = df.iloc[-test_days:].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(train[["ds","y"]])

    future   = model.make_future_dataframe(periods=test_days + 90)
    forecast = model.predict(future)

    # Métricas en test
    test_fc  = forecast[forecast["ds"].isin(test["ds"])]["yhat"].values
    mae_val  = mean_absolute_error(test["y"], test_fc)
    rmse_val = np.sqrt(mean_squared_error(test["y"], test_fc))
    mape_val = mape(test["y"], test_fc)

    print(f"📊 Prophet — MAE: {mae_val:,.0f} | RMSE: {rmse_val:,.0f} | MAPE: {mape_val:.2f}%")
    return model, forecast, {"MAE": mae_val, "RMSE": rmse_val, "MAPE": mape_val}


def plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Dashboard de forecast con bandas de confianza y componentes."""
    today    = pd.Timestamp.today()
    hist     = forecast[forecast["ds"] <= df["ds"].max()]
    future   = forecast[forecast["ds"] >  df["ds"].max()]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Forecast con Intervalo de Confianza 95%", "Descomposición — Trend"),
    )

    # Datos reales
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"], name="Ventas Reales",
        line=dict(color="white", width=1.5), mode="lines",
    ), row=1, col=1)

    # Forecast histórico
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist["yhat"], name="Ajuste Modelo",
        line=dict(color="#60a5fa", width=1, dash="dot"),
    ), row=1, col=1)

    # Proyección
    fig.add_trace(go.Scatter(
        x=future["ds"], y=future["yhat"], name="Proyección",
        line=dict(color="#f472b6", width=2.5),
    ), row=1, col=1)

    # Banda de confianza
    fig.add_trace(go.Scatter(
        x=pd.concat([future["ds"], future["ds"][::-1]]),
        y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(244,114,182,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="IC 95%",
    ), row=1, col=1)

    # Trend
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["trend"],
        name="Tendencia", line=dict(color="#34d399", width=2), showlegend=False,
    ), row=1, col=2)

    fig.add_vline(x=df["ds"].max(), line_dash="dash", line_color="yellow",
                  annotation_text="Hoy")

    fig.update_layout(
        title="📈 Sales Forecasting Dashboard — Prophet",
        template="plotly_dark", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


if __name__ == "__main__":
    df = pd.read_csv("data/sample/sales_daily_sample.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.rename(columns={"y": "y"})

    print("🔄 Entrenando modelo Prophet...")
    model, forecast, metrics = train_prophet(df, test_days=30)

    print("\n📊 Proyección próximos 90 días:")
    future_fc = forecast[forecast["ds"] > df["ds"].max()]
    print(f"  Promedio diario  : ${future_fc['yhat'].mean():,.0f}")
    print(f"  Total proyectado : ${future_fc['yhat'].sum():,.0f}")
    print(f"  Rango pesimista  : ${future_fc['yhat_lower'].mean():,.0f}/día")
    print(f"  Rango optimista  : ${future_fc['yhat_upper'].mean():,.0f}/día")

    fig = plot_forecast(df, forecast)
    fig.show()
