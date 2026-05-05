from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, pacf


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "overleaf" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def build_daily_count_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, usecols=["trans_date_trans_time"])
    dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    if dt.isna().any():
        raise ValueError("Found NaT in trans_date_trans_time while parsing timestamps.")

    daily_counts = dt.dt.floor("D").value_counts().sort_index()
    full_index = pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq="D")
    daily_counts = daily_counts.reindex(full_index, fill_value=0)
    daily_counts.index.name = "date"
    daily_counts.name = "tx_count"
    return daily_counts.astype(float)


def adf_summary(series: pd.Series) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    stat, pvalue, used_lag, nobs, critical, _icbest = result
    return {
        "adf_stat": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "critical_values": {k: float(v) for k, v in critical.items()},
    }


@dataclass(frozen=True)
class ModelRow:
    model: str
    aic: float
    bic: float


@dataclass(frozen=True)
class Unit6Summary:
    frequency: str
    start_date: str
    end_date: str
    n_days: int
    n_missing_filled: int
    missing_dates: list[str]
    decomposition_model: str
    seasonal_period: int
    adf_raw: dict
    adf_diff: dict
    differencing_order_d: int
    acf_significant_lags: list[int]
    pacf_significant_lags: list[int]
    suggested_p: int
    suggested_q: int
    candidates: list[ModelRow]
    selected_model: str
    ljung_box_pvalue: float
    test_horizon_days: int
    rmse: float
    mae: float
    mape_pct: float


def _significant_lags(values: np.ndarray, nobs: int, max_lag: int) -> list[int]:
    # 95% CI approx for ACF/PACF under white-noise: ±1.96/sqrt(N)
    conf = 1.96 / np.sqrt(nobs)
    lags: list[int] = []
    for lag in range(1, max_lag + 1):
        if abs(values[lag]) > conf:
            lags.append(lag)
    return lags


def main() -> None:
    series = build_daily_count_series(ROOT / "fraudTrain.csv")

    full_index = series.index
    observed_days = series[series > 0].index
    missing_dates = full_index.difference(observed_days)

    # 1) Raw series plot
    plt.figure(figsize=(11, 4))
    plt.plot(series.index, series.values, linewidth=1.0)
    plt.title("Daily Transaction Count (Raw Series)")
    plt.xlabel("Date")
    plt.ylabel("Transactions per day")
    _savefig(FIG_DIR / "fig7_raw_series.png")

    # 2) Decomposition (weekly seasonality, daily frequency)
    seasonal_period = 7
    # Using additive model: seasonal amplitude is roughly constant (counts do not scale exponentially).
    decomp = seasonal_decompose(series, model="additive", period=seasonal_period, extrapolate_trend="freq")
    fig = decomp.plot()
    fig.set_size_inches(11, 7)
    plt.suptitle("Seasonal Decomposition (Additive, Period = 7 days)", y=0.98)
    _savefig(FIG_DIR / "fig7_decomposition.png")

    # 3) Stationarity testing
    adf_raw = adf_summary(series)
    d = 0
    stationary = series.copy()
    while adf_raw["p_value"] > 0.05 and d < 3:
        d += 1
        stationary = stationary.diff().dropna()
        adf_raw = adf_summary(stationary)
        if adf_raw["p_value"] <= 0.05:
            break
    if d == 0:
        adf_series_raw = adf_summary(series)
        adf_series_diff = adf_series_raw
    else:
        adf_series_raw = adf_summary(series)
        adf_series_diff = adf_summary(stationary)

    # 4) ACF / PACF on stationary series
    max_lag = 30
    st = stationary.dropna()
    acf_vals = acf(st, nlags=max_lag, fft=True)
    pacf_vals = pacf(st, nlags=max_lag, method="ywm")
    sig_acf = _significant_lags(acf_vals, nobs=len(st), max_lag=max_lag)
    sig_pacf = _significant_lags(pacf_vals, nobs=len(st), max_lag=max_lag)

    # Heuristic: choose p as first cutoff-ish from PACF, q from ACF.
    suggested_p = min(sig_pacf[-1], 3) if sig_pacf else 0
    suggested_q = min(sig_acf[-1], 3) if sig_acf else 0

    plt.figure(figsize=(11, 4))
    plt.stem(range(0, max_lag + 1), acf_vals, basefmt=" ")
    conf = 1.96 / np.sqrt(len(st))
    plt.hlines([conf, -conf], xmin=0, xmax=max_lag, colors="r", linestyles="--", linewidth=1)
    plt.title("ACF (Stationary Series)")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    _savefig(FIG_DIR / "fig7_acf.png")

    plt.figure(figsize=(11, 4))
    plt.stem(range(0, max_lag + 1), pacf_vals, basefmt=" ")
    plt.hlines([conf, -conf], xmin=0, xmax=max_lag, colors="r", linestyles="--", linewidth=1)
    plt.title("PACF (Stationary Series)")
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    _savefig(FIG_DIR / "fig7_pacf.png")

    # 5) Fit candidate models (compare AIC)
    test_horizon = 30
    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]

    candidate_orders = [
        (max(0, min(suggested_p, 2)), d, max(0, min(suggested_q, 2))),
        (1, d, 1),
    ]
    # Ensure distinct candidates
    candidate_orders = list(dict.fromkeys(candidate_orders))

    candidates: list[ModelRow] = []
    fitted_models: dict[str, object] = {}

    for order in candidate_orders:
        name = f"ARIMA{order}"
        res = ARIMA(train, order=order).fit()
        candidates.append(ModelRow(model=name, aic=float(res.aic), bic=float(res.bic)))
        fitted_models[name] = res

    # Seasonal candidate: weekly seasonality
    sarima_name = f"SARIMA(1,{d},1)(1,0,1,7)"
    sarima = sm.tsa.statespace.SARIMAX(
        train,
        order=(1, d, 1),
        seasonal_order=(1, 0, 1, 7),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    candidates.append(ModelRow(model=sarima_name, aic=float(sarima.aic), bic=float(sarima.bic)))
    fitted_models[sarima_name] = sarima

    selected = min(candidates, key=lambda r: r.aic).model
    selected_res = fitted_models[selected]

    # 6) Residual diagnostics (Ljung-Box + QQ)
    resid = pd.Series(getattr(selected_res, "resid"), index=train.index).dropna()
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])

    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(221)
    ax1.plot(resid.index, resid.values, linewidth=1)
    ax1.set_title("Residuals (Time Plot)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Residual")

    ax2 = fig.add_subplot(222)
    ax2.hist(resid.values, bins=30)
    ax2.set_title("Residual Histogram")

    ax3 = fig.add_subplot(223)
    sm.graphics.tsa.plot_acf(resid.values, lags=30, ax=ax3)
    ax3.set_title("Residual ACF")

    ax4 = fig.add_subplot(224)
    sm.qqplot(resid.values, line="s", ax=ax4)
    ax4.set_title("Q-Q Plot")
    _savefig(FIG_DIR / "fig7_residual_diagnostics.png")

    # 7) Forecast on test horizon + metrics
    if hasattr(selected_res, "get_forecast"):
        fc = selected_res.get_forecast(steps=len(test))
        pred_mean = fc.predicted_mean
        conf_int = fc.conf_int(alpha=0.05)
    else:
        # SARIMAX results also have get_forecast; this is a safety fallback.
        pred_mean = selected_res.forecast(steps=len(test))
        conf_int = None

    pred = pd.Series(np.asarray(pred_mean), index=test.index)
    rmse = float(np.sqrt(np.mean((test.values - pred.values) ** 2)))
    mae = float(np.mean(np.abs(test.values - pred.values)))
    denom = np.where(test.values == 0, np.nan, test.values)
    mape = float(np.nanmean(np.abs((test.values - pred.values) / denom)) * 100.0)

    plt.figure(figsize=(11, 4))
    plt.plot(train.index, train.values, label="Train", linewidth=1)
    plt.plot(test.index, test.values, label="Test (Actual)", linewidth=2)
    plt.plot(pred.index, pred.values, label="Forecast", linewidth=2)
    if conf_int is not None:
        plt.fill_between(
            test.index,
            conf_int.iloc[:, 0].to_numpy(),
            conf_int.iloc[:, 1].to_numpy(),
            alpha=0.2,
            label="95% CI",
        )
    plt.title(f"{selected}: {len(test)}-Day Forecast (Test Window)")
    plt.xlabel("Date")
    plt.ylabel("Transactions per day")
    plt.legend()
    _savefig(FIG_DIR / "fig7_forecast_test.png")

    summary = Unit6Summary(
        frequency="D",
        start_date=str(series.index.min().date()),
        end_date=str(series.index.max().date()),
        n_days=len(series.index),
        n_missing_filled=int((series == 0).sum()),
        missing_dates=[str(d.date()) for d in missing_dates],
        decomposition_model="additive",
        seasonal_period=seasonal_period,
        adf_raw=adf_series_raw,
        adf_diff=adf_series_diff,
        differencing_order_d=d,
        acf_significant_lags=sig_acf,
        pacf_significant_lags=sig_pacf,
        suggested_p=int(suggested_p),
        suggested_q=int(suggested_q),
        candidates=candidates,
        selected_model=selected,
        ljung_box_pvalue=lb_p,
        test_horizon_days=int(test_horizon),
        rmse=rmse,
        mae=mae,
        mape_pct=mape,
    )

    out_path = ROOT / "unit6_summary.json"
    out_path.write_text(
        json.dumps(
            {
                **{k: v for k, v in asdict(summary).items() if k != "candidates"},
                "candidates": [asdict(r) for r in summary.candidates],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote figures to: {FIG_DIR}")
    print(f"Wrote summary to: {out_path}")


if __name__ == "__main__":
    main()

