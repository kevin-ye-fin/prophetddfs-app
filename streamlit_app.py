import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Regressor Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("Regressor Predictor Analysis")
st.caption("Correlation analysis + NeuralProphet baseline vs enhanced forecasting")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snowflake_available() -> bool:
    try:
        return bool(st.secrets.get("snowflake"))
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def run_snowflake_query(sql_text: str) -> pd.DataFrame:
    import snowflake.connector
    cfg = st.secrets["snowflake"]
    conn = snowflake.connector.connect(
        user=cfg["user"],
        password=cfg["password"],
        account=cfg["account"],
        warehouse=cfg["warehouse"],
        database=cfg["database"],
        schema=cfg["schema"],
    )
    cur = conn.cursor()
    cur.execute(sql_text)
    df = cur.fetch_pandas_all()
    cur.close()
    conn.close()
    df.columns = [c.lower() for c in df.columns]
    return df


def load_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    else:
        st.error("Unsupported file type. Please upload .csv, .xlsx, or .xls")
        return pd.DataFrame()


def detect_date_columns(df: pd.DataFrame) -> list:
    date_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_cols.append(c)
        else:
            try:
                parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
                if parsed.notna().sum() > len(df) * 0.8:
                    date_cols.append(c)
            except Exception:
                pass
    return date_cols


def detect_numeric_columns(df: pd.DataFrame, exclude: list) -> list:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def aggregate_to_weekly(df: pd.DataFrame, ds_col: str, target: str, regressors: list,
                        agg_rules: dict) -> pd.DataFrame:
    tmp = df[[ds_col, target] + regressors].copy()
    tmp["week"] = tmp[ds_col].dt.to_period("W").apply(lambda x: x.start_time)
    agg_map = {target: agg_rules.get(target, "sum")}
    for r in regressors:
        agg_map[r] = agg_rules.get(r, "sum")
    agg_map["_day_count"] = (ds_col, "count")

    real_agg = {}
    for out_col, rule in agg_map.items():
        if out_col == "_day_count":
            real_agg[out_col] = rule
        else:
            real_agg[out_col] = (out_col, rule)

    wk = tmp.groupby("week").agg(**real_agg).reset_index()
    wk = wk[wk["_day_count"] == 7].drop(columns=["_day_count"])
    wk = wk.rename(columns={"week": "ds"}).sort_values("ds").reset_index(drop=True)
    return wk


# ---------------------------------------------------------------------------
# Phase 1 — Correlation
# ---------------------------------------------------------------------------

def run_phase1(df: pd.DataFrame, target: str, regressors: list, max_lag: int):
    metrics = [target] + regressors

    pearson = df[metrics].corr(method="pearson")
    spearman = df[metrics].corr(method="spearman")

    lag_rows = []
    for reg in regressors:
        for lag in range(0, max_lag + 1):
            shifted = df[reg].shift(lag)
            r = df[target].corr(shifted)
            lag_rows.append({"regressor": reg, "lag": lag, "pearson_r": round(r, 4)})
    lag_df = pd.DataFrame(lag_rows)

    diff_df = df[["ds"] + metrics].copy()
    for col in metrics:
        diff_df[f"{col}_diff1"] = diff_df[col].diff(1)
    diff_df = diff_df.dropna()
    diff_metrics = [f"{m}_diff1" for m in metrics]
    diff_pearson = diff_df[diff_metrics].corr(method="pearson")

    diff_lag_rows = []
    for reg in regressors:
        for lag in range(0, max_lag + 1):
            shifted = diff_df[f"{reg}_diff1"].shift(lag)
            r = diff_df[f"{target}_diff1"].corr(shifted)
            diff_lag_rows.append({"regressor": f"{reg} (WoW)", "lag": lag, "pearson_r": round(r, 4)})
    diff_lag_df = pd.DataFrame(diff_lag_rows)

    return pearson, spearman, lag_df, diff_pearson, diff_lag_df


# ---------------------------------------------------------------------------
# Phase 2 — NeuralProphet
# ---------------------------------------------------------------------------

def build_model(freq: str, seasonality_mode: str = "multiplicative",
                trend_reg: float = 0, n_changepoints: int = 15, **overrides):
    from neuralprophet import NeuralProphet
    params = dict(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=(freq == "D"),
        daily_seasonality=False,
        n_changepoints=n_changepoints,
        trend_reg=trend_reg,
        learning_rate=0.1,
    )
    params.update(overrides)
    m = NeuralProphet(**params)
    m.add_country_holidays("US")
    return m


def run_phase2(df: pd.DataFrame, target: str, regressors: list,
               freq: str, holdout: int, n_lags: int,
               seasonality_mode: str = "multiplicative",
               trend_reg: float = 0, n_changepoints: int = 15,
               progress_callback=None):
    np_df = df[["ds", target] + regressors].copy()
    np_df = np_df.rename(columns={target: "y"})

    if freq == "W":
        split_date = np_df["ds"].max() - pd.Timedelta(weeks=holdout)
    else:
        split_date = np_df["ds"].max() - pd.Timedelta(days=holdout)

    train = np_df[np_df["ds"] <= split_date].copy()
    test = np_df[np_df["ds"] > split_date].copy()

    if len(test) == 0:
        return None, None, "Holdout period too large — no test data."

    model_kw = dict(seasonality_mode=seasonality_mode, trend_reg=trend_reg,
                    n_changepoints=n_changepoints)

    configs = {}
    for k in range(1, len(regressors) + 1):
        for combo in combinations(regressors, k):
            key = " + ".join(combo)
            label = "+ " + " & ".join(combo)
            configs[key] = {"regressors": list(combo), "label": label}

    total_models = 1 + len(configs)
    step = 0

    def align_weekly(fc_df):
        fc_df = fc_df.copy()
        fc_df["ds"] = fc_df["ds"].dt.to_period("W").dt.start_time
        return fc_df

    m_base = build_model(freq, **model_kw)
    train_base = train[["ds", "y"]].copy()
    m_base.fit(train_base, freq=freq, progress=None)
    future_base = m_base.make_future_dataframe(train_base, periods=len(test))
    fc_base = m_base.predict(future_base)
    if freq == "W":
        fc_base = align_weekly(fc_base)
    fc_base = fc_base[fc_base["ds"] > split_date][["ds", "yhat1"]].copy()
    fc_base = fc_base.merge(test[["ds", "y"]], on="ds", how="inner")
    fc_base["abs_pct_error"] = ((fc_base["yhat1"] - fc_base["y"]) / fc_base["y"]).abs() * 100
    fc_base["abs_error"] = (fc_base["yhat1"] - fc_base["y"]).abs()
    base_mape = fc_base["abs_pct_error"].mean()
    base_mae = fc_base["abs_error"].mean()

    step += 1
    if progress_callback:
        progress_callback(step / total_models, f"Baseline done (MAPE {base_mape:.2f}%)")

    enhanced = {}
    for key, cfg in configs.items():
        m = build_model(freq, **model_kw)
        for reg in cfg["regressors"]:
            m.add_lagged_regressor(reg, n_lags=n_lags, normalize="minmax")
        cols = ["ds", "y"] + cfg["regressors"]
        train_m = train[cols].copy()
        m.fit(train_m, freq=freq, progress=None)
        full_m = np_df[cols].copy()
        future_m = m.make_future_dataframe(full_m, periods=0, n_historic_predictions=True)
        fc = m.predict(future_m)
        if freq == "W":
            fc = align_weekly(fc)
        fc = fc[fc["ds"] > split_date][["ds", "yhat1"]].copy()
        fc = fc.merge(test[["ds", "y"]], on="ds", how="inner")
        fc["abs_pct_error"] = ((fc["yhat1"] - fc["y"]) / fc["y"]).abs() * 100
        fc["abs_error"] = (fc["yhat1"] - fc["y"]).abs()
        mape = fc["abs_pct_error"].mean()
        mae = fc["abs_error"].mean()
        enhanced[key] = {"fc": fc, "mape": mape, "mae": mae, "label": cfg["label"]}

        step += 1
        if progress_callback:
            progress_callback(step / total_models, f"{cfg['label']} done (MAPE {mape:.2f}%)")

    summary_rows = [{"Model": "Baseline", "Regressors": "—", "MAPE (%)": round(base_mape, 2), "MAE": round(base_mae, 0)}]
    for key in configs:
        r = enhanced[key]
        reg_list = ", ".join(configs[key]["regressors"])
        summary_rows.append({
            "Model": r["label"],
            "Regressors": reg_list,
            "MAPE (%)": round(r["mape"], 2),
            "MAE": round(r["mae"], 0),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("MAPE (%)").reset_index(drop=True)
    summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))

    comparison = fc_base[["ds", "y", "yhat1"]].rename(columns={"yhat1": "baseline_fc"})
    for key in configs:
        r = enhanced[key]
        comparison = comparison.merge(
            r["fc"][["ds", "yhat1"]].rename(columns={"yhat1": f"{key}_fc"}),
            on="ds", how="inner",
        )
    comparison["baseline_pct_err"] = (comparison["baseline_fc"] - comparison["y"]) / comparison["y"] * 100
    for key in configs:
        comparison[f"{key}_pct_err"] = (comparison[f"{key}_fc"] - comparison["y"]) / comparison["y"] * 100

    return summary_df, comparison, None


# ---------------------------------------------------------------------------
# Phase 3 — Forward Forecast
# ---------------------------------------------------------------------------

def run_phase3(df: pd.DataFrame, target: str, regressors: list,
               freq: str, forecast_horizon: int, n_lags: int,
               summary_df: pd.DataFrame,
               seasonality_mode: str = "multiplicative",
               trend_reg: float = 0, n_changepoints: int = 15):
    """Retrain the best model from Phase 2 on all data, forecast forward."""
    best_row = summary_df.loc[summary_df["MAPE (%)"].idxmin()]
    best_model_name = best_row["Model"]
    best_regressors_str = best_row["Regressors"]
    best_mape = best_row["MAPE (%)"]

    is_baseline = (best_regressors_str == "—")
    best_regs = [] if is_baseline else [r.strip() for r in best_regressors_str.split(",")]

    cols = ["ds", target] + best_regs
    full_df = df[cols].copy().rename(columns={target: "y"})

    model_kw = dict(seasonality_mode=seasonality_mode, trend_reg=trend_reg,
                    n_changepoints=n_changepoints)
    m = build_model(freq, **model_kw)
    for reg in best_regs:
        m.add_lagged_regressor(reg, n_lags=n_lags, normalize="minmax")

    train_cols = ["ds", "y"] + best_regs
    m.fit(full_df[train_cols], freq=freq, progress=None)

    future = m.make_future_dataframe(full_df[train_cols], periods=forecast_horizon,
                                     n_historic_predictions=True)
    forecast = m.predict(future)

    if freq == "W":
        forecast["ds"] = forecast["ds"].dt.to_period("W").dt.start_time

    last_actual = full_df["ds"].max()

    hist = forecast[forecast["ds"] <= last_actual][["ds", "yhat1"]].copy()
    hist = hist.merge(full_df[["ds", "y"]], on="ds", how="inner")
    hist = hist.rename(columns={"y": "actual", "yhat1": "fitted"})

    fwd = forecast[forecast["ds"] > last_actual][["ds", "yhat1"]].copy()
    fwd = fwd.rename(columns={"yhat1": "forecast"})
    fwd = fwd.head(forecast_horizon)

    return hist, fwd, best_model_name, best_mape, best_regressors_str


# ---------------------------------------------------------------------------
# Sidebar — Data Input
# ---------------------------------------------------------------------------

HAS_SNOWFLAKE = _snowflake_available()

with st.sidebar:
    st.header("1. Data Input")

    source_options = ["Upload CSV / Excel"]
    if HAS_SNOWFLAKE:
        source_options.insert(0, "SQL Query (Snowflake)")

    input_mode = st.radio("Data source", source_options)

    if input_mode == "SQL Query (Snowflake)":
        sql_text = st.text_area("SQL Query", height=200, placeholder="SELECT ds, volume, paid_subs, signups FROM ...")
        if st.button("Run Query", type="primary", use_container_width=True):
            if not sql_text.strip():
                st.error("Enter a SQL query first.")
            else:
                with st.spinner("Querying Snowflake..."):
                    try:
                        result = run_snowflake_query(sql_text.strip())
                        st.session_state["raw_df"] = result
                        st.session_state["data_source"] = "snowflake"
                        st.success(f"Loaded {len(result):,} rows, {len(result.columns)} columns")
                    except Exception as e:
                        st.error(f"Query failed: {e}")
    else:
        uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            df_up = load_file(uploaded)
            if not df_up.empty:
                st.session_state["raw_df"] = df_up
                st.session_state["data_source"] = "file"
                st.success(f"Loaded {len(df_up):,} rows, {len(df_up.columns)} columns")


# ---------------------------------------------------------------------------
# Sidebar — Column Configuration
# ---------------------------------------------------------------------------

if "raw_df" in st.session_state:
    raw_df = st.session_state["raw_df"].copy()

    date_candidates = detect_date_columns(raw_df)
    if not date_candidates:
        st.sidebar.error("No date column detected. Ensure your data has a parseable date column.")
        st.stop()

    with st.sidebar:
        st.header("2. Configure Columns")

        ds_col = st.selectbox("Date column", date_candidates, index=0)
        raw_df[ds_col] = pd.to_datetime(raw_df[ds_col])

        numeric_cols = detect_numeric_columns(raw_df, exclude=[ds_col])
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns (1 target + 1 regressor).")
            st.stop()

        target_col = st.selectbox(
            "Target column — the metric you want to forecast",
            numeric_cols,
            index=0,
            help="This is the dependent variable (e.g., volume, revenue). The model will try to predict this.",
        )
        remaining = [c for c in numeric_cols if c != target_col]
        regressor_cols = st.multiselect(
            "Regressor columns — potential predictors",
            remaining,
            default=remaining,
            help="These are independent variables that may help predict the target. "
                 "Select multiple to test each one individually, all pairwise combos, and all combined.",
        )

        if not regressor_cols:
            st.warning("Select at least one regressor column.")
            st.stop()

        n_regs = len(regressor_cols)
        n_combos = sum(1 for k in range(1, n_regs + 1) for _ in combinations(regressor_cols, k))
        st.caption(f"Will train **{1 + n_combos}** models: 1 baseline + {n_combos} regressor combinations")

        st.header("3. Analysis Settings")

        freq = st.radio("Data frequency", ["Weekly", "Daily"], horizontal=True)
        freq_code = "W" if freq == "Weekly" else "D"

        with st.expander("Model configuration", expanded=False):
            seasonality_mode = st.selectbox("Seasonality mode", ["multiplicative", "additive"], index=0)
            trend_reg = st.number_input("Trend regularization", min_value=0.0, max_value=10.0,
                                        value=0.0, step=0.1,
                                        help="0 = off, 0.5-1.0 = smooth trend, higher = more conservative")
            n_changepoints = st.select_slider("Trend changepoints", options=[5, 10, 15, 20, 30], value=15,
                                              help="Fewer = smoother trend, more = flexible")

        agg_rules = {}
        if freq_code == "W":
            with st.expander("Weekly aggregation rules", expanded=False):
                st.caption("Choose how each column is aggregated when converting daily → weekly")
                for col in [target_col] + regressor_cols:
                    agg_rules[col] = st.selectbox(f"{col}", ["sum", "mean"], index=0, key=f"agg_{col}")

        if freq_code == "W":
            holdout = st.slider("Test holdout (weeks)", min_value=4, max_value=26, value=13)
            max_lag = st.slider("Max correlation lag (weeks)", min_value=4, max_value=20, value=12)
            n_lags = st.slider("NeuralProphet regressor lags (weeks)", min_value=1, max_value=12, value=4)
            forecast_horizon = st.slider("Forecast horizon (weeks ahead)", min_value=1, max_value=26, value=13)
        else:
            holdout = st.slider("Test holdout (days)", min_value=14, max_value=180, value=90)
            max_lag = st.slider("Max correlation lag (days)", min_value=7, max_value=60, value=28)
            n_lags = st.slider("NeuralProphet regressor lags (days)", min_value=1, max_value=30, value=14)
            forecast_horizon = st.slider("Forecast horizon (days ahead)", min_value=7, max_value=180, value=90)

        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    # -------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------

    work_df = raw_df[[ds_col, target_col] + regressor_cols].dropna().copy()
    work_df = work_df.rename(columns={ds_col: "ds"})
    work_df = work_df.sort_values("ds").reset_index(drop=True)

    if freq_code == "W" and agg_rules:
        renamed_rules = {}
        for col, rule in agg_rules.items():
            renamed_rules[col] = rule
        work_df = aggregate_to_weekly(work_df, "ds", target_col, regressor_cols, renamed_rules)

    st.subheader("Data Preview")
    st.dataframe(work_df.head(20), use_container_width=True)
    st.caption(f"{len(work_df):,} rows  |  {work_df['ds'].min().date()} → {work_df['ds'].max().date()}")

    # -------------------------------------------------------------------
    # Run Analysis (store results in session_state so they survive reruns)
    # -------------------------------------------------------------------

    if run_clicked:
        with st.spinner("Computing correlations..."):
            pearson, spearman, lag_df, diff_pearson, diff_lag_df = run_phase1(
                work_df, target_col, regressor_cols, max_lag
            )
        st.session_state["phase1"] = {
            "pearson": pearson, "spearman": spearman,
            "lag_df": lag_df, "diff_pearson": diff_pearson, "diff_lag_df": diff_lag_df,
        }

        progress_bar = st.progress(0, text="Training models...")

        def update_progress(frac, text):
            progress_bar.progress(frac, text=text)

        summary_df, comparison, error = run_phase2(
            work_df, target_col, regressor_cols,
            freq_code, holdout, n_lags,
            seasonality_mode=seasonality_mode,
            trend_reg=trend_reg,
            n_changepoints=n_changepoints,
            progress_callback=update_progress,
        )
        progress_bar.empty()

        st.session_state["phase2"] = {
            "summary_df": summary_df, "comparison": comparison, "error": error,
        }

        unit = "weeks" if freq_code == "W" else "days"
        if summary_df is not None:
            with st.spinner("Training best model on full dataset for forward forecast..."):
                hist, fwd, best_name, best_mape_val, best_regs_str = run_phase3(
                    work_df, target_col, regressor_cols,
                    freq_code, forecast_horizon, n_lags,
                    summary_df=summary_df,
                    seasonality_mode=seasonality_mode,
                    trend_reg=trend_reg,
                    n_changepoints=n_changepoints,
                )
            st.session_state["phase3"] = {
                "hist": hist, "fwd": fwd,
                "forecast_horizon": forecast_horizon, "unit": unit,
                "best_name": best_name, "best_mape": best_mape_val,
                "best_regs": best_regs_str,
            }

    # -------------------------------------------------------------------
    # Display results from session_state (persists across download clicks)
    # -------------------------------------------------------------------

    if "phase1" in st.session_state:
        p1 = st.session_state["phase1"]
        pearson, spearman = p1["pearson"], p1["spearman"]
        lag_df, diff_pearson, diff_lag_df = p1["lag_df"], p1["diff_pearson"], p1["diff_lag_df"]

        st.divider()
        st.header("Phase 1 — Correlation Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Pearson Correlation")
            st.dataframe(pearson.style.format("{:.4f}"), use_container_width=True)
        with col2:
            st.subheader("Spearman Correlation")
            st.dataframe(spearman.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Lagged Cross-Correlation")
        lag_unit = "weeks" if freq_code == "W" else "days"
        for reg in regressor_cols:
            sub = lag_df[lag_df["regressor"] == reg].sort_values("pearson_r", ascending=False)
            best = sub.iloc[0]
            st.write(f"**{reg}**: best lag = {int(best['lag'])} {lag_unit}, r = {best['pearson_r']:.4f}")

        lag_pivot = lag_df.pivot_table(index="lag", columns="regressor", values="pearson_r").reset_index()
        st.dataframe(lag_pivot.style.format("{:.4f}", subset=[c for c in lag_pivot.columns if c != "lag"]),
                      use_container_width=True)

        st.subheader("WoW Differenced Correlation")
        st.dataframe(diff_pearson.style.format("{:.4f}"), use_container_width=True)

        diff_pivot = diff_lag_df.pivot_table(index="lag", columns="regressor", values="pearson_r").reset_index()
        st.dataframe(diff_pivot.style.format("{:.4f}", subset=[c for c in diff_pivot.columns if c != "lag"]),
                      use_container_width=True)

    if "phase2" in st.session_state:
        p2 = st.session_state["phase2"]
        summary_df, comparison, error = p2["summary_df"], p2["comparison"], p2["error"]

        st.divider()
        st.header("Phase 2 — NeuralProphet: Baseline vs Enhanced")

        if error:
            st.error(error)
        else:
            st.subheader("Model Comparison")
            best_idx = summary_df["MAPE (%)"].idxmin()
            st.dataframe(
                summary_df.style.highlight_min(subset=["MAPE (%)"], color="#d4edda"),
                use_container_width=True,
                hide_index=True,
            )
            best_model = summary_df.loc[best_idx, "Model"]
            best_mape = summary_df.loc[best_idx, "MAPE (%)"]
            st.success(f"Best model: **{best_model}** (MAPE {best_mape:.2f}%)")

            st.subheader("Forecast vs Actuals")
            chart_df = comparison[["ds", "y", "baseline_fc"]].copy()
            fc_cols = [c for c in comparison.columns if c.endswith("_fc") and c != "baseline_fc"]
            for c in fc_cols:
                chart_df[c] = comparison[c]
            chart_df = chart_df.set_index("ds")
            st.line_chart(chart_df, use_container_width=True)

            st.subheader("Detailed Results")
            st.dataframe(comparison, use_container_width=True)

            csv = comparison.to_csv(index=False)
            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name="regressor_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if "phase3" in st.session_state:
        p3 = st.session_state["phase3"]
        hist, fwd = p3["hist"], p3["fwd"]
        forecast_horizon_saved = p3["forecast_horizon"]
        unit = p3["unit"]
        best_name = p3.get("best_name", "Baseline")
        best_mape_display = p3.get("best_mape", "—")
        best_regs = p3.get("best_regs", "—")

        st.divider()
        st.header("Phase 3 — Forward Forecast")
        st.caption(f"Best model: **{best_name}** (backtest MAPE: {best_mape_display}%) | "
                   f"Regressors: {best_regs} | "
                   f"Forecasting {forecast_horizon_saved} {unit} ahead")

        chart_hist = hist[["ds", "actual", "fitted"]].set_index("ds")
        chart_fwd = fwd[["ds", "forecast"]].set_index("ds")
        chart_combined = pd.concat([chart_hist, chart_fwd])

        st.subheader("Historical Fit + Forward Forecast")
        st.line_chart(chart_combined, use_container_width=True)

        col_h, col_f = st.columns(2)
        with col_h:
            st.subheader("Historical Fit")
            hist_display = hist.copy()
            hist_display["ds"] = hist_display["ds"].dt.strftime("%Y-%m-%d")
            st.dataframe(hist_display, use_container_width=True, height=300)
        with col_f:
            st.subheader(f"Forecast ({forecast_horizon_saved} {unit})")
            fwd_display = fwd.copy()
            fwd_display["ds"] = fwd_display["ds"].dt.strftime("%Y-%m-%d")
            st.dataframe(fwd_display, use_container_width=True, height=300)

        forecast_csv = fwd.to_csv(index=False)
        st.download_button(
            f"Download forecast ({forecast_horizon_saved} {unit}) as CSV",
            data=forecast_csv,
            file_name="forward_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("Load data using the sidebar to get started.")
    st.markdown("""
### How to use this app

**1. Load your data** via SQL query or CSV/Excel upload. Your data should have:
- A **date column** (e.g., `ds`, `date`, `week`)
- A **target column** — the metric you want to forecast (e.g., volume, revenue, orders)
- One or more **regressor columns** — potential predictors (e.g., paid subs, signups, promo spend)

**2. Configure columns** — pick which column is the target and which are regressors

**3. Run Analysis** — the app will:
- Compute correlation between each regressor and the target at various lags
- Train NeuralProphet models for **every combination** of regressors (each alone, all pairs, all triples, etc.)
- Rank all models by forecast accuracy (MAPE) so you can see which combination works best
""")
