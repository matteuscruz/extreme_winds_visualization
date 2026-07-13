"""
IRC Vendaval — Results Dashboard (Streamlit)
==================================================
Interactive visualization of extreme wind gust bias correction.

Tab 1 — MLP Explorer:
  cluster map + stations, time series, distribution,
  scatter with regression, metrics and feature importance.

Tab 2 — LazyPredict Screening:
  ranking of the 43 evaluated models per cluster.

Uso
---
streamlit run app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Configuração da página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="IRC Vendaval",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constantes ────────────────────────────────────────────────────────────────

ARTIFACTS = Path("artifacts/mlp_clusters")
LAZY_DIR = Path("artifacts/lazy_clusters")
LSTM_DIR = Path("artifacts/lstm_pytorch")
SHP_PATH = Path("dataset/shp/shp_vento.shp")

# Matriz de ablation (sincronizada de scripts/sync_ablation_to_dashboard.py no
# repo de pesquisa) — fonte única de "qual experimento" pras abas MLP Explorer
# e LazyPredict Screening, substituindo os antigos exp1-3/exp1-5 (baseline,
# superados pela matriz). Ver seletor global "Configuration" no sidebar.
ABLATION_DIR = Path("artifacts/ablation")
ABLATION_ARMS = ["original", "synthetic", "newfeatures", "all"]

TRAIN_PERIOD = ("2000-01-01", "2022-12-31")
VAL_PERIOD = ("2023-01-01", "2023-12-31")

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]
YAXIS_WIND = "Maximum Gust (m/s)"
TOP_N = 15

_MLP_PREDS_COLS = [
    "estacao", "time", "latitude", "longitude", "cluster_id",
    "y_true", "y_pred", "era5_wind_mag_max", "ratio_pred", "ratio_true",
]
_IMP_COLS = ["feature", "importance", "std", "cluster_id"]
_STATION_COLS = ["estacao", "latitude", "longitude", "cluster_id"]
_LAZY_COLS = [
    "cluster_id", "n_stations", "Model", "R2", "Adj_R2", "RMSE", "Time",
]


def _cluster_color(cid) -> str:
    return CLUSTER_COLORS.get(cid, PALETTE[0])


def _cluster_members(cid) -> list[int]:
    """Base polygons of a cluster_id ('1-2-3' -> [1,2,3]; 4 -> [4])."""
    s = str(cid)
    if "-" in s:
        return [int(x) for x in s.split("-") if x.strip().isdigit()]
    try:
        return [int(float(s))]
    except ValueError:
        return []


# ── Descoberta de experimentos ───────────────────────────────────────────────

def discover_experiments(base: Path, results_name: str) -> list[dict]:
    """
    Scans ``base/exp*`` and builds the list of experiments. Fallback: uses the
    flat directory ``base`` as a single experiment "(root)".
    """
    exps: list[dict] = []
    for d in base.glob("exp*"):
        if not d.is_dir() or not (d / results_name).exists():
            continue
        merge = None
        synthetic = None
        meta = d / "run_meta.json"
        if meta.exists():
            try:
                meta_d = json.loads(meta.read_text())
                merge = meta_d.get("cluster_merge")
                synthetic = meta_d.get("synthetic_csv")
            except (json.JSONDecodeError, OSError):
                merge, synthetic = None, None
        if synthetic:
            suffix = " — augmented (GAN)"
        elif merge:
            suffix = f" — merge {merge}"
        else:
            suffix = " — baseline"
        exps.append({"id": d.name, "dir": str(d), "label": d.name + suffix})

    def _exp_num(e: dict) -> int:
        m = re.search(r"\d+", e["id"])
        return int(m.group()) if m else 0

    exps.sort(key=_exp_num)

    if not exps and (base / results_name).exists():
        exps.append({"id": "(root)", "dir": str(base), "label": "(root)"})
    return exps


def discover_ablation_as_experiments(pipeline: str, results_name: str) -> list[dict]:
    """Mesmo formato de retorno de discover_experiments ({id, dir, label}),
    mas a lista de 'experimentos' são os 4 braços da matriz de ablation
    (artifacts/ablation/<pipeline>/<arm>/) em vez de exp1/exp2/exp3 —
    substitui a fonte, não mistura as duas (os antigos exp* nunca aparecem
    aqui)."""
    exps: list[dict] = []
    for arm in ABLATION_ARMS:
        d = ABLATION_DIR / pipeline / arm
        if (d / results_name).exists():
            exps.append({"id": arm, "dir": str(d), "label": arm})
    return exps


# ── Carregamento de dados (cacheado por experimento) ─────────────────────────

@st.cache_data
def load_geojson():
    gdf = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
    return json.loads(gdf.to_json())


@st.cache_data
def load_mlp(mlp_dir: str):
    d = Path(mlp_dir)

    res_path = d / "mlp_cluster_results.csv"
    results = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()

    imp_path = d / "feature_importance.csv"
    importance = (
        pd.read_csv(imp_path) if imp_path.exists()
        else pd.DataFrame(columns=_IMP_COLS)
    )

    st_path = d / "stations_metadata.csv"
    stations = (
        pd.read_csv(st_path) if st_path.exists()
        else pd.DataFrame(columns=_STATION_COLS)
    )

    preds_path = d / "predictions_by_station.csv"
    preds = (
        pd.read_csv(preds_path, parse_dates=["time"])
        if preds_path.exists()
        else pd.DataFrame(columns=_MLP_PREDS_COLS)
    )
    return results, importance, stations, preds


@st.cache_data
def load_lazy(lazy_csv: str):
    p = Path(lazy_csv)
    if not p.exists():
        return pd.DataFrame(columns=_LAZY_COLS)
    lazy = pd.read_csv(p).rename(columns={
        "R-Squared": "R2",
        "Adjusted R-Squared": "Adj_R2",
        "Time Taken": "Time",
    })
    for col in ("R2", "Adj_R2", "RMSE"):
        if col in lazy.columns:
            lazy[col] = lazy[col].round(4)
    if "Time" in lazy.columns:
        lazy["Time"] = lazy["Time"].round(3)
    return lazy


geojson_clusters = load_geojson()

# ── Sidebar ───────────────────────────────────────────────────────────────────

# Experimentos MLP — matriz de ablation (substitui os antigos exp1-3/exp1-5,
# ver discover_ablation_as_experiments). Seletor único no sidebar
# ("Configuration") dirige MLP Explorer e LazyPredict Screening ao mesmo tempo.
mlp_experiments = discover_ablation_as_experiments("mlp", "mlp_cluster_results.csv")
_mlp_by_id = {e["id"]: e for e in mlp_experiments}
_mlp_default = mlp_experiments[0]["id"] if mlp_experiments else "(none)"
if "global_ablation_arm" not in st.session_state:
    st.session_state["global_ablation_arm"] = _mlp_default

with st.sidebar:
    st.title("🌬️ IRC Vendaval")
    st.caption("Extreme wind gust bias correction")
    st.divider()

    # Controle único: dirige MLP Explorer e LazyPredict Screening pro MESMO
    # braço da matriz de ablation — não existe mais um seletor de experimento
    # independente dentro de cada aba (evita mostrar resultados diferentes
    # em abas diferentes).
    st.selectbox(
        "Configuration",
        ABLATION_ARMS,
        key="global_ablation_arm",
        help=(
            "Same configuration used by MLP Explorer and LazyPredict "
            "Screening below — original / synthetic / newfeatures / all."
        ),
    )
    mlp_exp_id = st.session_state["global_ablation_arm"]
    if mlp_exp_id not in _mlp_by_id:
        mlp_exp_id = _mlp_default

    results_df, importance_df, stations_df, preds_df = load_mlp(
        _mlp_by_id[mlp_exp_id]["dir"] if mlp_exp_id in _mlp_by_id
        else str(ARTIFACTS)
    )

    CLUSTER_IDS = (
        sorted(results_df["cluster_id"].tolist())
        if not results_df.empty else []
    )
    CLUSTER_COLORS = {
        cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(CLUSTER_IDS)
    }
    CLUSTER_MEMBERS = {cid: _cluster_members(cid) for cid in CLUSTER_IDS}

    selected_cluster = st.selectbox(
        "Cluster",
        CLUSTER_IDS,
        format_func=lambda c: f"Cluster {c}",
    )

    stations_in_cluster = stations_df[
        stations_df["cluster_id"] == selected_cluster
    ]["estacao"].tolist()

    selected_station = st.selectbox(
        "Station",
        ["(none)"] + stations_in_cluster,
    )
    if selected_station == "(none)":
        selected_station = None

    st.divider()

    # Split temporal
    st.caption("**Temporal Split**")
    st.caption(f"Train: `{TRAIN_PERIOD[0]}` → `{TRAIN_PERIOD[1]}`")
    st.caption(f"Validation: `{VAL_PERIOD[0]}` → `{VAL_PERIOD[1]}`")

    has_counts = "n_train" in results_df.columns
    if has_counts:
        total_train = int(results_df["n_train"].sum())
        total_val = int(results_df["n_val"].sum())
        st.caption(
            f"Train samples: **{total_train:,}**  |  "
            f"Validation: **{total_val:,}**"
        )


# ── Funções de plot ───────────────────────────────────────────────────────────

def build_map(sel_station: str | None = None) -> go.Figure:
    fig = go.Figure()

    # Expande cada cluster (possivelmente agregado) nos polígonos base
    # para casar com featureidkey "properties.cluster" ("01".."14").
    locations: list[str] = []
    zvals: list[int] = []
    texts: list[str] = []
    for i, r in enumerate(results_df.itertuples()):
        htext = (
            f"Cluster {r.cluster_id}<br>Stations: {r.n_stations}"
            f"<br>R²: {r.MLP_R2:.3f}<br>RMSE: {r.MLP_RMSE:.3f}"
        )
        for m in CLUSTER_MEMBERS.get(r.cluster_id, []):
            locations.append(f"{m:02d}")
            zvals.append(i)
            texts.append(htext)

    if locations:
        fig.add_trace(go.Choroplethmap(
            geojson=geojson_clusters,
            featureidkey="properties.cluster",
            locations=locations,
            z=zvals,
            colorscale="Viridis",
            zmin=0, zmax=max(len(CLUSTER_IDS) - 1, 1),
            marker_opacity=0.30,
            marker_line_width=1.2,
            marker_line_color="white",
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            showscale=False,
            name="Clusters",
        ))

    if not stations_df.empty:
        colors = [_cluster_color(c) for c in stations_df["cluster_id"]]
        sizes = [
            16 if sel_station == r.estacao else 10
            for r in stations_df.itertuples()
        ]
        symbols = [
            "star" if sel_station == r.estacao else "circle"
            for r in stations_df.itertuples()
        ]
        fig.add_trace(go.Scattermap(
            lat=stations_df["latitude"],
            lon=stations_df["longitude"],
            mode="markers",
            marker={"size": sizes, "color": colors, "symbol": symbols},
            customdata=stations_df[["estacao", "cluster_id"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cluster: %{customdata[1]}<br>"
                "Lat: %{lat:.2f}°  Lon: %{lon:.2f}°"
                "<extra></extra>"
            ),
            name="Stations",
        ))

    fig.update_layout(
        map={
            "style": "carto-positron",
            "center": {"lat": -28.5, "lon": -52.5},
            "zoom": 5,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=420,
        uirevision="map",
    )
    return fig


def build_timeseries(estacao: str | None) -> go.Figure:
    fig = go.Figure()
    if estacao is None or preds_df.empty:
        fig.update_layout(
            title="Select a station in the sidebar or on the map",
            template="plotly_white", height=420,
            xaxis_title="Date", yaxis_title=YAXIS_WIND,
        )
        return fig

    df_st = preds_df[preds_df["estacao"] == estacao].sort_values("time")
    cid = df_st["cluster_id"].iloc[0] if len(df_st) else "?"
    r2 = df_st["y_true"].corr(df_st["y_pred"]) ** 2
    rmse = ((df_st["y_true"] - df_st["y_pred"]) ** 2).mean() ** 0.5

    fig.add_trace(go.Scatter(
        x=df_st["time"], y=df_st["y_true"],
        name="Observed INMET",
        line={"color": "royalblue", "width": 1.5},
    ))
    fig.add_trace(go.Scatter(
        x=df_st["time"], y=df_st["y_pred"],
        name="Predicted MLP",
        line={"color": "darkorange", "width": 1.5},
    ))
    fig.add_trace(go.Scatter(
        x=df_st["time"], y=df_st["era5_wind_mag_max"],
        name="ERA5 raw",
        line={"color": "tomato", "width": 1, "dash": "dot"},
    ))
    fig.update_layout(
        title=(
            f"Station {estacao} — Cluster {cid}"
            f"  |  R²={r2:.3f}  RMSE={rmse:.2f} m/s"
        ),
        xaxis_title="Date", yaxis_title=YAXIS_WIND,
        template="plotly_white", height=420,
        legend={"orientation": "h", "y": -0.18},
        margin={"t": 50, "b": 70},
    )
    return fig


def build_distribution(sel_cluster: int | None) -> go.Figure:
    if preds_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="predictions_by_station.csv not found",
            template="plotly_white", height=400,
        )
        return fig

    fig = go.Figure()
    for cid in CLUSTER_IDS:
        df_c = preds_df[preds_df["cluster_id"] == cid]
        opacity = 1.0 if (sel_cluster is None or sel_cluster == cid) else 0.18
        label = f"C{cid}"

        fig.add_trace(go.Violin(
            x=[label] * len(df_c), y=df_c["y_true"],
            name="INMET" if cid == CLUSTER_IDS[0] else None,
            legendgroup="INMET",
            showlegend=cid == CLUSTER_IDS[0],
            side="negative",
            line_color="royalblue",
            fillcolor="rgba(65,105,225,0.35)",
            opacity=opacity,
            meanline_visible=True,
            hovertemplate=(
                f"Cluster {cid} — INMET<br>"
                "Value: %{y:.2f} m/s<extra></extra>"
            ),
        ))
        fig.add_trace(go.Violin(
            x=[label] * len(df_c), y=df_c["y_pred"],
            name="MLP" if cid == CLUSTER_IDS[0] else None,
            legendgroup="MLP",
            showlegend=cid == CLUSTER_IDS[0],
            side="positive",
            line_color="darkorange",
            fillcolor="rgba(255,140,0,0.35)",
            opacity=opacity,
            meanline_visible=True,
            hovertemplate=(
                f"Cluster {cid} — MLP<br>"
                "Value: %{y:.2f} m/s<extra></extra>"
            ),
        ))

    p90 = preds_df["y_true"].quantile(0.9)
    fig.add_hline(
        y=p90, line_dash="dot", line_color="gray",
        annotation_text="Global P90", annotation_position="top right",
    )
    fig.update_layout(
        title=(
            "Distribution by Cluster — INMET (left) vs MLP (right)"
        ),
        yaxis_title=YAXIS_WIND, xaxis_title="Cluster",
        violingap=0.1, violinmode="overlay",
        template="plotly_white", height=400,
        legend={"orientation": "h", "y": -0.18},
        margin={"t": 50, "b": 70},
    )
    return fig


def build_scatter(sel_cluster: int | None) -> go.Figure:
    if preds_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="predictions_by_station.csv not found",
            template="plotly_white", height=400,
        )
        return fig

    fig = go.Figure()
    g_min = preds_df[["y_true", "y_pred"]].min().min()
    g_max = preds_df[["y_true", "y_pred"]].max().max()
    ref = [g_min, g_max]

    fig.add_trace(go.Scatter(
        x=ref, y=ref, mode="lines", name="1:1",
        line={"color": "black", "dash": "dash", "width": 1.2},
    ))

    for cid in CLUSTER_IDS:
        df_c = preds_df[preds_df["cluster_id"] == cid].dropna(
            subset=["y_true", "y_pred"]
        )
        if df_c.empty:
            continue

        color = _cluster_color(cid)
        opacity = 1.0 if (
            sel_cluster is None or sel_cluster == cid
        ) else 0.10
        label = f"C{cid}"

        fig.add_trace(go.Scatter(
            x=df_c["y_true"], y=df_c["y_pred"],
            mode="markers", name=label, legendgroup=label,
            marker={"color": color, "size": 5, "opacity": opacity},
            hovertemplate=(
                f"Cluster {cid}<br>"
                "Obs: %{x:.2f} m/s<br>"
                "Pred: %{y:.2f} m/s<extra></extra>"
            ),
        ))

        coeffs = np.polyfit(df_c["y_true"], df_c["y_pred"], 1)
        x_line = np.array([df_c["y_true"].min(), df_c["y_true"].max()])
        r2_c = float(
            results_df.loc[results_df["cluster_id"] == cid, "MLP_R2"].iloc[0]
        )
        fig.add_trace(go.Scatter(
            x=x_line, y=np.polyval(coeffs, x_line),
            mode="lines",
            name=f"{label} reg (R²={r2_c:.2f})",
            legendgroup=label,
            opacity=opacity,
            line={"color": color, "width": 2},
            hovertemplate=(
                f"Cluster {cid} — regression<br>"
                f"a={coeffs[0]:.2f}  b={coeffs[1]:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Observed vs Predicted (line = OLS regression)",
        xaxis_title=f"Observed — {YAXIS_WIND}",
        yaxis_title=f"Predicted — {YAXIS_WIND}",
        template="plotly_white", height=400,
        legend={"orientation": "h", "y": -0.22, "font": {"size": 10}},
        margin={"t": 50, "b": 80},
    )
    return fig


def build_metrics_bar(metric: str) -> go.Figure:
    clusters = [f"C{c}" for c in results_df["cluster_id"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MLP", x=clusters,
        y=results_df[f"MLP_{metric}"], marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        name="ERA5", x=clusters,
        y=results_df[f"ERA5_{metric}"], marker_color="tomato",
    ))
    fig.update_layout(
        barmode="group",
        title=f"MLP vs ERA5 — {metric}",
        yaxis_title=metric,
        template="plotly_white", height=320,
        legend={"orientation": "h", "y": -0.25},
        margin={"t": 50, "b": 70},
    )
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    return fig


def build_importance(cluster_id: int) -> go.Figure:
    df_imp = (
        importance_df[importance_df["cluster_id"] == cluster_id]
        .sort_values("importance", ascending=True)
    )
    fig = go.Figure(go.Bar(
        x=df_imp["importance"], y=df_imp["feature"],
        orientation="h",
        error_x={"type": "data", "array": df_imp["std"], "visible": True},
        marker_color="steelblue",
    ))
    fig.add_vline(x=0, line_color="black", line_width=0.8)
    fig.update_layout(
        title=f"Permutation Importance — Cluster {cluster_id}",
        xaxis_title="Drop in R²",
        template="plotly_white", height=480,
        margin={"t": 50, "l": 160},
    )
    return fig


def build_lazy_bar(cluster_id: int) -> go.Figure:
    df_c = (
        lazy_df[lazy_df["cluster_id"] == cluster_id]
        .sort_values("R2", ascending=False)
        .head(TOP_N)
        .sort_values("R2", ascending=True)
    )
    is_mlp = df_c["Model"] == "MLPRegressor"
    colors = ["#ff7f0e" if m else "#1f77b4" for m in is_mlp]

    fig = go.Figure(go.Bar(
        x=df_c["R2"], y=df_c["Model"],
        orientation="h",
        marker_color=colors,
        text=df_c["R2"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {TOP_N} — Cluster {cluster_id} (orange = MLP)",
        xaxis_title="R²",
        template="plotly_white", height=520,
        margin={"t": 55, "l": 220, "r": 60},
        xaxis={"range": [
            max(0, df_c["R2"].min() - 0.05),
            min(1, df_c["R2"].max() + 0.10),
        ]},
    )
    return fig


def build_lazy_comparison(exps: list[dict]) -> go.Figure:
    """Grouped bars: best R² per cluster × experiment."""
    frames = []
    for e in exps:
        _csv = str(Path(e["dir"]) / "lazy_cluster_results.csv")
        df = load_lazy(_csv)
        if df.empty:
            continue
        best = (
            df.sort_values("R2", ascending=False)
            .groupby("cluster_id", sort=True)
            .first()
            .reset_index()[["cluster_id", "R2", "RMSE", "Model"]]
        )
        best["exp"] = e["label"]
        frames.append(best)

    if not frames:
        return go.Figure()

    combined = pd.concat(frames, ignore_index=True)
    fig = go.Figure()
    for exp_label in combined["exp"].unique():
        sub = combined[combined["exp"] == exp_label].sort_values("cluster_id")
        fig.add_trace(go.Bar(
            name=exp_label,
            x=[f"C{c}" for c in sub["cluster_id"]],
            y=sub["R2"],
            text=sub["R2"].apply(lambda v: f"{v:.3f}"),
            textposition="outside",
            hovertemplate=(
                "<b>" + exp_label + "</b><br>"
                "Cluster %{x}<br>R²: %{y:.4f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        title="Best R² per cluster × experiment",
        xaxis_title="Cluster",
        yaxis_title="R² (best model)",
        template="plotly_white",
        height=420,
        legend={"orientation": "h", "y": -0.25},
        margin={"t": 55, "b": 80},
    )
    return fig


def build_lazy_delta_table(exps: list[dict], baseline_id: str) -> pd.DataFrame:
    """Table with ΔR² and ΔRMSE for each experiment vs baseline per cluster."""
    dfs: dict[str, pd.DataFrame] = {}
    for e in exps:
        df = load_lazy(str(Path(e["dir"]) / "lazy_cluster_results.csv"))
        if df.empty:
            continue
        best = (
            df.sort_values("R2", ascending=False)
            .groupby("cluster_id", sort=True)
            .first()
            .reset_index()[["cluster_id", "R2", "RMSE", "Model"]]
        )
        dfs[e["id"]] = best

    if baseline_id not in dfs:
        return pd.DataFrame()

    base = dfs[baseline_id].set_index("cluster_id")
    rows = []
    for exp_id, best in dfs.items():
        best = best.set_index("cluster_id")
        for cid in sorted(best.index):
            _nan = float("nan")
            r2 = best.loc[cid, "R2"] if cid in best.index else _nan
            rmse = best.loc[cid, "RMSE"] if cid in best.index else _nan
            r2_b = base.loc[cid, "R2"] if cid in base.index else _nan
            rmse_b = base.loc[cid, "RMSE"] if cid in base.index else _nan
            modelo = (
                best.loc[cid, "Model"] if cid in best.index else ""
            )
            rows.append({
                "Experiment": exp_id,
                "Cluster": cid,
                "Best Model": modelo,
                "R²": round(r2, 4),
                "RMSE": round(rmse, 4),
                "ΔR²": round(r2 - r2_b, 4),
                "ΔRMSE": round(rmse - rmse_b, 4),
            })
    return pd.DataFrame(rows)


# ── Ganhos por experimento (matriz cluster × trimestre) ──────────────────────

SEASONS_ORDER = ["DJF", "MAM", "JJA", "SON"]
SEASON_PT = {
    "DJF": "Summer (DJF)", "MAM": "Autumn (MAM)",
    "JJA": "Winter (JJA)", "SON": "Spring (SON)",
}
# CVD-safe diverging scale, centered at R²=0 (red=bad ↔ blue=good)
R2_COLORSCALE = "RdBu"


def _gains_metric(frames: list[pd.DataFrame]) -> str:
    """UNIQUE metric for the whole comparison: Deploy R² only if ALL
    experiments have it (apples-to-apples comparison); otherwise raw R²."""
    all_have_deploy = all(
        "R2_deploy_mean" in f.columns and f["R2_deploy_mean"].notna().any()
        for f in frames
    )
    return "R2_deploy_mean" if all_have_deploy else "R-Squared"


def build_gains_long(
    exps: list[dict], force_raw: bool = False
) -> pd.DataFrame:
    """Best model by (experiment, cluster, season).

    Unique metric across the matrix (see _gains_metric). Experiments WITHOUT
    estratificação sazonal preenchem os 4 trimestres com o global model
    (marcado is_global=True → sufixo ** na exibição).

    ``force_raw`` forces raw R² ("R-Squared") across the matrix — used
    when there are deep learning experiments in the comparison (which only have R²
    validation), to keep everything apples-to-apples.
    """
    loaded = []
    for e in exps:
        csv = Path(e["dir"]) / "lazy_cluster_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if not df.empty:
            loaded.append((e, df))
    if not loaded:
        return pd.DataFrame()

    metric = (
        "R-Squared" if force_raw
        else _gains_metric([df for _, df in loaded])
    )

    rows: list[dict] = []
    for e, df in loaded:
        df = df.dropna(subset=[metric])
        has_season = (
            "season" in df.columns and df["season"].notna().any()
        )
        glob = df[df["season"].isna()] if has_season else df

        for cid in sorted(df["cluster_id"].astype(str).unique()):
            g_sub = glob[glob["cluster_id"].astype(str) == cid]
            g_best = (
                g_sub.loc[g_sub[metric].idxmax()] if not g_sub.empty else None
            )
            for season in SEASONS_ORDER:
                best, is_global = g_best, True
                if has_season:
                    s_sub = df[
                        (df["cluster_id"].astype(str) == cid)
                        & (df["season"] == season)
                    ]
                    if not s_sub.empty:
                        best = s_sub.loc[s_sub[metric].idxmax()]
                        is_global = False
                if best is None:
                    continue
                rows.append({
                    "exp": e["id"],
                    "cluster_id": cid,
                    "season": season,
                    "model": best["Model"],
                    "r2": float(best[metric]),
                    "is_global": is_global,
                    "metric": metric,
                })
    return pd.DataFrame(rows)


def build_gains_heatmap(long_df: pd.DataFrame, cid: str) -> go.Figure:
    """Heatmap experiments (rows) × seasons (columns) for a cluster."""
    sub = long_df[long_df["cluster_id"].astype(str) == str(cid)]
    exp_order = sorted(sub["exp"].unique(), key=lambda s: (
        int(m.group()) if (m := re.search(r"\d+", s)) else 0
    ))
    z, text, hover = [], [], []
    for exp in exp_order:
        zr, tr, hr = [], [], []
        for season in SEASONS_ORDER:
            cell = sub[(sub["exp"] == exp) & (sub["season"] == season)]
            if cell.empty:
                zr.append(None)
                tr.append("")
                hr.append("")
                continue
            r2 = cell.iloc[0]["r2"]
            flag = "**" if bool(cell.iloc[0]["is_global"]) else ""
            zr.append(r2)
            tr.append(f"{r2:.3f}{flag}")
            hr.append(
                f"<b>{exp}</b> — {SEASON_PT[season]}<br>"
                f"Model: {cell.iloc[0]['model']}<br>"
                f"Deploy R²: {r2:.3f}"
                + ("<br><i>global model (**)</i>" if flag else "")
            )
        z.append(zr)
        text.append(tr)
        hover.append(hr)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[SEASON_PT[s] for s in SEASONS_ORDER],
        y=exp_order,
        text=text,
        texttemplate="%{text}",
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=R2_COLORSCALE,
        zmid=0,
        colorbar={"title": "R²", "thickness": 14},
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        height=90 + 46 * len(exp_order),
        margin={"t": 30, "b": 40, "l": 80, "r": 20},
        yaxis={"autorange": "reversed"},
    )
    return fig


# ── Layout principal ──────────────────────────────────────────────────────────

st.title("Extreme Wind Gust Bias Correction")

tab_mlp, tab_lazy, tab_gains, tab_dl, tab_ablation = st.tabs(
    ["MLP Explorer", "LazyPredict Screening",
     "Gains per Experiment", "Deep Learning", "Model Comparison"]
)

# ── Aba 1: Explorador MLP ─────────────────────────────────────────────────────

with tab_mlp:

    # A configuração é escolhida no sidebar ("Configuration") — mesma fonte
    # usada pela aba LazyPredict Screening, pra garantir que as duas mostrem
    # sempre o mesmo resultado.
    st.caption(f"Configuration: **{mlp_exp_id}**")
    if mlp_exp_id not in _mlp_by_id:
        st.warning(f"No synced MLP data for arm '{mlp_exp_id}' yet.")

    # Row 1: Map + Time series
    col_map, col_ts = st.columns(2)

    with col_map:
        event = st.plotly_chart(
            build_map(selected_station),
            width="stretch",
            on_select="rerun",
            key="map_chart",
            selection_mode="points",
        )
        # Capture click on station in the map
        if (
            event
            and hasattr(event, "selection")
            and event.selection.points
        ):
            pt = event.selection.points[0]
            cd = pt.get("customdata")
            if cd and len(cd) >= 2:
                clicked_station = str(cd[0])
                if clicked_station in stations_in_cluster:
                    selected_station = clicked_station
                    st.rerun()

    with col_ts:
        st.plotly_chart(
            build_timeseries(selected_station),
            width="stretch",
        )

    # Row 2: Distribution + Scatter
    col_dist, col_scat = st.columns(2)

    with col_dist:
        st.plotly_chart(
            build_distribution(selected_cluster),
            width="stretch",
        )

    with col_scat:
        st.plotly_chart(
            build_scatter(selected_cluster),
            width="stretch",
        )

    # Row 3: Metrics + Importance
    col_met, col_imp = st.columns(2)

    with col_met:
        metric = st.selectbox(
            "Metric",
            ["RMSE", "RMSE_P90", "Bias_P90"],
            key="metric_sel",
        )
        st.plotly_chart(
            build_metrics_bar(metric),
            width="stretch",
        )

        # Tabela de métricas
        table_cols = [
            "cluster_id", "n_stations",
            *(["n_train", "n_val"] if "n_train" in results_df.columns else []),
            "MLP_R2", "MLP_RMSE", "MLP_Bias_P90",
            "ERA5_R2", "ERA5_RMSE", "ERA5_Bias_P90",
        ]
        available_cols = [c for c in table_cols if c in results_df.columns]
        display_df = results_df[available_cols].copy()
        float_cols = [
            c for c in available_cols
            if c not in ("cluster_id", "n_stations", "n_train", "n_val")
        ]
        display_df[float_cols] = display_df[float_cols].round(3)

        def _highlight_sel(row):
            c = "background-color: #d0e8ff; font-weight: bold"
            return [
                c if row["cluster_id"] == selected_cluster else ""
            ] * len(row)

        st.dataframe(
            display_df.style.apply(_highlight_sel, axis=1),
            width="stretch",
            hide_index=True,
        )

    with col_imp:
        st.plotly_chart(
            build_importance(selected_cluster),
            width="stretch",
        )

# ── Aba 2: Screening LazyPredict ──────────────────────────────────────────────

with tab_lazy:
    st.info(
        "Screening of 43 models via LazyPredict per spatial cluster "
        "(train 2000–2022 / validation 2023). "
        "The **MLPRegressor** (orange) leads in 3 out of 6 clusters and appears "
        "in the top-3 in 2 others, justifying its choice as the main model.",
        icon="ℹ️",
    )

    lazy_experiments = discover_ablation_as_experiments("lazy", "lazy_cluster_results.csv")
    _lazy_by_id = {e["id"]: e for e in lazy_experiments}
    _lazy_ids = [e["id"] for e in lazy_experiments] or ["(none)"]

    sub_ranking, sub_compare = st.tabs(
        ["Ranking per experiment", "Comparison between experiments"]
    )

    # ── Sub-aba: Ranking ──────────────────────────────────────────────────
    with sub_ranking:
        # Mesma configuração escolhida no sidebar ("Configuration") — igual à
        # aba MLP Explorer, pra sempre refletir o mesmo resultado.
        lazy_exp_id = st.session_state["global_ablation_arm"]
        st.caption(f"Configuration: **{lazy_exp_id}**")
        if lazy_exp_id not in _lazy_by_id:
            st.warning(f"No synced Lazy data for arm '{lazy_exp_id}' yet.")

        _lazy_csv = (
            str(
                Path(_lazy_by_id[lazy_exp_id]["dir"])
                / "lazy_cluster_results.csv"
            )
            if lazy_exp_id in _lazy_by_id
            else str(LAZY_DIR / "lazy_cluster_results.csv")
        )
        lazy_df = load_lazy(_lazy_csv)
        LAZY_CLUSTER_IDS = sorted(
            lazy_df["cluster_id"].unique().tolist()
        )

        col_lcid, _col_pad = st.columns(2)
        with col_lcid:
            lazy_cluster = st.selectbox(
                "Cluster",
                LAZY_CLUSTER_IDS,
                format_func=lambda c: f"Cluster {c}",
                key="lazy_cluster_sel",
            )

        col_lbar, col_ltbl = st.columns(2)

        with col_lbar:
            st.plotly_chart(
                build_lazy_bar(lazy_cluster),
                width="stretch",
            )

        with col_ltbl:
            df_c = (
                lazy_df[lazy_df["cluster_id"] == lazy_cluster]
                .sort_values("R2", ascending=False)
                .reset_index(drop=True)
            )
            df_c.insert(0, "Rank", range(1, len(df_c) + 1))

            def _highlight_mlp(row):
                color = "background-color: #fff3cd; font-weight: bold"
                return [
                    color if row["Model"] == "MLPRegressor" else ""
                ] * len(row)

            st.dataframe(
                df_c[["Rank", "Model", "R2", "Adj_R2", "RMSE", "Time"]]
                .style.apply(_highlight_mlp, axis=1),
                width="stretch",
                hide_index=True,
                height=530,
            )

    # ── Sub-aba: Comparação ───────────────────────────────────────────────
    with sub_compare:
        if len(lazy_experiments) < 2:
            st.info(
                "Run at least two experiments to compare.",
                icon="ℹ️",
            )
        else:
            col_base, _ = st.columns([1, 2])
            with col_base:
                baseline_id = st.selectbox(
                    "Baseline (reference)",
                    _lazy_ids,
                    format_func=lambda i: (
                        _lazy_by_id.get(i, {}).get("label", i)
                    ),
                    key="lazy_baseline_sel",
                )

            st.plotly_chart(
                build_lazy_comparison(lazy_experiments),
                use_container_width=True,
            )

            st.subheader("Delta vs baseline")
            delta_df = build_lazy_delta_table(
                lazy_experiments, baseline_id
            )

            def _color_delta(val, col):
                if col == "ΔR²":
                    return (
                        "color: green" if val > 0
                        else "color: red" if val < 0 else ""
                    )
                if col == "ΔRMSE":
                    return (
                        "color: green" if val < 0
                        else "color: red" if val > 0 else ""
                    )
                return ""

            styled = delta_df.style
            for c in ("ΔR²", "ΔRMSE"):
                if c in delta_df.columns:
                    styled = styled.map(
                        lambda v, col=c: _color_delta(v, col), subset=[c]
                    )

            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
            )


# ── Helpers Deep Learning (LSTM) ──────────────────────────────────────────────

def discover_lstm_experiments(base: Path) -> list[dict]:
    exps: list[dict] = []
    if not base.exists():
        return exps
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "csv" / "lstm_pytorch_results.csv").exists():
            exps.append({"id": d.name, "dir": str(d)})
    return exps


@st.cache_data
def load_lstm_results(exp_dir: str) -> pd.DataFrame:
    p = Path(exp_dir) / "csv" / "lstm_pytorch_results.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_lstm_loss(exp_dir: str, cid) -> pd.DataFrame:
    p = Path(exp_dir) / "loss_history" / f"loss_history_c{cid}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_lstm_preds(exp_dir: str, cid) -> pd.DataFrame:
    p = Path(exp_dir) / "predictions" / f"predictions_c{cid}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_lstm_arch(exp_dir: str, cid) -> dict:
    p = Path(exp_dir) / "architecture" / f"arch_c{cid}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def build_lstm_gains_rows(lstm_exps: list[dict]) -> list[dict]:
    """Rows in the schema of ``build_gains_long`` from LSTM results.

    Deep learning is not stratified by season → replicates the R² of
    validation (``val_R2``) in the 4 seasons (is_global=True → suffix **).
    Metric always raw validation R², comparable to ``R-Squared`` from
    LazyPredict (both evaluated on the validation split).
    """
    rows: list[dict] = []
    for e in lstm_exps:
        df = load_lstm_results(e["dir"])
        if df.empty or "val_R2" not in df.columns:
            continue
        for r in df.itertuples():
            cid = str(r.cluster_id)
            for season in SEASONS_ORDER:
                rows.append({
                    "exp": f"LSTM·{e['id']}",
                    "cluster_id": cid,
                    "season": season,
                    "model": "TR-LSTM",
                    "r2": float(r.val_R2),
                    "is_global": True,
                    "metric": "R-Squared",
                })
    return rows


with tab_gains:
    st.subheader("Gains per Experiment — best model per cluster × season")
    gains_exps = discover_experiments(LAZY_DIR, "lazy_cluster_results.csv")
    lstm_gains_exps = discover_lstm_experiments(LSTM_DIR)
    if not gains_exps and not lstm_gains_exps:
        st.info("No LazyPredict or LSTM experiments found.")
    else:
        # Deep learning in comparison → force raw validation R² for
        # all (LSTM lacks deploy R²), keeping apples-to-apples.
        long_df = build_gains_long(
            gains_exps, force_raw=bool(lstm_gains_exps)
        )
        lstm_rows = build_lstm_gains_rows(lstm_gains_exps)
        if lstm_rows:
            long_df = pd.concat(
                [long_df, pd.DataFrame(lstm_rows)], ignore_index=True
            )
        if long_df.empty:
            st.info("No metrics to display.")
        else:
            metric_used = long_df["metric"].iloc[0]
            metric_lbl = (
                "Deploy R² (monthly)" if metric_used == "R2_deploy_mean"
                else "Raw R² (validation)"
            )
            st.caption(
                f"Metric: **{metric_lbl}** (`{metric_used}`) — best model "
                "per cluster and climatic season. Compares LazyPredict/MLP "
                "with deep learning (**TR-LSTM**). Experiments **not** "
                "stratified by season (including LSTM) repeat the value "
                "of the global model across the 4 seasons, marked with **."
            )
            if lstm_gains_exps:
                st.caption(
                    "ℹ️ Since deep learning is included in the comparison, the entire matrix "
                    "uses raw validation R² — LSTM lacks deploy R², "
                    "so this is the common metric for all."
                )
            elif metric_used != "R2_deploy_mean":
                st.caption(
                    "ℹ️ Using raw R² because not all experiments have "
                    "deploy R² — automatic switch to deploy when all "
                    "have it (apples-to-apples comparison)."
                )
            g_clusters = sorted(
                long_df["cluster_id"].astype(str).unique(),
                key=lambda s: (
                    int(m.group()) if (m := re.search(r"\d+", s)) else 0, s
                ),
            )
            g_sel = st.selectbox(
                "Cluster", g_clusters, format_func=lambda c: f"Cluster {c}",
                key="gains_cluster",
            )
            st.plotly_chart(
                build_gains_heatmap(long_df, g_sel),
                use_container_width=True,
            )
            st.caption(
                "Rows = experiments (top to bottom, most recent). "
                "Blue = higher R²; red = R² ≤ 0. "
                "** = global model metric applied to the season."
            )

            with st.expander("Detailed table — R² and model per season"):
                tbl = long_df.copy()
                tbl["Season"] = tbl["season"].map(SEASON_PT)
                tbl["R²"] = tbl.apply(
                    lambda r: f"{r['r2']:.3f}{'**' if r['is_global'] else ''}",
                    axis=1,
                )
                cols_order = [SEASON_PT[s] for s in SEASONS_ORDER]
                r2_piv = tbl.pivot_table(
                    index=["exp", "cluster_id"], columns="Season",
                    values="R²", aggfunc="first",
                ).reindex(columns=cols_order)
                mdl_piv = tbl.pivot_table(
                    index=["exp", "cluster_id"], columns="Season",
                    values="model", aggfunc="first",
                ).reindex(columns=cols_order)
                st.markdown("**Deploy R²**")
                st.dataframe(r2_piv, use_container_width=True)
                st.markdown("**Chosen model**")
                st.dataframe(mdl_piv, use_container_width=True)


with tab_dl:
    st.subheader("TR-LSTM — Deep Learning")
    dl_exps = discover_lstm_experiments(LSTM_DIR)
    if not dl_exps:
        st.info(
            f"No LSTM experiment in `{LSTM_DIR}`. Copy the experiment folder "
            "(e.g., `lstm_v2/`) to this directory."
        )
    else:
        by_id = {e["id"]: e for e in dl_exps}
        c1, c2 = st.columns([1, 1])
        with c1:
            dl_exp = st.selectbox(
                "Experiment (LSTM)", list(by_id), key="dl_exp_id"
            )
        exp_dir = by_id[dl_exp]["dir"]
        dl_results = load_lstm_results(exp_dir)
        clusters = (
            sorted(dl_results["cluster_id"].tolist())
            if not dl_results.empty else []
        )
        with c2:
            dl_cid = st.selectbox("Cluster", clusters, key="dl_cid")

        if not dl_results.empty and dl_cid is not None:
            row = dl_results[dl_results["cluster_id"] == dl_cid].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("val R²", f"{row.get('val_R2', float('nan')):.3f}")
            m2.metric("val RMSE", f"{row.get('val_RMSE', float('nan')):.2f}")
            m3.metric(
                "val Bias@P90", f"{row.get('val_Bias_P90', float('nan')):+.2f}"
            )
            m4.metric("test R²", f"{row.get('test_R2', float('nan')):.3f}")

        sub_loss, sub_scatter, sub_arch = st.tabs(
            ["Loss Curve", "Scatter Obs×Pred", "Architecture & Weights"]
        )

        with sub_loss:
            loss = load_lstm_loss(exp_dir, dl_cid)
            if loss.empty:
                png = (
                    Path(exp_dir) / "plots" / "per_cluster"
                    / f"loss_curve_{dl_cid}.png"
                )
                if png.exists():
                    st.caption(
                        "Loss series not available as data in this "
                        "experiment — showing the PNG generated during training."
                    )
                    st.image(str(png), use_container_width=True)
                else:
                    st.warning("No loss_history for this cluster.")
            else:
                st.line_chart(
                    loss.set_index("epoch")[["train_loss", "val_loss"]]
                )
                best_ep = int(loss["val_loss"].idxmin()) + 1
                st.caption(
                    f"{len(loss)} epochs | best val = "
                    f"{loss['val_loss'].min():.4f} (epoch {best_ep})"
                )

        with sub_scatter:
            preds = load_lstm_preds(exp_dir, dl_cid)
            if preds.empty:
                st.warning("No predictions for this cluster.")
            else:
                split = st.radio(
                    "Split", ["val", "test", "train"],
                    horizontal=True, key="dl_split",
                )
                sub = preds[preds["split"] == split]
                if sub.empty:
                    st.info(f"No data for split '{split}'.")
                else:
                    lo = float(min(sub["y_true"].min(), sub["y_pred"].min()))
                    hi = float(max(sub["y_true"].max(), sub["y_pred"].max()))
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=sub["y_true"], y=sub["y_pred"], mode="markers",
                        marker=dict(size=5, opacity=0.4, color="#1f77b4"),
                        name=split,
                    ))
                    fig.add_trace(go.Scatter(
                        x=[lo, hi], y=[lo, hi], mode="lines",
                        line=dict(dash="dash", color="black"), name="1:1",
                    ))
                    fig.update_layout(
                        xaxis_title="Observed (m/s)",
                        yaxis_title="Predicted (m/s)",
                        height=520, margin=dict(l=10, r=10, t=30, b=10),
                    )
                    fig.update_yaxes(scaleanchor="x", scaleratio=1)
                    st.plotly_chart(fig, use_container_width=True)

        with sub_arch:
            arch = load_lstm_arch(exp_dir, dl_cid)
            if not arch:
                st.warning("No architecture json for this cluster.")
            else:
                st.markdown(
                    f"**{arch['model']}** — "
                    f"{arch['total_params']:,} parameters"
                )
                st.json(arch.get("hyperparams", {}), expanded=False)
                layers = pd.DataFrame([
                    {
                        "layer": ly["name"],
                        "shape": "×".join(map(str, ly["shape"])),
                        "params": ly["params"],
                        "mean": round(ly["mean"], 4),
                        "std": round(ly["std"], 4),
                    }
                    for ly in arch["layers"]
                ])
                st.dataframe(
                    layers, use_container_width=True, hide_index=True
                )
                hist_layers = [
                    ly for ly in arch["layers"] if "hist_counts" in ly
                ]
                if hist_layers:
                    sel = st.selectbox(
                        "Weights histogram — layer",
                        [ly["name"] for ly in hist_layers],
                        key="dl_hist_layer",
                    )
                    lyr = next(
                        ly for ly in hist_layers if ly["name"] == sel
                    )
                    edges = lyr["hist_edges"]
                    centers = [
                        round((edges[i] + edges[i + 1]) / 2, 4)
                        for i in range(len(edges) - 1)
                    ]
                    hist_df = pd.DataFrame(
                        {"weight": centers, "count": lyr["hist_counts"]}
                    ).set_index("weight")
                    st.bar_chart(hist_df)

        st.divider()
        st.caption(
            "Model: **TR-LSTM** — ref: *LSTM and Transformer-based "
            "framework for bias correction of ERA5 hourly wind speeds*."
        )


# ── Helpers Ablation Study (matriz 3 pipelines × 4 braços) ───────────────────
# Dados sincronizados do repo de pesquisa via
# scripts/sync_ablation_to_dashboard.py (results.csv/predictions.csv -> Parquet),
# schema tidy: pipeline, experiment, cluster_id, season, split, n_samples,
# R2, RMSE, Bias, Bias_P90, RMSE_P90 — diferente do schema wide legado usado
# pelas outras abas (mlp_cluster_results.csv / lazy_cluster_results.csv /
# lstm_pytorch_results.csv), por isso ganha aba e loaders próprios.

ABLATION_PIPELINES = ["lazy", "mlp", "lstm"]
ABLATION_METRICS = ["R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90"]
# "ALL" = agregado do ano inteiro (já calculado pela pipeline); nunca deve ser
# misturado com os trimestres individuais no mesmo agregado ponderado.
ABLATION_SEASONS_ORDER = ["ALL", "DJF", "MAM", "JJA", "SON"]
ABLATION_METRIC_DIRECTIONS = {
    "R2": "higher", "RMSE": "lower", "Bias": "zero",
    "Bias_P90": "zero", "RMSE_P90": "lower",
}
# Mesma paleta de scripts/_ablation_common.py no repo de pesquisa.
ARM_COLORS = {
    "original": "#2a78d6", "synthetic": "#1baf7a",
    "newfeatures": "#c98a1f", "all": "#4a3aa7",
}
PIPELINE_LABELS = {"lazy": "LazyPredict", "mlp": "MLP", "lstm": "LSTM (TF dual-head)"}


def discover_ablation_combos() -> list[dict]:
    """Grade fixa 3×4 — sem ambiguidade de glob, só confere o que já foi sincronizado."""
    combos = []
    for pipeline in ABLATION_PIPELINES:
        for arm in ABLATION_ARMS:
            d = ABLATION_DIR / pipeline / arm
            if (d / "results.parquet").exists():
                combos.append({"pipeline": pipeline, "arm": arm, "dir": str(d)})
    return combos


@st.cache_data
def load_ablation_results(combo_dir: str) -> pd.DataFrame:
    path = Path(combo_dir) / "results.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_ablation_predictions(combo_dir: str) -> pd.DataFrame:
    path = Path(combo_dir) / "predictions.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _ablation_select_split(df: pd.DataFrame) -> pd.DataFrame:
    """Por (pipeline, arm), prefere split='test'; senão usa o único disponível."""
    if df.empty:
        return df
    frames = []
    for _, g in df.groupby(["pipeline", "arm"]):
        splits = set(g["split"].unique())
        chosen = "test" if "test" in splits else sorted(splits)[0]
        frames.append(g[g["split"] == chosen])
    return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0]


# LazyPredict é a única pipeline que varia o tipo de modelo por cluster/trimestre
# (results.csv já traz o vencedor de ~30 candidatos, escolhido pela própria
# pipeline). MLP/LSTM usam sempre o mesmo tipo de modelo — label fixa aqui só
# pra manter a coluna "Model" consistente nas 3 pipelines.
MODEL_FALLBACK_LABEL = {"mlp": "MLPRegressor", "lstm": "LSTM (TF dual-head)"}


def _model_summary(models: pd.Series) -> str:
    """String compacta com os modelos vencedores e quantos clusters cada um
    venceu, ex: 'CatBoostRegressor (3), TweedieRegressor (2), ...' —
    responde 'qual modelo teve a melhor métrica' de forma transparente em
    vez de esconder atrás de uma média."""
    counts = models.value_counts()
    return ", ".join(f"{name} ({n})" for name, n in counts.items())


def _ablation_aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Agrega sobre season (média ponderada por n_samples)."""
    if df.empty:
        return df
    rows = []
    for key, g in df.groupby(group_cols):
        w = g["n_samples"]
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        for m in ABLATION_METRICS:
            row[m] = float(np.average(g[m], weights=w)) if w.sum() > 0 else float("nan")
        if "model" in g.columns and g["model"].notna().any():
            row["Model"] = _model_summary(g["model"].dropna())
        else:
            pipeline = row.get("pipeline")
            row["Model"] = MODEL_FALLBACK_LABEL.get(pipeline, "—")
        rows.append(row)
    return pd.DataFrame(rows)


def load_all_ablation(combos: list[dict]) -> pd.DataFrame:
    frames = []
    for c in combos:
        df = load_ablation_results(c["dir"])
        if df.empty:
            continue
        df = df.copy()
        df["pipeline"] = c["pipeline"]
        df["arm"] = c["arm"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_ablation_bars(summary_df: pd.DataFrame, metric: str) -> go.Figure:
    """Barras agrupadas: metric por pipeline, cor = braço de ablation."""
    fig = go.Figure()
    if summary_df.empty:
        return fig
    pipelines = [p for p in ABLATION_PIPELINES if p in summary_df["pipeline"].unique()]
    arms = [a for a in ABLATION_ARMS if a in summary_df["arm"].unique()]
    for arm in arms:
        sub = summary_df[summary_df["arm"] == arm].set_index("pipeline")
        y = [sub[metric].get(p, float("nan")) for p in pipelines]
        models = [sub["Model"].get(p, "—") if "Model" in sub.columns else "—" for p in pipelines]
        fig.add_trace(go.Bar(
            name=arm,
            x=[PIPELINE_LABELS.get(p, p) for p in pipelines],
            y=y,
            marker_color=ARM_COLORS.get(arm, "#898781"),
            text=[f"{v:.3f}" if v == v else "" for v in y],
            textposition="outside",
            customdata=models,
            hovertemplate=(
                f"<b>{arm}</b><br>%{{x}}<br>{metric}: %{{y:.4f}}"
                "<br>Model(s): %{customdata}<extra></extra>"
            ),
        ))
    direction = ABLATION_METRIC_DIRECTIONS.get(metric, "higher")
    hint = {
        "higher": "higher is better", "lower": "lower is better",
        "zero": "closer to 0 is better",
    }[direction]
    fig.update_layout(
        barmode="group",
        title=f"{metric} by pipeline × configuration ({hint})",
        yaxis_title=metric,
        template="plotly_white", height=440,
        legend={"orientation": "h", "y": -0.22},
        margin={"t": 55, "b": 80},
    )
    return fig


def build_ablation_heatmap(summary_df: pd.DataFrame, metric: str) -> go.Figure:
    if summary_df.empty:
        return go.Figure()
    df = summary_df.copy()
    df["combo"] = df["pipeline"].map(lambda p: PIPELINE_LABELS.get(p, p)) + " / " + df["arm"]
    df = df.sort_values(["pipeline", "arm"])
    direction = ABLATION_METRIC_DIRECTIONS.get(metric, "higher")
    z = (
        df[metric] if direction == "higher"
        else (-df[metric] if direction == "lower" else -df[metric].abs())
    )
    fig = go.Figure(go.Heatmap(
        z=[z.tolist()],
        x=df["combo"].tolist(),
        y=[metric],
        colorscale=R2_COLORSCALE,
        text=[df[metric].round(3).tolist()],
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(
        title=f"{metric} — all 12 combinations",
        template="plotly_white", height=220,
        margin={"t": 45, "b": 90, "l": 60},
    )
    fig.update_xaxes(tickangle=45)
    return fig


def build_ablation_delta_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Delta de cada braço vs. o braço 'original' da MESMA pipeline."""
    if df.empty:
        return df
    rows = []
    for pipeline, g in df.groupby("pipeline"):
        g = g.set_index("arm")
        if "original" not in g.index:
            continue
        base = g.loc["original", metric]
        for arm in ABLATION_ARMS:
            if arm not in g.index:
                continue
            val = g.loc[arm, metric]
            rows.append({
                "Pipeline": PIPELINE_LABELS.get(pipeline, pipeline),
                "Arm": arm,
                "Model": g.loc[arm, "Model"] if "Model" in g.columns else "—",
                metric: round(val, 4),
                f"Δ{metric}": round(val - base, 4),
            })
    return pd.DataFrame(rows)


def build_ablation_scatter(pred_df: pd.DataFrame, pipeline: str, arm: str) -> go.Figure:
    fig = go.Figure()
    if pred_df.empty:
        fig.update_layout(
            title="No prediction rows available for this combination",
            template="plotly_white", height=480,
        )
        return fig
    fig.add_trace(go.Scatter(
        x=pred_df["y_true"], y=pred_df["y_pred"],
        mode="markers",
        marker={"size": 5, "color": ARM_COLORS.get(arm, "#898781"), "opacity": 0.5},
        hovertemplate="Observed: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        name=f"{pipeline}/{arm}",
    ))
    lo = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
    hi = max(pred_df["y_true"].max(), pred_df["y_pred"].max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line={"color": "#898781", "dash": "dash"}, name="y = x", showlegend=False,
    ))
    fig.update_layout(
        title=f"Observed vs. Predicted — {PIPELINE_LABELS.get(pipeline, pipeline)} / {arm}",
        xaxis_title="Observed (m/s)", yaxis_title="Predicted (m/s)",
        template="plotly_white", height=480,
        margin={"t": 50, "l": 10, "r": 10, "b": 10},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# ── Aba 5: Ablation Study ─────────────────────────────────────────────────────

with tab_ablation:
    ablation_combos = discover_ablation_combos()

    if not ablation_combos:
        st.info(
            "No results synced yet. Run "
            "`scripts/sync_ablation_to_dashboard.py` from the research repo "
            "after a pipeline finishes.",
            icon="ℹ️",
        )
    else:
        st.caption(
            f"{len(ablation_combos)}/12 combinations synced "
            "(3 pipelines × 4 configurations: original / synthetic / newfeatures / all)."
        )

        sub_compare, sub_drill = st.tabs(
            ["Comparison (all combinations)", "Single combination"]
        )

        with sub_compare:
            all_results = load_all_ablation(ablation_combos)
            available_seasons = [
                s for s in ABLATION_SEASONS_ORDER
                if s in all_results["season"].unique()
            ] if not all_results.empty else []

            available_clusters = (
                sorted(all_results["cluster_id"].unique(), key=str)
                if not all_results.empty else []
            )

            col_metric, col_season, col_cluster = st.columns([1, 1, 1])
            with col_metric:
                metric = st.selectbox(
                    "Metric", ABLATION_METRICS, key="ablation_metric"
                )
            with col_season:
                season = st.selectbox(
                    "Climate quarter (season)",
                    available_seasons or ["ALL"],
                    key="ablation_season",
                )
            with col_cluster:
                cluster_choice = st.selectbox(
                    "Cluster",
                    ["All clusters (weighted avg)"] + [f"Cluster {c}" for c in available_clusters],
                    key="ablation_cluster",
                    help=(
                        "The default blends all clusters via a weighted average, "
                        "which can hide strong individual clusters — pick one to "
                        "see its own metric, matching the 'Single combination' table."
                    ),
                )

            by_season = all_results[all_results["season"] == season]
            if by_season.empty:
                st.info(
                    f"No combination has data for season '{season}' yet — "
                    "pipelines currently differ in season coverage "
                    "(e.g. MLP only reports 'ALL' today).",
                    icon="ℹ️",
                )

            by_cluster = by_season
            if cluster_choice != "All clusters (weighted avg)":
                chosen_cluster = cluster_choice.removeprefix("Cluster ")
                by_cluster = by_season[by_season["cluster_id"].astype(str) == chosen_cluster]

            selected = _ablation_select_split(by_cluster)
            summary_df = _ablation_aggregate(selected, ["pipeline", "arm"])

            st.plotly_chart(
                build_ablation_bars(summary_df, metric),
                use_container_width=True,
                key="ablation_bars_chart",
            )
            st.plotly_chart(
                build_ablation_heatmap(summary_df, metric),
                use_container_width=True,
                key="ablation_heatmap_chart",
            )

            st.subheader("Delta vs. 'original' arm (same pipeline)")
            delta_df = build_ablation_delta_table(summary_df, metric)

            def _color_ablation_delta(val, metric_name):
                direction = ABLATION_METRIC_DIRECTIONS.get(metric_name, "higher")
                if direction == "higher":
                    return "color: green" if val > 0 else "color: red" if val < 0 else ""
                if direction == "lower":
                    return "color: green" if val < 0 else "color: red" if val > 0 else ""
                return "color: red" if abs(val) > 1e-9 else ""

            delta_col = f"Δ{metric}"
            styled = delta_df.style
            if delta_col in delta_df.columns:
                styled = styled.map(
                    lambda v: _color_ablation_delta(v, metric), subset=[delta_col]
                )
            st.dataframe(
                styled, use_container_width=True, hide_index=True,
                key="ablation_delta_table",
            )

        with sub_drill:
            combo_labels = {f"{c['pipeline']}/{c['arm']}": c for c in ablation_combos}
            combo_key = st.selectbox(
                "Combination",
                list(combo_labels.keys()),
                format_func=lambda k: (
                    f"{PIPELINE_LABELS.get(combo_labels[k]['pipeline'], combo_labels[k]['pipeline'])}"
                    f" — {combo_labels[k]['arm']}"
                ),
                key="ablation_combo_sel",
            )
            combo = combo_labels[combo_key]
            results_df_ablation = load_ablation_results(combo["dir"])
            preds_df_ablation = load_ablation_predictions(combo["dir"])

            drill_seasons = [
                s for s in ABLATION_SEASONS_ORDER
                if s in results_df_ablation["season"].unique()
            ] if not results_df_ablation.empty else []
            drill_season = st.selectbox(
                "Climate quarter (season)",
                ["(all)"] + drill_seasons,
                key="ablation_drill_season",
            )

            table_df = results_df_ablation
            scatter_preds = preds_df_ablation
            if drill_season != "(all)":
                table_df = table_df[table_df["season"] == drill_season]
                if not scatter_preds.empty and "season" in scatter_preds.columns:
                    scatter_preds = scatter_preds[scatter_preds["season"] == drill_season]

            st.dataframe(
                table_df.sort_values(["cluster_id", "season", "split"]),
                use_container_width=True, hide_index=True,
                key="ablation_drilldown_table",
            )
            st.plotly_chart(
                build_ablation_scatter(
                    scatter_preds, combo["pipeline"], combo["arm"]
                ),
                use_container_width=True,
                key="ablation_scatter_chart",
            )
