"""
IRC Vendaval — Dashboard de Resultados (Streamlit)
==================================================
Visualização interativa da correção de viés de rajadas de vento extremo.

Aba 1 — Explorador MLP:
  mapa de clusters + estações, série temporal, distribuição,
  scatter com regressão, métricas e importância de features.

Aba 2 — Screening LazyPredict:
  ranking dos 43 modelos avaliados por cluster.

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

TRAIN_PERIOD = ("2000-01-01", "2022-12-31")
VAL_PERIOD = ("2023-01-01", "2023-12-31")

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]
YAXIS_WIND = "Rajada máxima (m/s)"
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
    """Polígonos base de um cluster_id ('1-2-3' -> [1,2,3]; 4 -> [4])."""
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
    Varre ``base/exp*`` e monta a lista de experimentos. Fallback: usa o
    diretório plano ``base`` como experimento único "(raiz)".
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
        exps.append({"id": "(raiz)", "dir": str(base), "label": "(raiz)"})
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

# Experimentos MLP — descoberta única (o seletor fica na aba, ver tab_mlp)
mlp_experiments = discover_experiments(ARTIFACTS, "mlp_cluster_results.csv")
_mlp_by_id = {e["id"]: e for e in mlp_experiments}
# Default: primeiro experimento (exp1 = baseline)
_mlp_default = mlp_experiments[0]["id"] if mlp_experiments else "(nenhum)"
if "mlp_exp_id" not in st.session_state:
    st.session_state["mlp_exp_id"] = _mlp_default

with st.sidebar:
    st.title("🌬️ IRC Vendaval")
    st.caption("Correção de viés de rajadas de vento extremo")
    st.divider()

    # O experimento é escolhido na aba MLP (st.session_state["mlp_exp_id"]).
    mlp_exp_id = st.session_state.get("mlp_exp_id", _mlp_default)
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
        "Estação",
        ["(nenhuma)"] + stations_in_cluster,
    )
    if selected_station == "(nenhuma)":
        selected_station = None

    st.divider()

    # Split temporal
    st.caption("**Split temporal**")
    st.caption(f"Treino: `{TRAIN_PERIOD[0]}` → `{TRAIN_PERIOD[1]}`")
    st.caption(f"Validação: `{VAL_PERIOD[0]}` → `{VAL_PERIOD[1]}`")

    has_counts = "n_train" in results_df.columns
    if has_counts:
        total_train = int(results_df["n_train"].sum())
        total_val = int(results_df["n_val"].sum())
        st.caption(
            f"Amostras treino: **{total_train:,}**  |  "
            f"Validação: **{total_val:,}**"
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
            f"Cluster {r.cluster_id}<br>Estações: {r.n_stations}"
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
            name="Estações",
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
            title="Selecione uma estação na sidebar ou no mapa",
            template="plotly_white", height=420,
            xaxis_title="Data", yaxis_title=YAXIS_WIND,
        )
        return fig

    df_st = preds_df[preds_df["estacao"] == estacao].sort_values("time")
    cid = df_st["cluster_id"].iloc[0] if len(df_st) else "?"
    r2 = df_st["y_true"].corr(df_st["y_pred"]) ** 2
    rmse = ((df_st["y_true"] - df_st["y_pred"]) ** 2).mean() ** 0.5

    fig.add_trace(go.Scatter(
        x=df_st["time"], y=df_st["y_true"],
        name="INMET observado",
        line={"color": "royalblue", "width": 1.5},
    ))
    fig.add_trace(go.Scatter(
        x=df_st["time"], y=df_st["y_pred"],
        name="MLP predito",
        line={"color": "darkorange", "width": 1.5},
    ))
    fig.add_trace(go.Scatter(
        x=df_st["time"], y=df_st["era5_wind_mag_max"],
        name="ERA5 raw",
        line={"color": "tomato", "width": 1, "dash": "dot"},
    ))
    fig.update_layout(
        title=(
            f"Estação {estacao} — Cluster {cid}"
            f"  |  R²={r2:.3f}  RMSE={rmse:.2f} m/s"
        ),
        xaxis_title="Data", yaxis_title=YAXIS_WIND,
        template="plotly_white", height=420,
        legend={"orientation": "h", "y": -0.18},
        margin={"t": 50, "b": 70},
    )
    return fig


def build_distribution(sel_cluster: int | None) -> go.Figure:
    if preds_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="predictions_by_station.csv não encontrado",
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
                "Valor: %{y:.2f} m/s<extra></extra>"
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
                "Valor: %{y:.2f} m/s<extra></extra>"
            ),
        ))

    p90 = preds_df["y_true"].quantile(0.9)
    fig.add_hline(
        y=p90, line_dash="dot", line_color="gray",
        annotation_text="P90 global", annotation_position="top right",
    )
    fig.update_layout(
        title=(
            "Distribuição por Cluster — INMET (esq.) vs MLP (dir.)"
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
            title="predictions_by_station.csv não encontrado",
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
                f"Cluster {cid} — regressão<br>"
                f"a={coeffs[0]:.2f}  b={coeffs[1]:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Observado × Predito (linha = regressão OLS)",
        xaxis_title=f"Observado — {YAXIS_WIND}",
        yaxis_title=f"Predito — {YAXIS_WIND}",
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
        title=f"Importância por Permutação — Cluster {cluster_id}",
        xaxis_title="Queda em R²",
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
        title=f"Top {TOP_N} — Cluster {cluster_id} (laranja = MLP)",
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
    """Barras agrupadas: best R² por cluster × experimento."""
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
        title="Melhor R² por cluster × experimento",
        xaxis_title="Cluster",
        yaxis_title="R² (best model)",
        template="plotly_white",
        height=420,
        legend={"orientation": "h", "y": -0.25},
        margin={"t": 55, "b": 80},
    )
    return fig


def build_lazy_delta_table(exps: list[dict], baseline_id: str) -> pd.DataFrame:
    """Tabela com ΔR² e ΔRMSE de cada experimento vs baseline por cluster."""
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
                "Experimento": exp_id,
                "Cluster": cid,
                "Melhor Modelo": modelo,
                "R²": round(r2, 4),
                "RMSE": round(rmse, 4),
                "ΔR²": round(r2 - r2_b, 4),
                "ΔRMSE": round(rmse - rmse_b, 4),
            })
    return pd.DataFrame(rows)


# ── Layout principal ──────────────────────────────────────────────────────────

st.title("Correção de Viés de Rajadas de Vento Extremo")

tab_mlp, tab_lazy, tab_dl = st.tabs(
    ["Explorador MLP", "Screening LazyPredict", "Deep Learning"]
)

# ── Aba 1: Explorador MLP ─────────────────────────────────────────────────────

with tab_mlp:

    # Seletor de experimento (grava em st.session_state["mlp_exp_id"],
    # lido pela sidebar para carregar os dados deste experimento).
    col_exp, _col_pad = st.columns([1, 2])
    with col_exp:
        st.selectbox(
            "Experimento (MLP)",
            [e["id"] for e in mlp_experiments] or ["(nenhum)"],
            format_func=lambda i: _mlp_by_id.get(i, {}).get("label", i),
            key="mlp_exp_id",
        )

    # Linha 1: Mapa + Série temporal
    col_map, col_ts = st.columns(2)

    with col_map:
        event = st.plotly_chart(
            build_map(selected_station),
            width="stretch",
            on_select="rerun",
            key="map_chart",
            selection_mode="points",
        )
        # Capturar clique em estação no mapa
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

    # Linha 2: Distribuição + Scatter
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

    # Linha 3: Métricas + Importância
    col_met, col_imp = st.columns(2)

    with col_met:
        metric = st.selectbox(
            "Métrica",
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
        "Screening de 43 modelos via LazyPredict por cluster espacial "
        "(treino 2000–2022 / validação 2023). "
        "O **MLPRegressor** (laranja) lidera em 3 de 6 clusters e aparece "
        "no top-3 em outros 2, justificando sua escolha como modelo principal.",
        icon="ℹ️",
    )

    lazy_experiments = discover_experiments(
        LAZY_DIR, "lazy_cluster_results.csv"
    )
    _lazy_by_id = {e["id"]: e for e in lazy_experiments}
    _lazy_ids = [e["id"] for e in lazy_experiments] or ["(nenhum)"]

    sub_ranking, sub_compare = st.tabs(
        ["Ranking por experimento", "Comparação entre experimentos"]
    )

    # ── Sub-aba: Ranking ──────────────────────────────────────────────────
    with sub_ranking:
        col_lexp, col_lcid = st.columns(2)
        with col_lexp:
            lazy_exp_id = st.selectbox(
                "Experimento (Lazy)",
                _lazy_ids,
                format_func=lambda i: _lazy_by_id.get(i, {}).get("label", i),
                key="lazy_exp_sel",
            )

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
                "Execute ao menos dois experimentos para comparar.",
                icon="ℹ️",
            )
        else:
            col_base, _ = st.columns([1, 2])
            with col_base:
                baseline_id = st.selectbox(
                    "Baseline (referência)",
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


with tab_dl:
    st.subheader("TR-LSTM — Deep Learning")
    dl_exps = discover_lstm_experiments(LSTM_DIR)
    if not dl_exps:
        st.info(
            f"Nenhum experimento LSTM em `{LSTM_DIR}`. Copie a pasta do "
            "experimento (ex.: `lstm_v2/`) para esse diretório."
        )
    else:
        by_id = {e["id"]: e for e in dl_exps}
        c1, c2 = st.columns([1, 1])
        with c1:
            dl_exp = st.selectbox(
                "Experimento (LSTM)", list(by_id), key="dl_exp_id"
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
            ["Curva de Loss", "Scatter Obs×Pred", "Arquitetura & Pesos"]
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
                        "Série de loss não disponível como dado neste "
                        "experimento — exibindo o PNG gerado no treino."
                    )
                    st.image(str(png), use_container_width=True)
                else:
                    st.warning("Sem loss_history para este cluster.")
            else:
                st.line_chart(
                    loss.set_index("epoch")[["train_loss", "val_loss"]]
                )
                best_ep = int(loss["val_loss"].idxmin()) + 1
                st.caption(
                    f"{len(loss)} épocas | melhor val = "
                    f"{loss['val_loss'].min():.4f} (época {best_ep})"
                )

        with sub_scatter:
            preds = load_lstm_preds(exp_dir, dl_cid)
            if preds.empty:
                st.warning("Sem predictions para este cluster.")
            else:
                split = st.radio(
                    "Split", ["val", "test", "train"],
                    horizontal=True, key="dl_split",
                )
                sub = preds[preds["split"] == split]
                if sub.empty:
                    st.info(f"Sem dados para o split '{split}'.")
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
                        xaxis_title="Observado (m/s)",
                        yaxis_title="Predito (m/s)",
                        height=520, margin=dict(l=10, r=10, t=30, b=10),
                    )
                    fig.update_yaxes(scaleanchor="x", scaleratio=1)
                    st.plotly_chart(fig, use_container_width=True)

        with sub_arch:
            arch = load_lstm_arch(exp_dir, dl_cid)
            if not arch:
                st.warning("Sem architecture json para este cluster.")
            else:
                st.markdown(
                    f"**{arch['model']}** — "
                    f"{arch['total_params']:,} parâmetros"
                )
                st.json(arch.get("hyperparams", {}), expanded=False)
                layers = pd.DataFrame([
                    {
                        "camada": ly["name"],
                        "shape": "×".join(map(str, ly["shape"])),
                        "params": ly["params"],
                        "média": round(ly["mean"], 4),
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
                        "Histograma de pesos — camada",
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
                        {"peso": centers, "contagem": lyr["hist_counts"]}
                    ).set_index("peso")
                    st.bar_chart(hist_df)

        st.divider()
        st.caption(
            "Modelo: **TR-LSTM** — ref: *LSTM and Transformer-based "
            "framework for bias correction of ERA5 hourly wind speeds*."
        )
