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
SHP_PATH = Path("dataset/shp/shp_vento.shp")

TRAIN_PERIOD = ("2000-01-01", "2022-12-31")
VAL_PERIOD = ("2023-01-01", "2023-12-31")

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]
YAXIS_WIND = "Rajada máxima (m/s)"
TOP_N = 15


def _cluster_color(cid: int) -> str:
    return PALETTE[(int(cid) - 1) % len(PALETTE)]


# ── Carregamento de dados (cacheado) ─────────────────────────────────────────

@st.cache_data
def load_data():
    results = pd.read_csv(ARTIFACTS / "mlp_cluster_results.csv")
    importance = pd.read_csv(ARTIFACTS / "feature_importance.csv")
    stations = pd.read_csv(ARTIFACTS / "stations_metadata.csv")
    lazy = pd.read_csv(LAZY_DIR / "lazy_cluster_results.csv")

    preds_path = ARTIFACTS / "predictions_by_station.csv"
    preds = (
        pd.read_csv(preds_path, parse_dates=["time"])
        if preds_path.exists()
        else pd.DataFrame(columns=[
            "estacao", "time", "latitude", "longitude",
            "cluster_id", "y_true", "y_pred",
            "era5_wind_mag_max", "ratio_pred", "ratio_true",
        ])
    )

    gdf = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
    geojson = json.loads(gdf.to_json())

    results["cluster_str"] = results["cluster_id"].apply(
        lambda x: f"{int(x):02d}"
    )
    stations["cluster_str"] = stations["cluster_id"].apply(
        lambda x: f"{int(x):02d}"
    )

    lazy = lazy.rename(columns={
        "R-Squared": "R2",
        "Adjusted R-Squared": "Adj_R2",
        "Time Taken": "Time",
    })
    for col in ("R2", "Adj_R2", "RMSE"):
        lazy[col] = lazy[col].round(4)
    lazy["Time"] = lazy["Time"].round(3)

    return results, importance, stations, preds, lazy, geojson


results_df, importance_df, stations_df, preds_df, lazy_df, geojson_clusters = (
    load_data()
)

CLUSTER_IDS = sorted(results_df["cluster_id"].tolist())
LAZY_CLUSTER_IDS = sorted(lazy_df["cluster_id"].unique().tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌬️ IRC Vendaval")
    st.caption("Correção de viés de rajadas de vento extremo")
    st.divider()

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

    hover_texts = [
        f"Cluster {r.cluster_id}<br>Estações: {r.n_stations}"
        f"<br>R²: {r.MLP_R2:.3f}<br>RMSE: {r.MLP_RMSE:.3f}"
        for r in results_df.itertuples()
    ]
    fig.add_trace(go.Choroplethmap(
        geojson=geojson_clusters,
        featureidkey="properties.cluster",
        locations=results_df["cluster_str"].tolist(),
        z=results_df["cluster_id"].tolist(),
        colorscale="Viridis",
        zmin=1, zmax=6,
        marker_opacity=0.30,
        marker_line_width=1.2,
        marker_line_color="white",
        text=hover_texts,
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
        label = f"C{int(cid)}"

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
        label = f"C{int(cid)}"

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
    clusters = [f"C{int(c)}" for c in results_df["cluster_id"]]
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


# ── Layout principal ──────────────────────────────────────────────────────────

st.title("IRC Vendaval — Correção de Viés de Rajadas de Vento Extremo")

tab_mlp, tab_lazy = st.tabs(["Explorador MLP", "Screening LazyPredict"])

# ── Aba 1: Explorador MLP ─────────────────────────────────────────────────────

with tab_mlp:

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
        st.dataframe(
            display_df.style.highlight_between(
                subset=["cluster_id"],
                left=selected_cluster,
                right=selected_cluster,
                color="#d0e8ff",
            ),
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
            return [color if row["Model"] == "MLPRegressor" else ""] * len(row)

        st.dataframe(
            df_c[["Rank", "Model", "R2", "Adj_R2", "RMSE", "Time"]]
            .style.apply(_highlight_mlp, axis=1),
            width="stretch",
            hide_index=True,
            height=530,
        )
