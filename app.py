"""
IRC Vendaval — Results Dashboard (Streamlit)
==================================================
Unified view of the 18-combination ablation matrix (3 pipelines — LazyPredict,
MLP, TR-LSTM — × 6 feature configurations — original, synthetic, newfeatures,
all, basin, all_basin). Every visualization queries this same matrix via a
global sidebar control panel; no legacy pipeline (old exp1-5 baselines, the
PyTorch `lstm_pytorch` experiments) is used anywhere in this file.

1. Global Comparison Panel — all 18 combinations side by side.
2. Spatial & Temporal Error Inspector — map, observed/predicted density
   overlay and residual behaviour for the selected combinations.
3. Model Diagnostics & Explainability — scatter+regression or LSTM loss
   curve, plus permutation feature importance (MLP).

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
import shapely
import streamlit as st
import xarray as xr

from theme import PIPELINE_COLORS

# ── Configuração da página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="IRC Vendaval",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constantes ────────────────────────────────────────────────────────────────

SHP_PATH = Path("dataset/shp/shp_vento.shp")
# INMET_Stratified.nc sincronizado direto do repo de pesquisa (não passa por
# nenhuma pipeline) — fonte de verdade do "Observed (INMET)" no mapa, pra não
# ficar preso ao station-count/período de qualquer arm já treinado. Precisa
# ser ressincronizado manualmente sempre que o INMET for expandido de novo.
INMET_RAW_PATH = Path("dataset/raw/INMET_Stratified.nc")
INMET_TARGET_VAR = "daily_wind_gust_max"

# Matriz de ablation (sincronizada de scripts/sync_ablation_to_dashboard.py no
# repo de pesquisa) — ÚNICA fonte de dado deste dashboard. Nenhum pipeline
# legado (exp1-5 antigos, lstm_pytorch) é lido em lugar nenhum deste arquivo.
ABLATION_DIR = Path("artifacts/ablation")
ABLATION_PIPELINES = ["lazy", "mlp", "lstm"]
ABLATION_ARMS = ["original", "synthetic", "newfeatures", "all", "basin", "all_basin"]
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
    "basin": "#3d8b3d", "all_basin": "#8a5a2e",
}
PIPELINE_LABELS = {"lazy": "LazyPredict", "mlp": "MLP", "lstm": "LSTM (TF dual-head)"}
# LazyPredict é a única pipeline que varia o tipo de modelo por cluster/trimestre
# (results.csv já traz o vencedor de ~30 candidatos). MLP/LSTM usam sempre o
# mesmo tipo de modelo — label fixa só pra manter a coluna "Model" consistente.
MODEL_FALLBACK_LABEL = {"mlp": "MLPRegressor", "lstm": "LSTM (TF dual-head)"}

TRAIN_PERIOD = ("2000-01-01", "2022-12-31")
VAL_PERIOD = ("2023-01-01", "2023-12-31")

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]
YAXIS_WIND = "Maximum Gust (m/s)"
R2_COLORSCALE = "RdBu"

# Bounding boxes dos dois domínios ERA5 usados pelo projeto (coordenadas fixas,
# lidas direto dos .nc de origem no repo de pesquisa — não copiamos o netCDF
# de 86×77 pontos pra cá só pra desenhar um retângulo). "Basin" é o domínio
# novo (era5_basin_loader.py); "18UTC" é o antigo, restrito ao Paraná.
ERA5_BASIN_EXTENT = {"lat": (-35.0, -13.75), "lon": (-58.0, -39.0)}
ERA5_18UTC_EXTENT = {"lat": (-27.0, -22.0), "lon": (-55.0, -48.0)}


def _extent_ring(extent: dict) -> tuple[list[float], list[float]]:
    """4 cantos de um bounding box + fecha o laço, pra desenhar como linha."""
    lat_lo, lat_hi = extent["lat"]
    lon_lo, lon_hi = extent["lon"]
    lats = [lat_lo, lat_lo, lat_hi, lat_hi, lat_lo]
    lons = [lon_lo, lon_hi, lon_hi, lon_lo, lon_lo]
    return lats, lons

_MLP_PREDS_COLS = [
    "estacao", "time", "latitude", "longitude", "cluster_id",
    "y_true", "y_pred", "era5_wind_mag_max", "ratio_pred", "ratio_true",
]
_IMP_COLS = ["feature", "importance", "std", "cluster_id"]
_STATION_COLS = ["estacao", "latitude", "longitude", "cluster_id"]


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


# ── Descoberta da matriz de ablation ──────────────────────────────────────────

def discover_ablation_as_experiments(pipeline: str, results_name: str) -> list[dict]:
    """Localiza os braços (arms) de uma pipeline que têm um arquivo específico
    sincronizado — usado pra achar os CSVs legados-mas-dentro-do-combo do MLP
    (stations_metadata.csv, feature_importance.csv, predictions_by_station.csv),
    que só existem pro MLP mas vivem dentro da própria matriz de ablation."""
    exps: list[dict] = []
    for arm in ABLATION_ARMS:
        d = ABLATION_DIR / pipeline / arm
        if (d / results_name).exists():
            exps.append({"id": arm, "dir": str(d), "label": arm})
    return exps


def discover_ablation_combos() -> list[dict]:
    """Grade fixa 3×4 — sem ambiguidade de glob, só confere o que já foi sincronizado."""
    combos = []
    for pipeline in ABLATION_PIPELINES:
        for arm in ABLATION_ARMS:
            d = ABLATION_DIR / pipeline / arm
            if (d / "results.parquet").exists():
                combos.append({
                    "pipeline": pipeline, "arm": arm, "dir": str(d),
                    "label": f"{PIPELINE_LABELS.get(pipeline, pipeline)} / {arm}",
                })
    return combos


# ── Carregamento de dados (cacheado) ──────────────────────────────────────────

@st.cache_data
def load_geojson():
    gdf = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
    return json.loads(gdf.to_json())


@st.cache_data
def load_basin_geometries() -> tuple[dict, object]:
    """Geometrias reais dos 14 polígonos-base (shp_vento.shp, coluna 'cluster'
    zero-padded '01'..'14') — usadas pra recortar o campo interpolado (grade
    IDW) na forma real da bacia/cluster, em vez de um retângulo bounding-box.
    Retorna ({'01': geom, ...}, união de todas as 14 geometrias)."""
    gdf = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
    per_polygon = dict(zip(gdf["cluster"], gdf["geometry"]))
    basin_union = shapely.union_all(list(per_polygon.values()))
    return per_polygon, basin_union


def _geom_for_cluster_choice(cluster_choice: str | None):
    """Geometria (shapely) correspondente à escolha de 'Cluster focus':
    união dos polígonos-base do cluster escolhido, ou a bacia inteira se
    `cluster_choice` for None/"All clusters".

    `shapely.prepare()` é OBRIGATÓRIO aqui, não só uma otimização opcional:
    a união dos 14 polígonos tem ~364 mil coordenadas, e um
    `shapely.contains` NÃO preparado contra essa geometria custa ~9s pra uma
    grade 70×70 (medido) — inviável × 4 painéis por render. Preparado, cai
    pra ~0.06s. `st.cache_data` faz um round-trip de pickle antes de
    devolver o valor cacheado, o que DESFAZ o estado preparado (confirmado:
    `shapely.is_prepared` vira False depois do pickle) — por isso o prepare
    tem que rodar aqui FORA da função cacheada, a cada chamada."""
    per_polygon, basin_union = load_basin_geometries()
    if not cluster_choice or cluster_choice == "All clusters":
        geom = basin_union
    else:
        members = _cluster_members(cluster_choice)
        geoms = [per_polygon[f"{m:02d}"] for m in members if f"{m:02d}" in per_polygon]
        geom = shapely.union_all(geoms) if geoms else basin_union
    shapely.prepare(geom)
    return geom


@st.cache_data
def load_stations_geo() -> pd.DataFrame:
    """Referência geográfica das estações (INMET) — independente de pipeline,
    já que a rede física de estações/clusters é a mesma pras 3 pipelines.
    stations_metadata.csv é byte-idêntico nos 4 braços do MLP (verificado);
    carrega do primeiro braço sincronizado disponível."""
    for arm in ABLATION_ARMS:
        p = ABLATION_DIR / "mlp" / arm / "stations_metadata.csv"
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame(columns=_STATION_COLS)


@st.cache_data
def load_mlp_importance(combo_dir: str) -> pd.DataFrame:
    p = Path(combo_dir) / "feature_importance.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=_IMP_COLS)


@st.cache_data
def load_mlp_predictions_by_station(combo_dir: str) -> pd.DataFrame:
    """Carrega predictions_by_station.csv (estacao/lat/lon/time reais por
    observação) de qualquer pipeline — não é mais MLP-only: LazyPredict e
    LSTM também passaram a sincronizar esse arquivo (antes só reportavam
    agregado por cluster_id). O nome da função ficou por compatibilidade
    com os call sites existentes."""
    p = Path(combo_dir) / "predictions_by_station.csv"
    if not p.exists():
        return pd.DataFrame(columns=_MLP_PREDS_COLS)
    return pd.read_csv(p, parse_dates=["time"])


@st.cache_data
def load_ablation_results(combo_dir: str) -> pd.DataFrame:
    path = Path(combo_dir) / "results.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_ablation_predictions(combo_dir: str) -> pd.DataFrame:
    path = Path(combo_dir) / "predictions.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_lstm_histories(combo_dir: str) -> dict:
    """{cluster_id str: {season: {loss: [...], val_loss: [...]}}} — curva de
    convergência de treino/validação por cluster×season."""
    p = Path(combo_dir) / "histories.json"
    return json.loads(p.read_text()) if p.exists() else {}


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


def _model_summary(models: pd.Series) -> str:
    """String compacta com os modelos vencedores e quantos clusters cada um
    venceu, ex: 'CatBoostRegressor (3), TweedieRegressor (2), ...' —
    responde 'qual modelo teve a melhor métrica' de forma transparente em
    vez de esconder atrás de uma média."""
    counts = models.value_counts()
    return ", ".join(f"{name} ({n})" for name, n in counts.items())


def _resolve_all_season(all_results: pd.DataFrame) -> pd.DataFrame:
    """Linhas efetivas pra season='ALL': usa a linha nativa 'ALL' quando o
    combo já reporta uma (MLP/LazyPredict), e DERIVA 'ALL' pra quem só
    reporta por trimestre (LSTM só tem DJF/JJA/MAM/SON, nunca 'ALL') —
    agregando as 4 linhas de trimestre (split='test', ponderado por
    n_samples) por (pipeline, arm, cluster_id). Sem isso, combos sem linha
    'ALL' nativa somem inteiros da visão default do painel."""
    native = all_results[all_results["season"] == "ALL"]
    covered = set(zip(native["pipeline"], native["arm"]))
    quarterly = all_results[
        (all_results["season"] != "ALL")
        & ~all_results.apply(lambda r: (r["pipeline"], r["arm"]) in covered, axis=1)
    ]
    quarterly_test = _ablation_select_split(quarterly)
    if quarterly_test.empty:
        return native
    derived = _ablation_aggregate(quarterly_test, ["pipeline", "arm", "cluster_id"])
    if derived.empty:
        return native
    derived["season"] = "ALL"
    derived["split"] = "test"
    n_by_key = quarterly_test.groupby(["pipeline", "arm", "cluster_id"])["n_samples"].sum()
    derived["n_samples"] = [
        n_by_key.get((r.pipeline, r.arm, r.cluster_id), 0) for r in derived.itertuples()
    ]
    return pd.concat([native, derived], ignore_index=True)


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


# ── Seção 1 — Global Comparison Panel ─────────────────────────────────────────

def build_ablation_bars(summary_df: pd.DataFrame, metric: str) -> go.Figure:
    """Barras agrupadas: metric por pipeline, cor = configuração."""
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
        title=f"{metric} — all combinations",
        template="plotly_white", height=220,
        margin={"t": 45, "b": 90, "l": 60},
    )
    fig.update_xaxes(tickangle=45)
    return fig


def build_ablation_delta_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Delta de cada configuração vs. 'original' da MESMA pipeline."""
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


def build_lazy_top5_per_arm(
    combos: list[dict], season: str, cluster_choice: str
) -> pd.DataFrame:
    """Top-5 modelos do LazyPredict (por R²) pra cada configuração —
    lazy_cluster_results.csv tem uma linha por (modelo, cluster, season),
    então dá pra rankear os ~43 candidatos de verdade, não só o "vencedor"
    já reduzido no results.csv."""
    rows = []
    for c in combos:
        if c["pipeline"] != "lazy":
            continue
        p = Path(c["dir"]) / "lazy_cluster_results.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p).rename(columns={"R-Squared": "R2"})
        if "season" in df.columns:
            df = df[df["season"].isna()] if season == "ALL" else df[df["season"] == season]
        if df.empty:
            continue

        if cluster_choice != "All clusters (weighted avg)":
            chosen = cluster_choice.removeprefix("Cluster ")
            df = df[df["cluster_id"].astype(str) == chosen]
            top5 = df.sort_values("R2", ascending=False).head(5)[["Model", "R2"]]
        else:
            agg = (
                df.groupby("Model")
                .apply(lambda g: pd.Series({
                    "R2": np.average(g["R2"], weights=g["n_stations"])
                    if g["n_stations"].sum() > 0 else g["R2"].mean()
                }), include_groups=False)
                .reset_index()
            )
            top5 = agg.sort_values("R2", ascending=False).head(5)

        top5 = top5.copy()
        top5["arm"] = c["arm"]
        top5["rank"] = range(1, len(top5) + 1)
        rows.append(top5)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_lazy_top5_chart(top5_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if top5_df.empty:
        return fig
    arms = [a for a in ABLATION_ARMS if a in top5_df["arm"].unique()]
    rank_colors = ["#0d366b", "#2a78d6", "#6da7ec", "#b7d3f6", "#e1e0d9"]

    for rank in range(1, 6):
        sub = top5_df[top5_df["rank"] == rank].set_index("arm")
        y = [sub["R2"].get(a, float("nan")) for a in arms]
        models = [sub["Model"].get(a, "—") for a in arms]
        fig.add_trace(go.Bar(
            name=f"#{rank}",
            x=arms,
            y=y,
            marker_color=rank_colors[rank - 1],
            customdata=models,
            text=[f"{v:.3f}" if v == v else "" for v in y],
            textposition="outside",
            hovertemplate="<b>#%{fullData.name}</b> — %{customdata}<br>R2: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title="LazyPredict — top 5 models per configuration (by R²)",
        yaxis_title="R2",
        template="plotly_white", height=440,
        legend={"orientation": "h", "y": -0.22},
        margin={"t": 55, "b": 80},
    )
    return fig


# ── Seção 2 — Spatial & Temporal Error Inspector ──────────────────────────────

def build_inspector_map(
    stations_df: pd.DataFrame, sel_station: str | None,
    value_by_estacao: "pd.Series | None" = None,
    cmin: float | None = None, cmax: float | None = None,
) -> go.Figure:
    """Mapa de clusters + estações — independente de pipeline (mesma rede
    física INMET pras 3 pipelines). Generaliza o antigo build_map, removendo
    a dependência de R²/RMSE (que eram específicos do MLP).

    Mesmo padrão visual/cores dos painéis de `build_interp_map`: círculos com
    halo colorido por valor (Turbo, `cmin`/`cmax` compartilhados), não mais
    cor categórica por cluster nem `symbol: "star"` (símbolo depende de
    sprite carregado de forma assíncrona pelo maplibre — mesmo bug de
    confiabilidade já corrigido em `build_interp_map`)."""
    fig = go.Figure()

    locations: list[str] = []
    texts: list[str] = []
    for cid in CLUSTER_IDS_ALL:
        htext = f"Cluster {cid}"
        for m in CLUSTER_MEMBERS.get(cid, []):
            locations.append(f"{m:02d}")
            texts.append(htext)

    if locations:
        fig.add_trace(go.Choroplethmap(
            geojson=geojson_clusters,
            featureidkey="properties.cluster",
            locations=locations,
            z=list(range(len(locations))),
            colorscale="Viridis",
            marker_opacity=0.25,
            marker_line_width=1.2,
            marker_line_color="white",
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            showscale=False,
            name="Clusters",
        ))

    if not stations_df.empty:
        has_values = value_by_estacao is not None
        vals = (
            stations_df["estacao"].map(value_by_estacao).to_numpy(float)
            if has_values else None
        )
        sizes = [20 if sel_station == r.estacao else 15 for r in stations_df.itertuples()]
        halo_sizes = [24 if sel_station == r.estacao else 18 for r in stations_df.itertuples()]
        halo_colors = ["#c0392b" if sel_station == r.estacao else "black" for r in stations_df.itertuples()]

        fig.add_trace(go.Scattermap(
            lat=stations_df["latitude"], lon=stations_df["longitude"],
            mode="markers", marker={"size": halo_sizes, "color": halo_colors},
            hoverinfo="skip", showlegend=False,
        ))
        marker = {"size": sizes, "showscale": False}
        if has_values:
            marker.update({"color": vals, "colorscale": "Turbo", "cmin": cmin, "cmax": cmax})
        else:
            marker["color"] = [_cluster_color(c) for c in stations_df["cluster_id"]]
        fig.add_trace(go.Scattermap(
            lat=stations_df["latitude"],
            lon=stations_df["longitude"],
            mode="markers",
            marker=marker,
            customdata=stations_df[["estacao", "cluster_id"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Cluster: %{customdata[1]}<br>"
                "Lat: %{lat:.2f}°  Lon: %{lon:.2f}°<extra></extra>"
            ),
            name="Stations",
        ))

    # Cobertura ERA5 — retângulos dos dois domínios (novo Basin × antigo
    # 18UTC/Paraná-only), pra deixar visível o quanto a cobertura cresceu.
    old_lats, old_lons = _extent_ring(ERA5_18UTC_EXTENT)
    fig.add_trace(go.Scattermap(
        lat=old_lats, lon=old_lons, mode="lines",
        line={"color": "#b23a3a", "width": 2},
        hoverinfo="skip", name="ERA5 18UTC coverage (old, Paraná only)",
    ))
    basin_lats, basin_lons = _extent_ring(ERA5_BASIN_EXTENT)
    fig.add_trace(go.Scattermap(
        lat=basin_lats, lon=basin_lons, mode="lines",
        line={"color": "#1f8a4c", "width": 2},
        hoverinfo="skip", name="ERA5-Basin coverage (new)",
    ))

    fig.update_layout(
        # Centro/zoom ajustados pra caber as 243 estações do INMET expandido
        # (lat -33.7..-14.4, lon -57.1..-39.9 — vai de RS até Minas Gerais),
        # não só o antigo recorte Sul (RS/SC/PR, zoom 5 cortava a metade norte).
        map={"style": "carto-positron", "center": {"lat": -24.0, "lon": -48.5}, "zoom": 4},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=420,
        uirevision="map",
        showlegend=True,
        legend={"orientation": "h", "y": 0, "bgcolor": "rgba(255,255,255,0.7)"},
    )
    return fig


def build_prediction_density(combos: list[dict], sel_cluster) -> go.Figure:
    """Densidade sobreposta observado × previsto pra cada combinação
    selecionada, a partir do predictions.parquet (schema tidy, comum às 3
    pipelines). Generaliza o antigo build_distribution (que só mostrava
    INMET vs. MLP)."""
    fig = go.Figure()
    shown_observed = False
    for c in combos:
        df = load_ablation_predictions(c["dir"])
        if df.empty:
            continue
        if sel_cluster is not None:
            df = df[df["cluster_id"] == sel_cluster]
        if df.empty:
            continue
        color = ARM_COLORS.get(c["arm"], "#898781")
        if not shown_observed:
            fig.add_trace(go.Histogram(
                x=df["y_true"], name="Observed (INMET)",
                marker_color="#52514e", opacity=0.35, histnorm="probability density",
            ))
            shown_observed = True
        fig.add_trace(go.Histogram(
            x=df["y_pred"], name=f"{c['label']} (predicted)",
            marker_color=color, opacity=0.45, histnorm="probability density",
        ))
    fig.update_layout(
        barmode="overlay",
        title="Observed vs. predicted density — selected combinations",
        xaxis_title=YAXIS_WIND, yaxis_title="Density",
        template="plotly_white", height=420,
        legend={"orientation": "h", "y": -0.22},
        margin={"t": 50, "b": 80},
    )
    return fig


def build_residual_timeseries(estacao: str | None, mlp_preds_df: pd.DataFrame) -> go.Figure:
    """MLP-only: única pipeline com data real por observação
    (predictions_by_station.csv). Plota o resíduo (previsto - observado) ao
    longo do tempo pra estação selecionada."""
    fig = go.Figure()
    if estacao is None or mlp_preds_df.empty:
        fig.update_layout(
            title="Select a station on the map to see the MLP residual series",
            template="plotly_white", height=380,
            xaxis_title="Date", yaxis_title="Residual (predicted − observed, m/s)",
        )
        return fig

    df_st = mlp_preds_df[mlp_preds_df["estacao"] == estacao].sort_values("time")
    if df_st.empty:
        fig.update_layout(
            title=f"No MLP data for station {estacao}",
            template="plotly_white", height=380,
        )
        return fig
    residual = df_st["y_pred"] - df_st["y_true"]

    fig.add_trace(go.Scatter(
        x=df_st["time"], y=residual, mode="lines",
        name="Residual (MLP)", line={"color": PIPELINE_COLORS["mlp"], "width": 1.3},
    ))
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(
        title=f"MLP residual over time — station {estacao}",
        xaxis_title="Date", yaxis_title="Residual (m/s)",
        template="plotly_white", height=380,
        margin={"t": 50, "b": 60},
    )
    return fig


def build_residual_boxplot(combos: list[dict], sel_cluster) -> go.Figure:
    """LazyPredict/LSTM não têm timestamp por observação em nenhum arquivo
    sincronizado — mostra a distribuição do resíduo por combinação em vez de
    uma série temporal fabricada."""
    fig = go.Figure()
    for c in combos:
        df = load_ablation_predictions(c["dir"])
        if df.empty:
            continue
        if sel_cluster is not None:
            df = df[df["cluster_id"] == sel_cluster]
        if df.empty:
            continue
        residual = df["y_pred"] - df["y_true"]
        fig.add_trace(go.Box(
            y=residual, name=c["label"],
            marker_color=ARM_COLORS.get(c["arm"], "#898781"),
            line={"color": PIPELINE_COLORS.get(c["pipeline"], "#333")},
            boxmean=True,
        ))
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(
        title="Residual distribution (no per-observation date available)",
        yaxis_title="Residual (predicted − observed, m/s)",
        template="plotly_white", height=380,
        margin={"t": 50, "b": 60},
    )
    return fig


# ── Interpolação espacial IDW das predições (qualquer pipeline) ───────────────
# Reimplementa os 3 métodos de suavização do plot.py (IDW original / IDW+epsilon
# / Gaussian kernel), aplicados aos VALORES PREVISTOS em vez das grades .nc
# corrigidas (que não vivem neste repo). Só o MLP exporta predições com
# latitude/longitude por estação (predictions_by_station.csv) — LazyPredict e
# LSTM só reportam por cluster_id (predictions.parquet). Pra esses dois, o
# valor agregado do cluster é "espalhado" (broadcast) pra todas as estações
# daquele cluster via aggregate_cluster_values — resolução de cluster, não de
# estação; a UI deixa isso explícito na legenda.

INTERP_METRICS = {
    "Historical max": ("max", None),
    "P95": ("quantile", 0.95),
    "P99": ("quantile", 0.99),
    "Mean": ("mean", None),
}
INTERP_VALUES = {
    "Predicted": "y_pred",
    "Observed (INMET)": "y_true",
    "ERA5 raw": "era5_wind_mag_max",
    "Correction (Pred − ERA5)": "_correction",
}
INTERP_METHODS = {
    "IDW (original)": "original",
    "IDW + epsilon": "epsilon",
    "Gaussian kernel": "gaussian",
}


@st.cache_data
def aggregate_station_values(
    combo_dir: str, value_label: str, metric_label: str
) -> pd.DataFrame:
    """Colapsa predictions_by_station.csv (uma linha por observação) num valor
    por estação, segundo a métrica (máx/p95/p99/média) herdada do plot.py."""
    df = load_mlp_predictions_by_station(combo_dir)
    cols = ["estacao", "latitude", "longitude", "cluster_id", "value"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    col = INTERP_VALUES[value_label]
    df = df.copy()
    if col == "_correction":
        df["_correction"] = df["y_pred"] - df["era5_wind_mag_max"]
    agg, q = INTERP_METRICS[metric_label]
    grp = df.groupby(["estacao", "latitude", "longitude", "cluster_id"])[col]
    if agg == "max":
        s = grp.max()
    elif agg == "mean":
        s = grp.mean()
    else:
        s = grp.quantile(q)
    return s.reset_index().rename(columns={col: "value"})


@st.cache_data
def load_inmet_observed(metric_label: str, stations_df: pd.DataFrame) -> pd.DataFrame:
    """Observado (INMET) direto do netCDF bruto (INMET_RAW_PATH) — todas as
    estações da rede atual (243, sincronizadas manualmente do repo de
    pesquisa), independente de qualquer pipeline/arm já treinado. As 3
    pipelines ainda vão ser re-treinadas com esse INMET expandido; a
    "verdade" (ground truth) não deveria ficar presa ao station-count do
    último sync de resultados enquanto isso não acontece."""
    cols = ["estacao", "latitude", "longitude", "cluster_id", "value"]
    if not INMET_RAW_PATH.exists() or stations_df.empty:
        return pd.DataFrame(columns=cols)
    ds = xr.open_dataset(INMET_RAW_PATH)
    gust = ds[INMET_TARGET_VAR]
    agg, q = INTERP_METRICS[metric_label]
    if agg == "max":
        vals = gust.max(dim="time", skipna=True)
    elif agg == "mean":
        vals = gust.mean(dim="time", skipna=True)
    else:
        vals = gust.quantile(q, dim="time", skipna=True)
    df = pd.DataFrame({
        "estacao": ds["estacao"].values,
        "value": np.asarray(vals.values, dtype=float),
    })
    ds.close()
    merged = stations_df[["estacao", "latitude", "longitude", "cluster_id"]].merge(
        df, on="estacao", how="inner",
    )
    return merged[cols]


@st.cache_data
def aggregate_cluster_values(
    combo_dir: str, value_label: str, metric_label: str, stations_df: pd.DataFrame
) -> pd.DataFrame:
    """Equivalente de aggregate_station_values pra LazyPredict/LSTM: essas
    pipelines só sincronizam predictions.parquet com granularidade de
    cluster_id (sem estação/lat/lon por linha). Colapsa y_true/y_pred por
    cluster_id (mesma métrica máx/p95/p99/média) e espalha o valor único do
    cluster pra todas as estações-membro via stations_metadata.csv — ou seja,
    todas as estações de um mesmo cluster mostram o mesmo valor no mapa."""
    cols = ["estacao", "latitude", "longitude", "cluster_id", "value"]
    df = load_ablation_predictions(combo_dir)
    if df.empty or stations_df.empty:
        return pd.DataFrame(columns=cols)
    col = INTERP_VALUES[value_label]
    # combo_dir já fixa um (pipeline, arm) só — sem a coluna "arm" que
    # _ablation_select_split (Seção 1) espera. Prefere split="test"; senão
    # usa o único disponível, dentro deste único dataframe já homogêneo.
    splits = set(df["split"].unique())
    chosen_split = "test" if "test" in splits else sorted(splits)[0]
    df = df[df["split"] == chosen_split]
    agg, q = INTERP_METRICS[metric_label]
    grp = df.groupby("cluster_id")[col]
    if agg == "max":
        s = grp.max()
    elif agg == "mean":
        s = grp.mean()
    else:
        s = grp.quantile(q)
    cluster_vals = s.reset_index().rename(columns={col: "value"})
    cluster_vals["cluster_id"] = cluster_vals["cluster_id"].astype(str)
    st_df = stations_df[["estacao", "latitude", "longitude", "cluster_id"]].copy()
    st_df["cluster_id"] = st_df["cluster_id"].astype(str)
    merged = st_df.merge(cluster_vals, on="cluster_id", how="inner")
    return merged[cols]


def _idw_grid(st_lon, st_lat, st_val, method, n=70, margin=0.6,
              power=2.0, eps=1e-3, sigma=0.6):
    """Campo IDW numa grade regular. Distâncias em graus — aproximação plana,
    suficiente pro recorte pequeno do Sul do Brasil."""
    lon_min, lon_max = st_lon.min() - margin, st_lon.max() + margin
    lat_min, lat_max = st_lat.min() - margin, st_lat.max() + margin
    glon = np.linspace(lon_min, lon_max, n)
    glat = np.linspace(lat_min, lat_max, n)
    LON, LAT = np.meshgrid(glon, glat)
    d2 = (LON[..., None] - st_lon) ** 2 + (LAT[..., None] - st_lat) ** 2
    if method == "gaussian":
        w = np.exp(-d2 / (2.0 * sigma ** 2))
    elif method == "epsilon":
        w = 1.0 / (np.sqrt(d2) ** power + eps)
    else:  # original — piso minúsculo na distância evita divisão por zero
        w = 1.0 / (np.maximum(np.sqrt(d2), 1e-6) ** power)
    z = np.sum(w * st_val, axis=2) / np.sum(w, axis=2)
    return LON.ravel(), LAT.ravel(), z.ravel()


def build_interp_map(
    sdf: pd.DataFrame, method_label: str, is_diff: bool,
    cmin_override: float | None = None, cmax_override: float | None = None,
    height: int = 520, show_colorbar: bool = True,
    mask_geom=None, bounds_override: dict | None = None,
) -> go.Figure:
    """Mapa Plotly: campo interpolado (grade, pontos minúsculos e esmaecidos —
    é estimativa) + estações (círculos grandes com halo — é dado real). Mesma
    escala de cor nos dois, pra comparar valor previsto no ponto vs. campo.
    cmin_override/cmax_override permitem forçar a MESMA escala de cor em
    vários mapas lado a lado (comparação Observado × LazyPredict × MLP ×
    LSTM) — sem isso cada painel normalizaria pro próprio min/max.

    `mask_geom` (geometria shapely, opcional): recorta a grade IDW pra forma
    real da bacia/cluster (point-in-polygon), em vez do retângulo
    bounding-box que `_idw_grid` usa internamente pra gerar a grade.
    `bounds_override` (dict west/east/south/north, opcional): força a MESMA
    janela de mapa em vários painéis lado a lado — sem isso cada painel
    calcularia seu próprio enquadramento a partir das suas próprias estações."""
    fig = go.Figure()
    base_layout = {
        "map": {"style": "carto-positron",
                "center": {"lat": -28.5, "lon": -52.5}, "zoom": 5},
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0}, "height": height,
    }
    if sdf.empty or len(sdf) < 2:
        fig.update_layout(**base_layout)
        return fig
    lon = sdf["longitude"].to_numpy(float)
    lat = sdf["latitude"].to_numpy(float)
    val = sdf["value"].to_numpy(float)
    flon, flat, fz = _idw_grid(lon, lat, val, INTERP_METHODS[method_label])

    if mask_geom is not None:
        inside = shapely.contains(mask_geom, shapely.points(flon, flat))
        flon, flat, fz = flon[inside], flat[inside], fz[inside]

    allv = np.concatenate([fz[np.isfinite(fz)], val])
    if is_diff:
        m = float(np.nanmax(np.abs(allv))) or 1.0
        cmin, cmax, cmid, cscale = -m, m, 0, "RdBu_r"
    elif cmin_override is not None and cmax_override is not None:
        cmin, cmax, cmid, cscale = cmin_override, cmax_override, None, "Turbo"
    else:
        cmin, cmax, cmid, cscale = (
            float(np.nanmin(allv)), float(np.nanmax(allv)), None, "Turbo"
        )

    # A grade IDW entre estações não é dado observado — é uma estimativa que
    # pode estar bem errada longe de qualquer estação. O estilo abaixo
    # comunica isso visualmente: campo bem esmaecido (baixa opacidade, sem
    # borda) por baixo, estações grandes e com contorno sólido por cima —
    # só o marcador de estação é "dado real"; o resto é chute plausível.
    field_marker = {
        "size": 2, "color": fz, "colorscale": cscale, "cmin": cmin,
        "cmax": cmax, "opacity": 0.22, "showscale": show_colorbar,
    }
    if show_colorbar:
        field_marker["colorbar"] = {"title": "m/s"}
    # go.Scattermap não aceita marker.line (sem borda nativa, ao contrário de
    # go.Scatter) — o "contorno" é simulado com um halo: círculo preto um
    # pouco maior desenhado ANTES do círculo colorido, no mesmo ponto.
    # Círculo (não "star"): símbolos não-circulares em go.Scattermap dependem
    # de um sprite de ícone carregado de forma assíncrona pelo maplibre — numa
    # atualização rápida de bounds (troca de cluster focus) essa corrida pode
    # falhar silenciosamente e a camada de símbolo inteira não aparece (visto
    # ao vivo: estações sumiram do mapa depois de trocar o cluster focus,
    # mesmo a legenda continuando a mostrar "Stations"). Círculo é o marcador
    # nativo do Scattermap, sem essa dependência — sempre aparece.
    station_halo_marker = {"size": 20, "color": "black"}
    station_marker = {
        "size": 15, "color": val, "colorscale": cscale, "cmin": cmin,
        "cmax": cmax, "showscale": False,
    }
    if cmid is not None:
        field_marker["cmid"] = cmid
        station_marker["cmid"] = cmid

    fig.add_trace(go.Scattermap(
        lat=flat, lon=flon, mode="markers", marker=field_marker,
        hovertemplate="%{lat:.2f}, %{lon:.2f}<br>~%{marker.color:.2f} m/s (estimated)<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scattermap(
        lat=lat, lon=lon, mode="markers", marker=station_halo_marker,
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattermap(
        lat=lat, lon=lon, mode="markers", marker=station_marker,
        customdata=sdf[["estacao", "cluster_id", "value"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b> (actual data)<br>Cluster %{customdata[1]}<br>"
            "%{customdata[2]:.2f} m/s<extra></extra>"
        ),
        name="Stations",
    ))
    # bounds > center/zoom: ajusta a janela do mapa pra caber exatamente a
    # região de interesse (o retângulo bounding-box de estações antes deixava
    # o mapa zoomado pra fora quando o cluster focado era pequeno perto da
    # bacia inteira). bounds_override garante a MESMA janela em vários
    # painéis lado a lado; sem ele, cai pro bounding-box das próprias
    # estações + margem.
    del base_layout["map"]["center"]
    del base_layout["map"]["zoom"]
    if bounds_override is not None:
        base_layout["map"]["bounds"] = bounds_override
    else:
        pad = 0.3
        base_layout["map"]["bounds"] = {
            "west": float(lon.min()) - pad, "east": float(lon.max()) + pad,
            "south": float(lat.min()) - pad, "north": float(lat.max()) + pad,
        }
    fig.update_layout(
        **base_layout, legend={"orientation": "h", "y": 0}, uirevision="interp",
    )
    return fig


# ── Seção 3 — Model Diagnostics & Explainability ──────────────────────────────

def _obs_pred_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Bias/RMSE/MAE/Corr/N entre observado e previsto, ignorando pares com
    NaN em qualquer lado — mesma régua estatística de
    `interpolation comparisson/streamlit_app/validation.py::compute_validation_metrics`,
    usada tanto no card de métrica quanto no subtítulo do scatter (calculada
    uma única vez por combo, reaproveitada nos dois lugares)."""
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    valid = yt.notna() & yp.notna()
    yt, yp = yt[valid], yp[valid]
    n = len(yt)
    if n == 0:
        return {"bias": float("nan"), "rmse": float("nan"), "mae": float("nan"), "corr": float("nan"), "n": 0}
    diff = yp - yt
    corr = float(yt.corr(yp)) if n >= 2 and yt.std() > 0 and yp.std() > 0 else float("nan")
    return {
        "bias": float(diff.mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "mae": float(diff.abs().mean()),
        "corr": corr,
        "n": n,
    }


def build_ablation_scatter_with_ols(
    pred_df: pd.DataFrame, pipeline: str, arm: str, metrics: dict | None = None,
) -> go.Figure:
    """Scatter observado×previsto com linha 1:1 E reta de regressão OLS
    ajustada (np.polyfit) — usado pra lazy/mlp. `metrics` pode ser passado
    pré-calculado (ver Seção 3, cards de métrica) pra evitar recomputar
    bias/RMSE/MAE/corr duas vezes pro mesmo combo."""
    fig = go.Figure()
    if pred_df.empty:
        fig.update_layout(
            title="No prediction rows available for this combination",
            template="plotly_white", height=480,
        )
        return fig
    color = ARM_COLORS.get(arm, "#898781")
    outline = PIPELINE_COLORS.get(pipeline, "#333")
    if metrics is None:
        metrics = _obs_pred_metrics(pred_df["y_true"], pred_df["y_pred"])
    fig.add_trace(go.Scatter(
        x=pred_df["y_true"], y=pred_df["y_pred"],
        mode="markers",
        marker={"size": 5, "color": color, "opacity": 0.4, "line": {"width": 1, "color": outline}},
        hovertemplate="Observed: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        name=f"{pipeline}/{arm}",
    ))
    lo = float(min(pred_df["y_true"].min(), pred_df["y_pred"].min()))
    hi = float(max(pred_df["y_true"].max(), pred_df["y_pred"].max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line={"color": "#898781", "dash": "dash"}, name="y = x",
    ))
    coeffs = np.polyfit(pred_df["y_true"], pred_df["y_pred"], 1)
    x_line = np.array([lo, hi])
    r2 = float(pred_df["y_true"].corr(pred_df["y_pred"]) ** 2)
    fig.add_trace(go.Scatter(
        x=x_line, y=np.polyval(coeffs, x_line), mode="lines",
        line={"color": "black", "width": 2},
        name=f"OLS fit (R²={r2:.3f})",
    ))
    fig.update_layout(
        title=(
            f"Observed vs. Predicted — {PIPELINE_LABELS.get(pipeline, pipeline)} / {arm}<br>"
            f"<sup>bias={metrics['bias']:+.2f} · RMSE={metrics['rmse']:.2f} · "
            f"MAE={metrics['mae']:.2f} · corr={metrics['corr']:.2f} · n={metrics['n']}</sup>"
        ),
        xaxis_title="Observed (m/s)", yaxis_title="Predicted (m/s)",
        template="plotly_white", height=480,
        legend={"orientation": "h", "y": -0.18},
        margin={"t": 65, "l": 10, "r": 10, "b": 60},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def build_lstm_loss_curve(histories: dict, cluster_id, season: str) -> go.Figure:
    fig = go.Figure()
    leaf = histories.get(str(cluster_id), {}).get(season)
    if not leaf or not leaf.get("loss"):
        fig.update_layout(
            title=f"No training history for cluster {cluster_id} / {season}",
            template="plotly_white", height=380,
        )
        return fig
    loss = leaf["loss"]
    val_loss = leaf.get("val_loss", [])
    if all(v != v for v in loss):  # todos NaN — season sem treino válido
        fig.update_layout(
            title=f"Training history for cluster {cluster_id} / {season} is empty (NaN) — known pipeline gap",
            template="plotly_white", height=380,
        )
        return fig
    epochs = list(range(1, len(loss) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=loss, mode="lines", name="train loss",
                              line={"color": "#2a78d6"}))
    if val_loss:
        fig.add_trace(go.Scatter(x=list(range(1, len(val_loss) + 1)), y=val_loss,
                                  mode="lines", name="val loss", line={"color": "#c98a1f"}))
    fig.update_layout(
        title=f"LSTM training convergence — cluster {cluster_id} / {season}",
        xaxis_title="Epoch", yaxis_title="Loss",
        template="plotly_white", height=380,
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 50, "b": 60},
    )
    return fig


def build_importance(importance_df: pd.DataFrame, cluster_id) -> go.Figure:
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
        title=f"Permutation Importance (MLP) — Cluster {cluster_id}",
        xaxis_title="Drop in R²",
        template="plotly_white", height=480,
        margin={"t": 50, "l": 160},
    )
    return fig


# ── Layout principal ──────────────────────────────────────────────────────────

geojson_clusters = load_geojson()
stations_geo_df = load_stations_geo()
CLUSTER_IDS_ALL = (
    sorted(stations_geo_df["cluster_id"].unique().tolist())
    if not stations_geo_df.empty else []
)
CLUSTER_COLORS = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(CLUSTER_IDS_ALL)}
CLUSTER_MEMBERS = {cid: _cluster_members(cid) for cid in CLUSTER_IDS_ALL}

ablation_combos = discover_ablation_combos()

st.title("🌬️ Extreme Wind Gust Bias Correction")

if "global_pipelines" not in st.session_state:
    st.session_state["global_pipelines"] = ["mlp"]
if "global_arms" not in st.session_state:
    st.session_state["global_arms"] = ["original"]
if "global_station" not in st.session_state:
    st.session_state["global_station"] = None

# O clique no mapa (Seção 2) não pode escrever direto em
# st.session_state["global_station"] no MESMO run em que o widget
# (sidebar, key="global_station") já foi instanciado — Streamlit proíbe
# mutar o state de um widget já criado nesse run. Em vez disso, o clique
# grava um valor pendente aqui, processado ANTES do widget existir.
if "_pending_station" in st.session_state:
    st.session_state["global_station"] = st.session_state.pop("_pending_station")

with st.sidebar:
    st.title("🌬️ IRC Vendaval")
    st.caption("Extreme wind gust bias correction")
    st.divider()

    st.caption("**Global comparison scope** — drives Sections 2 & 3 below.")
    st.multiselect(
        "Pipeline(s)", ABLATION_PIPELINES,
        format_func=lambda p: PIPELINE_LABELS.get(p, p),
        key="global_pipelines",
    )
    st.multiselect("Configuration(s)", ABLATION_ARMS, key="global_arms")

    selected_pipelines = st.session_state["global_pipelines"]
    selected_arms = st.session_state["global_arms"]
    selected_combos = [
        c for c in ablation_combos
        if c["pipeline"] in selected_pipelines and c["arm"] in selected_arms
    ]
    if len(selected_combos) > 4:
        st.warning("Selecting many combinations will crowd the overlay charts below.")
    elif not selected_combos:
        st.info("Pick at least one pipeline and one configuration.", icon="ℹ️")

    combo_cluster_ids: set = set()
    for c in selected_combos:
        df = load_ablation_results(c["dir"])
        if not df.empty:
            combo_cluster_ids |= set(df["cluster_id"].unique().tolist())
    cluster_options = sorted(combo_cluster_ids, key=str) or CLUSTER_IDS_ALL

    global_cluster = st.selectbox(
        "Cluster", cluster_options, format_func=lambda c: f"Cluster {c}",
        key="global_cluster",
    ) if cluster_options else None
    global_season = st.selectbox(
        "Climate quarter", ABLATION_SEASONS_ORDER, key="global_season",
    )

    stations_in_cluster = (
        stations_geo_df[stations_geo_df["cluster_id"] == global_cluster]["estacao"].tolist()
        if global_cluster is not None else []
    )
    station_options = ["(none)"] + stations_in_cluster
    if st.session_state["global_station"] not in station_options:
        st.session_state["global_station"] = "(none)"
    st.selectbox("Station", station_options, key="global_station")
    global_station = (
        None if st.session_state["global_station"] == "(none)"
        else st.session_state["global_station"]
    )

    st.divider()
    st.caption("**Temporal Split**")
    st.caption(f"Train: `{TRAIN_PERIOD[0]}` → `{TRAIN_PERIOD[1]}`")
    st.caption(f"Validation: `{VAL_PERIOD[0]}` → `{VAL_PERIOD[1]}`")

tab_global, tab_inspector, tab_diag = st.tabs([
    "📊 Global Comparison Panel",
    "🗺️ Spatial & Temporal Error Inspector",
    "🔬 Model Diagnostics & Explainability",
])

# ── Seção 1: Global Comparison Panel ──────────────────────────────────────────

with tab_global:
    if not ablation_combos:
        st.info(
            "No results synced yet. Run "
            "`scripts/sync_ablation_to_dashboard.py` from the research repo "
            "after a pipeline finishes.",
            icon="ℹ️",
        )
    else:
        all_results = load_all_ablation(ablation_combos)
        panel_seasons = [
            s for s in ABLATION_SEASONS_ORDER
            if s in all_results["season"].unique()
        ] if not all_results.empty else []
        panel_clusters = (
            sorted(all_results["cluster_id"].unique(), key=str)
            if not all_results.empty else []
        )

        col_metric, col_season, col_cluster = st.columns([1, 1, 1])
        with col_metric:
            metric = st.selectbox("Metric", ABLATION_METRICS, key="ablation_metric")
        with col_season:
            panel_season = st.selectbox(
                "Climate quarter (season)", panel_seasons or ["ALL"], key="ablation_season",
            )
        with col_cluster:
            panel_cluster_choice = st.selectbox(
                "Cluster",
                ["All clusters (weighted avg)"] + [f"Cluster {c}" for c in panel_clusters],
                key="ablation_cluster",
                help=(
                    "The default blends all clusters via a weighted average, "
                    "which can hide strong individual clusters — pick one to "
                    "see its own metric."
                ),
            )

        by_season = (
            _resolve_all_season(all_results) if panel_season == "ALL"
            else all_results[all_results["season"] == panel_season]
        )
        if by_season.empty:
            st.info(
                f"No combination has data for season '{panel_season}' yet — "
                "pipelines currently differ in season coverage "
                "(e.g. MLP only reports 'ALL' today).",
                icon="ℹ️",
            )

        by_cluster = by_season
        if panel_cluster_choice != "All clusters (weighted avg)":
            chosen_cluster = panel_cluster_choice.removeprefix("Cluster ")
            by_cluster = by_season[by_season["cluster_id"].astype(str) == chosen_cluster]

        selected_rows = _ablation_select_split(by_cluster)
        summary_df = _ablation_aggregate(selected_rows, ["pipeline", "arm"])

        # ── Cards de métrica — melhor configuração por pipeline vs. "original" ──
        direction = ABLATION_METRIC_DIRECTIONS.get(metric, "higher")
        delta_df = build_ablation_delta_table(summary_df, metric)
        if not delta_df.empty:
            card_cols = st.columns(len(ABLATION_PIPELINES))
            for card_col, pipeline in zip(card_cols, ABLATION_PIPELINES):
                pipeline_label = PIPELINE_LABELS.get(pipeline, pipeline)
                sub = delta_df[delta_df["Pipeline"] == pipeline_label]
                if sub.empty:
                    card_col.metric(pipeline_label, "—")
                    continue
                if direction == "higher":
                    best_row = sub.loc[sub[metric].idxmax()]
                    delta_val, delta_color, delta_label = best_row[f"Δ{metric}"], "normal", f"Δ{metric}"
                elif direction == "lower":
                    best_row = sub.loc[sub[metric].idxmin()]
                    delta_val, delta_color, delta_label = best_row[f"Δ{metric}"], "inverse", f"Δ{metric}"
                else:  # "zero" (Bias/Bias_P90) — mais perto de zero é melhor, sinal
                    # bruto do delta não distingue melhora de piora; compara |valor|.
                    best_row = sub.loc[sub[metric].abs().idxmin()]
                    base_rows = sub[sub["Arm"] == "original"]
                    delta_val = (
                        abs(best_row[metric]) - abs(base_rows.iloc[0][metric])
                        if not base_rows.empty else float("nan")
                    )
                    delta_color, delta_label = "inverse", f"Δ|{metric}|"
                card_col.metric(
                    f"{pipeline_label} — best: {best_row['Arm']}",
                    f"{best_row[metric]:.3f}",
                    delta=f"{delta_val:+.3f} {delta_label} vs. original" if delta_val == delta_val else None,
                    delta_color=delta_color,
                )

        st.plotly_chart(
            build_ablation_bars(summary_df, metric),
            use_container_width=True, key="ablation_bars_chart",
        )
        st.plotly_chart(
            build_ablation_heatmap(summary_df, metric),
            use_container_width=True, key="ablation_heatmap_chart",
        )

        # delta_df já calculado acima pros cards de métrica — reaproveitado
        # aqui, não recomputado.

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
            styled = styled.map(lambda v: _color_ablation_delta(v, metric), subset=[delta_col])
        st.dataframe(
            styled, use_container_width=True, hide_index=True, key="ablation_delta_table",
        )

        with st.expander("LazyPredict — top 5 models per configuration"):
            lazy_top5_df = build_lazy_top5_per_arm(ablation_combos, panel_season, panel_cluster_choice)
            st.plotly_chart(
                build_lazy_top5_chart(lazy_top5_df),
                use_container_width=True, key="lazy_top5_chart",
            )
            if not lazy_top5_df.empty:
                st.dataframe(
                    lazy_top5_df[["arm", "rank", "Model", "R2"]].rename(
                        columns={"arm": "Configuration", "rank": "Rank"}
                    ),
                    use_container_width=True, hide_index=True, key="lazy_top5_table",
                )

# ── Seção 2: Spatial & Temporal Error Inspector ───────────────────────────────

with tab_inspector:
    _obs_sdf = None
    _shared_cmin = _shared_cmax = None
    _arms_with_data = [a for a in ABLATION_ARMS if any(c["arm"] == a for c in ablation_combos)]
    if not _arms_with_data:
        st.info("No synced results yet.", icon="ℹ️")
    else:
        mc0, mc1, mc2, mc3 = st.columns(4)
        with mc0:
            multi_arm = st.selectbox("Configuration", _arms_with_data, key="multi_map_arm")
        with mc1:
            multi_metric = st.selectbox("Metric", list(INTERP_METRICS), key="multi_map_metric")
        with mc2:
            multi_method = st.selectbox("Smoothing", list(INTERP_METHODS), key="multi_map_method")

        def _combo_for(pipeline: str, arm: str) -> dict | None:
            return next(
                (c for c in ablation_combos if c["pipeline"] == pipeline and c["arm"] == arm),
                None,
            )

        _mlp_combo = _combo_for("mlp", multi_arm)
        _lazy_combo = _combo_for("lazy", multi_arm)
        _lstm_combo = _combo_for("lstm", multi_arm)

        # Observado vem direto do INMET_Stratified.nc bruto (243 estações atuais),
        # não de um arm/pipeline já sincronizado — assim não fica preso ao
        # station-count antigo enquanto lazy/mlp/lstm ainda não foram re-treinados
        # com a rede expandida.
        _obs_sdf = load_inmet_observed(multi_metric, stations_geo_df)

        def _sdf_for_combo(combo: dict | None, value_label: str) -> pd.DataFrame | None:
            """Usa predictions_by_station.csv (granularidade real de estação)
            quando a pipeline já sincronizou esse arquivo pra esse arm; senão
            cai pro agregado por cluster_id espalhado (aggregate_cluster_values)
            — fallback só pra arms ainda não re-treinados com a granularidade
            nova. Antes, LazyPredict/LSTM sempre caíam no broadcast por
            cluster; agora que as duas também salvam predictions_by_station.csv,
            usam a mesma granularidade real que o MLP já tinha."""
            if combo is None:
                return None
            if (Path(combo["dir"]) / "predictions_by_station.csv").exists():
                return aggregate_station_values(combo["dir"], value_label, multi_metric)
            return aggregate_cluster_values(combo["dir"], value_label, multi_metric, stations_geo_df)

        _lazy_sdf = _sdf_for_combo(_lazy_combo, "Predicted")
        _mlp_sdf = _sdf_for_combo(_mlp_combo, "Predicted")
        _lstm_sdf = _sdf_for_combo(_lstm_combo, "Predicted")
        # ERA5 bruto (sem correção) — só o MLP sincroniza era5_wind_mag_max por
        # observação (predictions_by_station.csv); LazyPredict/LSTM só têm
        # y_true/y_pred no predictions.parquet. O valor em si não depende do
        # pipeline (é o insumo de entrada, não uma saída de modelo), então
        # reusar a fonte do MLP aqui não é diferente de qualquer outra feature
        # de entrada — só precisa que o MLP esteja sincronizado pra config atual.
        _era5_sdf = (
            aggregate_station_values(_mlp_combo["dir"], "ERA5 raw", multi_metric)
            if _mlp_combo else None
        )

        multi_panels = [
            ("Observed (INMET)", _obs_sdf),
            ("ERA5", _era5_sdf),
            ("LazyPredict", _lazy_sdf),
            ("MLP", _mlp_sdf),
            ("LSTM (TF dual-head)", _lstm_sdf),
        ]

        _all_cluster_ids = sorted({
            str(cid)
            for _, sdf in multi_panels if sdf is not None and not sdf.empty
            for cid in sdf["cluster_id"].unique()
        }, key=str)
        with mc3:
            multi_cluster = st.selectbox(
                "Cluster focus", ["All clusters"] + [f"Cluster {c}" for c in _all_cluster_ids],
                key="multi_map_cluster_focus",
            )
        if multi_cluster != "All clusters":
            _chosen = multi_cluster.removeprefix("Cluster ")
            multi_panels = [
                (
                    label,
                    sdf[sdf["cluster_id"].astype(str) == _chosen] if sdf is not None and not sdf.empty else sdf,
                )
                for label, sdf in multi_panels
            ]

        _all_vals = pd.concat(
            [sdf["value"] for _, sdf in multi_panels if sdf is not None and not sdf.empty],
            ignore_index=True,
        )
        _shared_cmin = float(_all_vals.min()) if not _all_vals.empty else None
        _shared_cmax = float(_all_vals.max()) if not _all_vals.empty else None

        # Recorta o campo interpolado na forma real da bacia/cluster (em vez
        # de um retângulo) e enquadra o mapa exatamente nessa região — os 4
        # painéis compartilham a MESMA geometria/janela, senão cada um
        # ajustaria seu próprio zoom a partir das suas próprias estações.
        _mask_geom = _geom_for_cluster_choice(
            _chosen if multi_cluster != "All clusters" else None
        )
        _mgw, _mgs, _mge, _mgn = _mask_geom.bounds

        # Garante que NENHUMA estação fique cortada fora da janela do mapa: o
        # enquadramento é a UNIÃO do polígono da bacia/cluster com a extensão
        # real das estações de todos os painéis (uma estação raramente cai
        # ligeiramente fora do próprio polígono do cluster, perto da borda).
        _sta_lons = pd.concat(
            [sdf["longitude"] for _, sdf in multi_panels if sdf is not None and not sdf.empty],
            ignore_index=True,
        )
        _sta_lats = pd.concat(
            [sdf["latitude"] for _, sdf in multi_panels if sdf is not None and not sdf.empty],
            ignore_index=True,
        )
        if not _sta_lons.empty:
            _mgw = min(_mgw, float(_sta_lons.min()))
            _mge = max(_mge, float(_sta_lons.max()))
            _mgs = min(_mgs, float(_sta_lats.min()))
            _mgn = max(_mgn, float(_sta_lats.max()))

        _bpad = 0.3
        _shared_bounds = {
            "west": _mgw - _bpad, "east": _mge + _bpad,
            "south": _mgs - _bpad, "north": _mgn + _bpad,
        }

        # Só 1 colorbar pra linha inteira (no último painel com dado válido) — os
        # 4 painéis já compartilham a mesma escala (_shared_cmin/_shared_cmax),
        # repetir a barra 4x só desperdiçava espaço horizontal sem info nova.
        _last_valid_idx = max(
            (i for i, (_, sdf) in enumerate(multi_panels) if sdf is not None and not sdf.empty and len(sdf) >= 2),
            default=-1,
        )
        map_cols = st.columns(len(multi_panels))
        for i, (map_col, (label, sdf)) in enumerate(zip(map_cols, multi_panels)):
            with map_col:
                st.markdown(f"**{label}**")
                if sdf is None:
                    st.info(f"Not synced for '{multi_arm}'.", icon="ℹ️")
                elif sdf.empty or len(sdf) < 2:
                    st.info("Not enough stations.", icon="ℹ️")
                else:
                    st.plotly_chart(
                        build_interp_map(
                            sdf, multi_method, False,
                            cmin_override=_shared_cmin, cmax_override=_shared_cmax, height=380,
                            mask_geom=_mask_geom, bounds_override=_shared_bounds,
                            show_colorbar=(i == _last_valid_idx),
                        ),
                        use_container_width=True, key=f"multi_map_{label}",
                    )

    st.divider()

    if not selected_combos:
        st.info("Pick at least one pipeline and configuration in the sidebar to inspect.", icon="ℹ️")
    else:
        st.caption(
            "Comparing: " + ", ".join(c["label"] for c in selected_combos)
            + f" — Cluster {global_cluster} — {global_season}"
        )

        _value_by_estacao = (
            _obs_sdf.set_index("estacao")["value"]
            if _obs_sdf is not None and not _obs_sdf.empty else None
        )
        col_map, col_density = st.columns(2)
        with col_map:
            event = st.plotly_chart(
                build_inspector_map(
                    stations_geo_df, global_station,
                    value_by_estacao=_value_by_estacao,
                    cmin=_shared_cmin, cmax=_shared_cmax,
                ),
                width="stretch", on_select="rerun", key="map_chart",
                selection_mode="points",
            )
            if event and hasattr(event, "selection") and event.selection.points:
                pt = event.selection.points[0]
                cd = pt.get("customdata")
                if cd and len(cd) >= 2:
                    clicked_station = str(cd[0])
                    if clicked_station in stations_geo_df["estacao"].tolist():
                        st.session_state["_pending_station"] = clicked_station
                        st.rerun()

        with col_density:
            st.plotly_chart(
                build_prediction_density(selected_combos, global_cluster),
                width="stretch", key="density_chart",
            )

        st.subheader("Residual behaviour")
        col_ts, col_box = st.columns(2)
        mlp_selected = [c for c in selected_combos if c["pipeline"] == "mlp"]
        other_selected = [c for c in selected_combos if c["pipeline"] != "mlp"]

        with col_ts:
            if mlp_selected:
                mlp_preds_df = load_mlp_predictions_by_station(mlp_selected[0]["dir"])
                st.plotly_chart(
                    build_residual_timeseries(global_station, mlp_preds_df),
                    width="stretch", key="residual_timeseries_chart",
                )
            else:
                st.info("Select MLP as one of the pipelines to see a real residual time series.", icon="ℹ️")

        with col_box:
            if other_selected:
                st.plotly_chart(
                    build_residual_boxplot(other_selected, global_cluster),
                    width="stretch", key="residual_boxplot_chart",
                )
            else:
                st.info("Select LazyPredict and/or LSTM to see their residual distribution.", icon="ℹ️")

# ── Seção 3: Model Diagnostics & Explainability ───────────────────────────────

with tab_diag:
    if not selected_combos:
        st.info("Pick at least one pipeline and configuration in the sidebar to see diagnostics.", icon="ℹ️")
    else:
        for combo in selected_combos:
            st.subheader(combo["label"])
            preds = load_ablation_predictions(combo["dir"])
            if global_cluster is not None and not preds.empty and "cluster_id" in preds.columns:
                preds_c = preds[preds["cluster_id"] == global_cluster]
            else:
                preds_c = preds

            # Cards + scatter/OLS pra TODAS as pipelines (inclusive LSTM) — o
            # R² sozinho (Seção 1) não explica POR QUE ele é negativo pra
            # alguns arms; ver os pontos reais + a reta de regressão contra a
            # 1:1 é o que realmente mostra a causa (viés sistemático,
            # variância baixa demais, outliers, etc.).
            combo_metrics = _obs_pred_metrics(preds_c["y_true"], preds_c["y_pred"]) if not preds_c.empty else None
            if combo_metrics is not None and combo_metrics["n"] > 0:
                mcol0, mcol1, mcol2, mcol3 = st.columns(4)
                mcol0.metric("Bias", f"{combo_metrics['bias']:+.2f} m/s")
                mcol1.metric("RMSE", f"{combo_metrics['rmse']:.2f} m/s")
                mcol2.metric("MAE", f"{combo_metrics['mae']:.2f} m/s")
                mcol3.metric("Corr", f"{combo_metrics['corr']:.2f}")
            st.plotly_chart(
                build_ablation_scatter_with_ols(preds_c, combo["pipeline"], combo["arm"], metrics=combo_metrics),
                use_container_width=True, key=f"diag_scatter_{combo['pipeline']}_{combo['arm']}",
            )

            if combo["pipeline"] == "lstm":
                histories = load_lstm_histories(combo["dir"])
                season_for_lstm = global_season if global_season != "ALL" else "DJF"
                st.plotly_chart(
                    build_lstm_loss_curve(histories, global_cluster, season_for_lstm),
                    use_container_width=True, key=f"diag_loss_{combo['arm']}",
                )

            if combo["pipeline"] == "mlp":
                importance_df = load_mlp_importance(combo["dir"])
                if not importance_df.empty and global_cluster is not None:
                    st.plotly_chart(
                        build_importance(importance_df, global_cluster),
                        use_container_width=True, key=f"diag_importance_{combo['arm']}",
                    )

            st.divider()
