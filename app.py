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
    layout="wide",
    initial_sidebar_state="expanded",
)

# Esconde o controle de atribuição (texto "© CARTO, © OpenStreetMap
# contributors" + botão "ⓘ") que a MapLibre/Mapbox GL desenha sobre os
# mapas — a pedido explícito, ciente de que isso normalmente vai contra os
# termos de uso desses provedores de tile gratuitos.
st.markdown(
    """
    <style>
    .maplibregl-ctrl-attrib, .mapboxgl-ctrl-attrib {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constantes ────────────────────────────────────────────────────────────────

SHP_PATH = Path("dataset/shp/shp_vento.shp")
# INMET_Stratified.nc sincronizado direto do repo de pesquisa (não passa por
# nenhuma pipeline) — fonte de verdade do "Observed (INMET)" no mapa, pra não
# ficar preso ao station-count/período de qualquer arm já treinado. Precisa
# ser ressincronizado manualmente sempre que o INMET for expandido de novo.
INMET_RAW_PATH = Path("dataset/raw/INMET_Stratified.nc")
INMET_TARGET_VAR = "daily_wind_gust_max"

# Grid corrigido (sincronizado via scripts/sync_corrected_grid_to_dashboard.py
# no repo de pesquisa) — versão "v1" do best-model-per-cluster/trimestre, ver
# src/dataset/creation/grid_generator.py. Não passa pela matriz de ablation
# (ARTIFACTS_DIR acima): é um produto espacial único, não um experimento.
CORRECTED_GRID_DIR = Path("artifacts/corrected_grid/v1")

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
# Hemisfério sul: DJF=Verão, MAM=Outono, JJA=Inverno, SON=Primavera — mesmo
# mapeamento de src/pipelines/common.py (SEASONS) no repo de pesquisa.
_MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
# Mesma paleta de scripts/_ablation_common.py no repo de pesquisa.
ARM_COLORS = {
    "original": "#2a78d6", "synthetic": "#1baf7a",
    "newfeatures": "#c98a1f", "all": "#4a3aa7",
    "basin": "#3d8b3d", "all_basin": "#8a5a2e",
}
PIPELINE_LABELS = {"lazy": "ML Models", "mlp": "MLP", "lstm": "LSTM (TF dual-head)"}
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

# Bounding boxes dos dois domínios ERA5 usados pelo projeto (coordenadas fixas,
# lidas direto dos .nc de origem no repo de pesquisa — não copiamos o netCDF
# de 86×77 pontos pra cá só pra desenhar um retângulo). "Basin" é o domínio
# novo (era5_basin_loader.py); "18UTC" é o antigo, restrito ao Paraná.
ERA5_BASIN_EXTENT = {"lat": (-35.0, -13.75), "lon": (-58.0, -39.0)}
ERA5_18UTC_EXTENT = {"lat": (-27.0, -22.0), "lon": (-55.0, -48.0)}


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


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray, p: int = 90) -> dict:
    """Réplica de compute_metrics() em src/pipelines/common.py (repo de
    pesquisa) — mesma fórmula (R2/RMSE via numpy puro, Bias_P90/RMSE_P90
    restritos à cauda >= percentil p de y_true), pra derivar métricas por
    trimestre localmente aqui no dashboard sem depender de sklearn."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    bias = float((y_pred - y_true).mean())
    mask = y_true >= np.percentile(y_true, p)
    bias_p = float((y_pred[mask] - y_true[mask]).mean()) if mask.sum() > 0 else float("nan")
    rmse_p = (
        float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
        if mask.sum() > 0 else float("nan")
    )
    return {
        "R2": r2, "RMSE": rmse, "Bias": bias,
        "Bias_P90": bias_p, "RMSE_P90": rmse_p,
    }


def derive_quarterly_results(combos: list[dict], all_results: pd.DataFrame) -> pd.DataFrame:
    """Pra combos cujo results.csv só reporta season='ALL' (hoje: MLP, que
    avalia o teste inteiro de uma vez, sem quebrar por trimestre), deriva
    linhas DJF/MAM/JJA/SON localmente a partir de predictions_by_station.csv
    (tem 'time' por observação) — mesma metodologia de compute_metrics()
    (ver _group_metrics), agrupando por (cluster_id, season, split), igual
    ao que lazy/lstm já fazem nativamente. Sem isso, essas pipelines somem
    do 'Per-quarter snapshot' mesmo tendo dado bruto suficiente pra calcular."""
    rows = []
    for c in combos:
        if not all_results.empty:
            seasons_here = all_results[
                (all_results["pipeline"] == c["pipeline"]) & (all_results["arm"] == c["arm"])
            ]["season"].unique()
            if any(s != "ALL" for s in seasons_here):
                continue  # já tem granularidade nativa — não duplica.
        preds = load_mlp_predictions_by_station(c["dir"])
        if preds.empty or "time" not in preds.columns:
            continue
        preds = preds.copy()
        preds["season"] = preds["time"].dt.month.map(_MONTH_TO_SEASON)
        for (cid, season, split), g in preds.groupby(["cluster_id", "season", "split"]):
            if len(g) < 2:
                continue
            metrics = _group_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
            rows.append({
                "pipeline": c["pipeline"], "arm": c["arm"], "experiment": c["arm"],
                "cluster_id": cid, "season": season, "split": split,
                "n_samples": len(g), **metrics,
            })
    return pd.DataFrame(rows)


# ── Seção 1 — Global Comparison Panel ─────────────────────────────────────────

def build_ablation_bars_by_quarter(all_results: pd.DataFrame, metric: str) -> go.Figure:
    """Mesma leitura do gráfico 'metric by pipeline × configuration' acima,
    mas quebrando cada pipeline nos 4 trimestres (eixo X vira pipeline >
    trimestre) — cor continua sendo a configuração. Complementa a visão
    agregada 'ALL' com o detalhe sazonal, sem precisar ir até o 'Per-quarter
    snapshot' (que já mostra isso, mas 1 subplot por pipeline separado)."""
    fig = go.Figure()
    quarters = [s for s in ("DJF", "MAM", "JJA", "SON") if s in all_results["season"].unique()]
    if all_results.empty or not quarters:
        return fig
    rows = all_results[all_results["season"].isin(quarters)]
    summary = _ablation_aggregate(_ablation_select_split(rows), ["pipeline", "arm", "season"])
    if summary.empty:
        return fig
    pipelines = [p for p in ABLATION_PIPELINES if p in summary["pipeline"].unique()]
    arms = [a for a in ABLATION_ARMS if a in summary["arm"].unique()]
    if not pipelines:
        return fig

    x_pipeline = [PIPELINE_LABELS.get(p, p) for p in pipelines for _ in quarters]
    x_quarter = [q for _ in pipelines for q in quarters]

    for arm in arms:
        sub = summary[summary["arm"] == arm].set_index(["pipeline", "season"])
        y = [sub[metric].get((p, q), float("nan")) for p in pipelines for q in quarters]
        fig.add_trace(go.Bar(
            name=arm,
            x=[x_pipeline, x_quarter],
            y=y,
            marker_color=ARM_COLORS.get(arm, "#898781"),
            hovertemplate=(
                f"<b>{arm}</b><br>%{{x}}<br>{metric}: " + "%{y:.4f}<extra></extra>"
            ),
        ))
    fig.update_layout(
        barmode="group",
        title=f"{metric} by pipeline × quarter × configuration",
        yaxis_title=metric,
        template="plotly_white", height=460,
        legend={"orientation": "h", "y": -0.25},
        margin={"t": 55, "b": 90},
    )
    return fig


def _best_combo_per_cluster(
    all_results: pd.DataFrame, metric: str, season: str | None = None
) -> dict[int, tuple[str, str]]:
    """Pra cada cluster, a combinação (pipeline, arm) vencedora pela métrica
    dada. season=None usa a visão agregada 'ALL' (ano inteiro); senão usa
    só as linhas (nativas ou derivadas) daquele trimestre específico."""
    if all_results.empty:
        return {}
    if season is None:
        selected = _ablation_select_split(_resolve_all_season(all_results))
    else:
        selected = _ablation_select_split(all_results[all_results["season"] == season])
    if selected.empty:
        return {}
    direction = ABLATION_METRIC_DIRECTIONS.get(metric, "higher")
    winners: dict[int, tuple[str, str]] = {}
    for cid, g in selected.groupby("cluster_id"):
        g = g.dropna(subset=[metric])
        if g.empty:
            continue
        if direction == "higher":
            row = g.loc[g[metric].idxmax()]
        elif direction == "lower":
            row = g.loc[g[metric].idxmin()]
        else:
            row = g.loc[g[metric].abs().idxmin()]
        winners[int(cid)] = (row["pipeline"], row["arm"])
    return winners


def _sdf_from_cluster_winners(
    ablation_combos: list[dict],
    winners: dict[int, tuple[str, str]],
    value_label: str,
    metric_label: str,
    season: str | None = None,
) -> pd.DataFrame:
    """Monta um DataFrame estacao/lat/lon/cluster_id/value 'costurado': o
    valor de cada estação vem do combo (pipeline, arm) que venceu o CLUSTER
    daquela estação (winners) — permite visualizar espacialmente 'o melhor
    resultado disponível, sempre', em vez de fixar 1 único pipeline/arm pro
    mapa inteiro. season=None agrega o ano inteiro; senão restringe àquele
    trimestre (mesmo recorte usado pra escolher o vencedor)."""
    cols = ["estacao", "latitude", "longitude", "cluster_id", "value"]
    frames = []
    for cid, (pipeline, arm) in winners.items():
        combo = next(
            (c for c in ablation_combos if c["pipeline"] == pipeline and c["arm"] == arm),
            None,
        )
        if combo is None:
            continue
        sdf = (
            aggregate_station_values(combo["dir"], value_label, metric_label)
            if season is None
            else aggregate_station_values_by_season(combo["dir"], value_label, metric_label, season)
        )
        if sdf.empty:
            continue
        sdf = sdf[sdf["cluster_id"].astype(str) == str(cid)]
        if not sdf.empty:
            frames.append(sdf)
    return pd.concat(frames, ignore_index=True)[cols] if frames else pd.DataFrame(columns=cols)


def build_best_per_cluster_bars(all_results: pd.DataFrame, metric: str) -> go.Figure:
    """4 barras por cluster (DJF/MAM/JJA/SON) = a MELHOR combinação
    (pipeline × configuração) daquele trimestre especificamente, entre
    TODAS as testadas (LazyPredict/MLP/LSTM × original/synthetic/
    newfeatures/all/basin/all_basin). Cor = pipeline vencedor. Eixo X é
    multi-categoria (cluster agrupando os 4 trimestres) — mais denso que a
    versão anterior (1 barra 'ALL' por cluster), mas mostra se o mesmo
    cluster tem um vencedor consistente ao longo do ano ou se a melhor
    escolha muda por trimestre."""
    fig = go.Figure()
    if all_results.empty:
        return fig
    quarters = [s for s in ("DJF", "MAM", "JJA", "SON") if s in all_results["season"].unique()]
    if not quarters:
        return fig
    selected = _ablation_select_split(all_results[all_results["season"].isin(quarters)])
    if selected.empty:
        return fig
    direction = ABLATION_METRIC_DIRECTIONS.get(metric, "higher")
    winners = []
    for (cid, season), g in selected.groupby(["cluster_id", "season"]):
        g = g.dropna(subset=[metric])
        if g.empty:
            continue
        if direction == "higher":
            row = g.loc[g[metric].idxmax()]
        elif direction == "lower":
            row = g.loc[g[metric].idxmin()]
        else:
            row = g.loc[g[metric].abs().idxmin()]
        winners.append(row)
    if not winners:
        return fig
    winners_df = pd.DataFrame(winners)
    winners_df["cluster_id"] = winners_df["cluster_id"].astype(int)
    winners_df["season"] = pd.Categorical(winners_df["season"], categories=quarters, ordered=True)
    winners_df = winners_df.sort_values(["cluster_id", "season"])

    cluster_labels = [f"Cluster {c}" for c in winners_df["cluster_id"]]
    season_labels = winners_df["season"].astype(str).tolist()

    # Trace real PRIMEIRO — traces "fantasma" (só pra legenda de cor) antes
    # dela confundem a inferência de tipo do eixo X do Plotly e as barras
    # somem (mesmo bug já visto na versão anterior deste gráfico).
    fig.add_trace(go.Bar(
        x=[cluster_labels, season_labels],
        y=winners_df[metric],
        marker_color=[PIPELINE_COLORS.get(p, "#898781") for p in winners_df["pipeline"]],
        showlegend=False,
        customdata=np.column_stack([
            [PIPELINE_LABELS.get(p, p) for p in winners_df["pipeline"]],
            winners_df["arm"].to_numpy(),
        ]),
        hovertemplate=(
            "<b>%{x}</b><br>%{customdata[0]} · %{customdata[1]}<br>"
            f"{metric}: " + "%{y:.4f}<extra></extra>"
        ),
    ))
    # Sempre mostra as 3 pipelines na legenda, mesmo as que não venceram
    # nenhum par cluster×trimestre — omitir deixaria parecer que a pipeline
    # nem foi considerada na comparação, quando na verdade só perdeu sempre.
    for pipeline in ABLATION_PIPELINES:
        fig.add_trace(go.Bar(
            x=[[cluster_labels[0]], [season_labels[0]]], y=[0],
            marker_color=PIPELINE_COLORS.get(pipeline),
            name=PIPELINE_LABELS.get(pipeline, pipeline), showlegend=True,
            hoverinfo="skip",
        ))
    fig.update_layout(
        yaxis_title=metric,
        barmode="overlay",
        template="plotly_white", height=480,
        legend={"orientation": "h", "y": -0.22},
        margin={"t": 20, "b": 80},
    )
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
        title="ML Models — top 5 per configuration (by R²)",
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
    confiabilidade já corrigido em `build_interp_map`). Tema (basemap/halo/
    legenda) segue `_map_theme_colors()`, a mesma paleta adaptativa."""
    _theme = _map_theme_colors()
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
            marker_line_color=_theme["cluster_line_color"],
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
        # Mesmo tamanho de marcador/halo dos demais mapas (build_interp_map):
        # estação normal = 7/9, só a selecionada fica um pouco maior pra
        # continuar destacável.
        sizes = [10 if sel_station == r.estacao else 7 for r in stations_df.itertuples()]
        halo_sizes = [13 if sel_station == r.estacao else 9 for r in stations_df.itertuples()]
        sel_color = "#ff6b5b" if _theme["is_dark"] else "#c0392b"
        halo_colors = [
            sel_color if sel_station == r.estacao else _theme["halo_color"]
            for r in stations_df.itertuples()
        ]

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

    # Cobertura ERA5 — máscara real de cada domínio (novo Basin × antigo
    # 18UTC/Paraná-only), não mais o retângulo bruto do bounding-box: a
    # interseção com a geometria real da bacia (mesma usada pros clusters em
    # build_interp_map) recorta a parte do retângulo que cai fora da área de
    # estudo, deixando só a região que de fato importa preenchida. Vermelho/
    # verde mais claros no tema escuro (os tons originais escurecem demais e
    # quase somem no basemap carto-darkmatter).
    old_color = "#ff6f6f" if _theme["is_dark"] else "#b23a3a"
    new_color = "#3fcf7f" if _theme["is_dark"] else "#1f8a4c"
    _, basin_union = load_basin_geometries()

    def _era5_mask_trace(extent: dict, color: str, label: str) -> go.Scattermap:
        box = shapely.box(
            extent["lon"][0], extent["lat"][0], extent["lon"][1], extent["lat"][1],
        )
        mask = shapely.intersection(box, basin_union)
        m_lons, m_lats = _polygon_boundary_lonlat(mask)
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        return go.Scattermap(
            lat=m_lats, lon=m_lons, mode="lines", fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.12)",
            line={"color": color, "width": 2},
            hoverinfo="skip", name=label,
        )

    fig.add_trace(_era5_mask_trace(
        ERA5_18UTC_EXTENT, old_color, "ERA5 18UTC coverage (old, Paraná only)",
    ))
    fig.add_trace(_era5_mask_trace(
        ERA5_BASIN_EXTENT, new_color, "ERA5-Basin coverage (new)",
    ))

    fig.update_layout(
        # Centro/zoom ajustados pra caber as 243 estações do INMET expandido
        # (lat -33.7..-14.4, lon -57.1..-39.9 — vai de RS até Minas Gerais),
        # não só o antigo recorte Sul (RS/SC/PR, zoom 5 cortava a metade norte).
        map={"style": _theme["map_style"], "center": {"lat": -24.0, "lon": -48.5}, "zoom": 4},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": _theme["font_color"]},
        uirevision="map",
        showlegend=True,
        legend={"orientation": "h", "y": 0, "bgcolor": _theme["legend_bgcolor"]},
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
    "P90": ("quantile", 0.90),
    "P95": ("quantile", 0.95),
    "P99": ("quantile", 0.99),
    "Mean": ("mean", None),
}
INTERP_VALUES = {
    "Predicted": "y_pred",
    "Observed (INMET)": "y_true",
    "ERA5 raw": "era5_wind_mag_max",
    "Correction (Pred − ERA5)": "_correction",
    "Error (|Pred − Obs|)": "_abs_error",
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
    elif col == "_abs_error":
        df["_abs_error"] = (df["y_pred"] - df["y_true"]).abs()
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
def aggregate_station_values_by_season(
    combo_dir: str, value_label: str, metric_label: str, season: str
) -> pd.DataFrame:
    """Igual a aggregate_station_values(), mas restrito a um trimestre
    climático (DJF/MAM/JJA/SON) — usado pelos mapas espaciais 'melhor
    modelo por cluster × trimestre'."""
    df = load_mlp_predictions_by_station(combo_dir)
    cols = ["estacao", "latitude", "longitude", "cluster_id", "value"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.copy()
    df["season"] = df["time"].dt.month.map(_MONTH_TO_SEASON)
    df = df[df["season"] == season]
    if df.empty:
        return pd.DataFrame(columns=cols)
    col = INTERP_VALUES[value_label]
    if col == "_correction":
        df["_correction"] = df["y_pred"] - df["era5_wind_mag_max"]
    elif col == "_abs_error":
        df["_abs_error"] = (df["y_pred"] - df["y_true"]).abs()
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


@st.cache_data
def _idw_grid(st_lon, st_lat, st_val, method, n=70, margin=0.6,
              power=2.0, eps=1e-3, sigma=0.6):
    """Campo IDW numa grade regular. Distâncias em graus — aproximação plana,
    suficiente pro recorte pequeno do Sul do Brasil. Cacheado: é uma função
    pura sobre os valores das estações — sem isso, cada rerun de um fragment
    (troca de metric/smoothing/cluster focus) refaz a interpolação em ~4900
    pontos do zero, mesmo quando os valores de entrada não mudaram."""
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


def _map_theme_colors() -> dict:
    """Paleta de mapa adaptada ao tema ativo do Streamlit (claro/escuro) —
    usada por build_interp_map() e build_inspector_map(), pra manter os dois
    mapas da dashboard consistentes entre si e com o resto da UI."""
    try:
        is_dark = st.context.theme.type == "dark"
    except Exception:
        is_dark = False
    return {
        "is_dark": is_dark,
        "map_style": "carto-darkmatter" if is_dark else "carto-positron",
        "halo_color": "#f2f2f2" if is_dark else "black",
        "boundary_color": "rgba(230,230,230,0.85)" if is_dark else "rgba(60,60,60,0.75)",
        "font_color": "#e6e6e6" if is_dark else "#2b2b2b",
        "legend_bgcolor": "rgba(30,30,30,0.7)" if is_dark else "rgba(255,255,255,0.7)",
        "cluster_line_color": "rgba(230,230,230,0.85)" if is_dark else "white",
    }


def _polygon_boundary_lonlat(geom) -> tuple[list, list]:
    """Extrai lon/lat do contorno (exterior + buracos) de um Polygon ou
    MultiPolygon shapely, com None separando cada anel — formato que
    go.Scattermap(mode='lines') precisa pra desenhar contornos desconexos
    numa única trace."""
    lons: list = []
    lats: list = []
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    for poly in parts:
        for ring in (poly.exterior, *poly.interiors):
            xs, ys = ring.xy
            lons.extend(xs)
            lats.extend(ys)
            lons.append(None)
            lats.append(None)
    return lons, lats


def build_interp_map(
    sdf: pd.DataFrame, method_label: str, is_diff: bool,
    cmin_override: float | None = None, cmax_override: float | None = None,
    height: int = 520, show_colorbar: bool = True,
    mask_geom=None, bounds_override: dict | None = None,
    hover_extra_by_station: dict[str, str] | None = None,
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
    calcularia seu próprio enquadramento a partir das suas próprias estações.
    `hover_extra_by_station` (dict estacao -> texto multi-linha, opcional):
    quando informado, o hover de CADA estação mostra o valor dela em TODOS
    os painéis (Observado/ERA5/LazyPredict/MLP/LSTM), não só o deste painel
    — permite comparar os 5 valores sem trocar de mapa."""
    _theme = _map_theme_colors()
    map_style, halo_color = _theme["map_style"], _theme["halo_color"]
    boundary_color, font_color = _theme["boundary_color"], _theme["font_color"]

    fig = go.Figure()
    base_layout = {
        "map": {"style": map_style,
                "center": {"lat": -28.5, "lon": -52.5}, "zoom": 5},
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0}, "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": font_color},
    }
    if sdf.empty or len(sdf) < 2:
        fig.update_layout(**base_layout)
        return fig
    lon = sdf["longitude"].to_numpy(float)
    lat = sdf["latitude"].to_numpy(float)
    val = sdf["value"].to_numpy(float)
    flon, flat, fz = _idw_grid(lon, lat, val, INTERP_METHODS[method_label])

    # Recorte + contorno da área com dado: usa mask_geom se foi passado
    # (Seção 2 — cluster/bacia focada, compartilhado entre painéis); senão
    # deriva automaticamente a união dos clusters presentes no próprio sdf
    # (Seção 1 — mapas "melhor por cluster", que antes ficavam sem recorte
    # nenhum e o campo IDW se espalhava pelo retângulo bounding-box inteiro,
    # inclusive fora de qualquer cluster com dado).
    if mask_geom is not None:
        boundary_geom = mask_geom
    else:
        per_polygon, basin_union = load_basin_geometries()
        cluster_ids = sdf["cluster_id"].unique() if "cluster_id" in sdf.columns else []
        present = {f"{int(float(c)):02d}" for c in cluster_ids}
        geoms = [per_polygon[c] for c in present if c in per_polygon]
        boundary_geom = shapely.union_all(geoms) if geoms else basin_union
    shapely.prepare(boundary_geom)
    inside = shapely.contains(boundary_geom, shapely.points(flon, flat))
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
    # pode estar bem errada longe de qualquer estação. Mesmo assim precisa
    # ser bem visível (tamanho/opacidade próximos do marcador de estação);
    # o contorno da área com dado (abaixo) já comunica "isso é estimativa
    # dentro da região coberta", sem precisar esmaecer o campo a ponto de
    # ficar invisível.
    field_marker = {
        "size": 7, "color": fz, "colorscale": cscale, "cmin": cmin,
        "cmax": cmax, "opacity": 0.65, "showscale": show_colorbar,
    }
    if show_colorbar:
        field_marker["colorbar"] = {"title": "m/s"}
    # go.Scattermap não aceita marker.line (sem borda nativa, ao contrário de
    # go.Scatter) — o "contorno" é simulado com um halo: círculo um pouco
    # maior (cor adaptada ao tema, pra não sumir no basemap escuro) desenhado
    # ANTES do círculo colorido, no mesmo ponto, sem transparência.
    # Círculo (não "star"): símbolos não-circulares em go.Scattermap dependem
    # de um sprite de ícone carregado de forma assíncrona pelo maplibre — numa
    # atualização rápida de bounds (troca de cluster focus) essa corrida pode
    # falhar silenciosamente e a camada de símbolo inteira não aparece (visto
    # ao vivo: estações sumiram do mapa depois de trocar o cluster focus,
    # mesmo a legenda continuando a mostrar "Stations"). Círculo é o marcador
    # nativo do Scattermap, sem essa dependência — sempre aparece.
    station_halo_marker = {"size": 9, "color": halo_color}
    station_marker = {
        "size": 7, "color": val, "colorscale": cscale, "cmin": cmin,
        "cmax": cmax, "showscale": False,
    }
    if cmid is not None:
        field_marker["cmid"] = cmid
        station_marker["cmid"] = cmid

    b_lons, b_lats = _polygon_boundary_lonlat(boundary_geom)
    fig.add_trace(go.Scattermap(
        lat=b_lats, lon=b_lons, mode="lines",
        line={"width": 1.5, "color": boundary_color},
        hoverinfo="skip", showlegend=False, name="Data coverage",
    ))
    fig.add_trace(go.Scattermap(
        lat=flat, lon=flon, mode="markers", marker=field_marker,
        hovertemplate="%{lat:.2f}, %{lon:.2f}<br>~%{marker.color:.2f} m/s (estimated)<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scattermap(
        lat=lat, lon=lon, mode="markers", marker=station_halo_marker,
        hoverinfo="skip", showlegend=False,
    ))
    if hover_extra_by_station:
        extra = sdf["estacao"].map(hover_extra_by_station).fillna("").to_numpy()
        station_customdata = np.column_stack([
            sdf["estacao"].to_numpy(), sdf["cluster_id"].to_numpy(), extra,
        ])
        station_hovertemplate = (
            "<b>%{customdata[0]}</b> (actual data)<br>Cluster %{customdata[1]}<br>"
            "%{customdata[2]}<extra></extra>"
        )
    else:
        station_customdata = sdf[["estacao", "cluster_id", "value"]].values
        station_hovertemplate = (
            "<b>%{customdata[0]}</b> (actual data)<br>Cluster %{customdata[1]}<br>"
            "%{customdata[2]:.2f} m/s<extra></extra>"
        )
    fig.add_trace(go.Scattermap(
        lat=lat, lon=lon, mode="markers", marker=station_marker,
        customdata=station_customdata,
        hovertemplate=station_hovertemplate,
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


# ── Seção 4 — Corrected Grid Explorer ─────────────────────────────────────────
# Grid corrigido (artifacts/corrected_grid/v1/) é um produto espacial único
# (best-model-per-cluster/trimestre "costurado", ver grid_generator.py no
# repo de pesquisa), não uma combinação pipeline×arm — por isso vive fora da
# matriz de ablation, com seus próprios loaders.

@st.cache_resource
def load_corrected_grid() -> xr.Dataset:
    """Concatena os .nc anuais e reconstrói o ERA5 original localmente via
    `rajada_corrigida - bias` (ver `rajada_corrigida = era5_vals + bias_grid`
    em grid_generator.py) — evita carregar o ERA5-Basin bruto (~6GB), que
    não vive neste repo. cache_resource (não cache_data): mantém o mesmo
    xr.Dataset entre reruns, em vez de serializar via pickle.

    Abre cada arquivo com open_dataset e combina em memória (combine_by_coords)
    em vez de open_mfdataset — este último exige `dask` (chunkmanager), que não
    está instalado na nuvem. Os grids são pequenos (~16MB/ano), então carregar
    tudo eager em numpy é barato e dispensa a dependência pesada."""
    files = sorted(CORRECTED_GRID_DIR.glob("grid_corrected_*.nc"))
    if not files:
        return xr.Dataset()
    ds = xr.combine_by_coords(
        [xr.open_dataset(f) for f in files]
    ).sortby("time")
    ds["ws_original"] = ds["rajada_max_corrigida"] - ds["bias"]
    return ds


@st.cache_data
def corrected_grid_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp] | None:
    ds = load_corrected_grid()
    if not ds.data_vars:
        return None
    times = pd.to_datetime(ds.time.values)
    return times.min(), times.max()


@st.cache_data
def corrected_grid_snapshot(date_str: str) -> pd.DataFrame:
    """Achata o grid num dia pra long-form (longitude/latitude/original/
    corrigido), formato consumido por build_grid_map (mesmo padrão de
    'pontos minúsculos coloridos por valor' de build_interp_map)."""
    cols = ["longitude", "latitude", "ws_original", "rajada_max_corrigida"]
    ds = load_corrected_grid()
    if not ds.data_vars:
        return pd.DataFrame(columns=cols)
    day = ds.sel(time=date_str, method="nearest")
    lon2d, lat2d = np.meshgrid(day.longitude.values, day.latitude.values)
    df = pd.DataFrame({
        "longitude": lon2d.ravel(),
        "latitude": lat2d.ravel(),
        "ws_original": np.asarray(day["ws_original"].values).ravel(),
        "rajada_max_corrigida": np.asarray(day["rajada_max_corrigida"].values).ravel(),
    })
    return df.dropna()


@st.cache_data
def corrected_grid_station_series(lat: float, lon: float) -> pd.DataFrame:
    """Série temporal (original/corrigido) no PIXEL do grid mais próximo de
    (lat, lon) — comparação correta ponto-a-ponto com uma estação, em vez de
    média espacial da bacia inteira (rajada é um fenômeno local; a média da
    bacia suaviza os picos que uma estação isolada registra)."""
    cols = ["time", "ws_original", "rajada_max_corrigida"]
    ds = load_corrected_grid()
    if not ds.data_vars:
        return pd.DataFrame(columns=cols)
    point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    return pd.DataFrame({
        "time": pd.to_datetime(point.time.values),
        "ws_original": np.asarray(point["ws_original"].values),
        "rajada_max_corrigida": np.asarray(point["rajada_max_corrigida"].values),
    })


@st.cache_data
def grid_station_percentile_values(metric_label: str, stations_df: pd.DataFrame) -> pd.DataFrame:
    """Valor agregado (P90/P95/P99/Mean/Historical max, mesmo INTERP_METRICS
    do resto do dashboard) do ERA5 original e corrigido no pixel mais
    próximo de CADA estação, extraído de uma vez com indexação vetorizada do
    xarray (em vez de 243 chamadas .sel individuais) — insumo do mapa
    espacial de "captura de extremos" (erro nos percentis altos, onde mora
    o vendaval, não na média)."""
    cols = ["estacao", "latitude", "longitude", "cluster_id", "era5_original", "era5_corrected"]
    ds = load_corrected_grid()
    if not ds.data_vars or stations_df.empty:
        return pd.DataFrame(columns=cols)

    agg, q = INTERP_METRICS[metric_label]
    lat_da = xr.DataArray(stations_df["latitude"].to_numpy(float), dims="estacao")
    lon_da = xr.DataArray(stations_df["longitude"].to_numpy(float), dims="estacao")
    # .load(): materializa os ~243 pontos extraídos (pequeno) antes do
    # quantile — dask recusa quantile com "time" com múltiplos chunks (um
    # chunk por arquivo/ano do open_mfdataset) como dimensão núcleo.
    pts = ds[["ws_original", "rajada_max_corrigida"]].sel(
        latitude=lat_da, longitude=lon_da, method="nearest",
    ).load()
    if agg == "max":
        vals = pts.max(dim="time", skipna=True)
    elif agg == "mean":
        vals = pts.mean(dim="time", skipna=True)
    else:
        vals = pts.quantile(q, dim="time", skipna=True)

    return pd.DataFrame({
        "estacao": stations_df["estacao"].to_numpy(),
        "latitude": stations_df["latitude"].to_numpy(),
        "longitude": stations_df["longitude"].to_numpy(),
        "cluster_id": stations_df["cluster_id"].to_numpy(),
        "era5_original": np.asarray(vals["ws_original"].values, dtype=float),
        "era5_corrected": np.asarray(vals["rajada_max_corrigida"].values, dtype=float),
    })


@st.cache_data
def load_inmet_station_series(estacao: str) -> pd.Series:
    """Rajada máxima diária observada (INMET_Stratified.nc) pra uma única
    estação — mesma variável física de rajada_max_corrigida."""
    if not INMET_RAW_PATH.exists():
        return pd.Series(dtype=float)
    ds = xr.open_dataset(INMET_RAW_PATH)
    if estacao not in ds["estacao"].values:
        ds.close()
        return pd.Series(dtype=float)
    da = ds[INMET_TARGET_VAR].sel(estacao=estacao)
    s = pd.Series(np.asarray(da.values, dtype=float), index=pd.to_datetime(da["time"].values))
    ds.close()
    return s


GRID_VMIN, GRID_VMAX = 0.0, 20.0  # rajada máxima (m/s) — mesma escala do notebook de referência


def build_grid_map(
    df: pd.DataFrame, value_col: str, stations_df: pd.DataFrame,
    sel_station: str | None, title: str, height: int = 380,
    show_colorbar: bool = True,
) -> go.Figure:
    """Raster do grid corrigido como pontos pequenos/esmaecidos coloridos por
    valor (mesma linguagem visual do campo IDW em build_interp_map) +
    estações INMET sobrepostas (halo + marcador, mesmo padrão de
    build_inspector_map), com a estação selecionada destacada. cmin/cmax
    fixos (GRID_VMIN/GRID_VMAX) pros dois painéis (original/corrigido)
    dividirem a MESMA escala de cor — show_colorbar=False no 1º painel
    evita duas barras de cor idênticas lado a lado (mesmo padrão de
    build_interp_map/show_colorbar usado no resto do dashboard)."""
    _theme = _map_theme_colors()
    fig = go.Figure()

    if not df.empty:
        marker = {
            "size": 6, "color": df[value_col], "colorscale": "Turbo",
            "cmin": GRID_VMIN, "cmax": GRID_VMAX, "opacity": 0.55,
            "showscale": show_colorbar,
        }
        if show_colorbar:
            marker["colorbar"] = {"title": "m/s"}
        fig.add_trace(go.Scattermap(
            lat=df["latitude"], lon=df["longitude"], mode="markers",
            marker=marker,
            hovertemplate="%{lat:.2f}, %{lon:.2f}<br>~%{marker.color:.2f} m/s<extra></extra>",
            showlegend=False, name="Grid",
        ))

    if not stations_df.empty:
        sizes = [10 if sel_station == r.estacao else 6 for r in stations_df.itertuples()]
        halo_sizes = [13 if sel_station == r.estacao else 9 for r in stations_df.itertuples()]
        sel_color = "#ff6b5b" if _theme["is_dark"] else "#c0392b"
        halo_colors = [
            sel_color if sel_station == r.estacao else _theme["halo_color"]
            for r in stations_df.itertuples()
        ]
        fig.add_trace(go.Scattermap(
            lat=stations_df["latitude"], lon=stations_df["longitude"],
            mode="markers", marker={"size": halo_sizes, "color": halo_colors},
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scattermap(
            lat=stations_df["latitude"], lon=stations_df["longitude"],
            mode="markers", marker={"size": sizes, "color": "black"},
            customdata=stations_df["estacao"].values,
            hovertemplate="<b>%{customdata}</b><extra></extra>",
            name="Estações INMET",
        ))

    fig.update_layout(
        map={
            "style": _theme["map_style"],
            "center": {
                "lat": sum(ERA5_BASIN_EXTENT["lat"]) / 2,
                "lon": sum(ERA5_BASIN_EXTENT["lon"]) / 2,
            },
            "zoom": 4,
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0}, height=height,
        paper_bgcolor="rgba(0,0,0,0)", font={"color": _theme["font_color"]},
        title=title, uirevision="corrected-grid-map",
    )
    return fig


def build_grid_station_timeseries(
    estacao: str | None, series_df: pd.DataFrame, inmet_series: pd.Series, year: int,
) -> go.Figure:
    """Mesma paleta/estilo de build_residual_timeseries: royalblue = INMET
    observado, darkorange = ERA5 corrigido, tomato pontilhado = ERA5
    original — comparação no pixel do grid mais próximo da estação."""
    fig = go.Figure()
    if estacao is None or series_df.empty:
        fig.update_layout(
            title="Selecione uma estação no mapa",
            template="plotly_white", height=380,
        )
        return fig

    s = series_df[series_df["time"].dt.year == year]
    obs = inmet_series[inmet_series.index.year == year] if inmet_series is not None else None

    fig.add_trace(go.Scatter(
        x=s["time"], y=s["ws_original"], name="ERA5 original (pixel)",
        line={"color": "tomato", "width": 1, "dash": "dot"},
    ))
    fig.add_trace(go.Scatter(
        x=s["time"], y=s["rajada_max_corrigida"], name="ERA5 corrigido (pixel)",
        line={"color": "darkorange", "width": 1.5},
    ))
    if obs is not None and not obs.empty:
        fig.add_trace(go.Scatter(
            x=obs.index, y=obs.values, name=f"INMET — {estacao}",
            line={"color": "royalblue", "width": 1.5},
        ))

    fig.update_layout(
        title=f"Rajada máxima diária — {estacao} ({year})",
        xaxis_title="Data", yaxis_title="Rajada máxima (m/s)",
        template="plotly_white", height=380,
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 50, "b": 60},
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

st.title("Extreme Wind Gust Bias Correction")

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
    st.title("IRC Vendaval")
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

tab_global, tab_inspector, tab_diag, tab_grid = st.tabs([
    "Global Comparison Panel",
    "Spatial & Temporal Error Inspector",
    "Model Diagnostics & Explainability",
    "AI database",
])

# ── Seção 1: Global Comparison Panel ──────────────────────────────────────────
# Cada aba roda como um st.fragment: um widget mexido DENTRO da aba só
# reprocessa aquela aba (não as outras 3), evitando recomputar IDW/OLS de
# todas as seções a cada interação — só um rerun completo (troca de algo na
# barra lateral, fora de qualquer fragment) ainda reprocessa tudo.

@st.fragment
def _render_tab_global():
    if not ablation_combos:
        st.info(
            "No results synced yet. Run "
            "`scripts/sync_ablation_to_dashboard.py` from the research repo "
            "after a pipeline finishes.",
            icon="ℹ️",
        )
    else:
        all_results = load_all_ablation(ablation_combos)
        derived_quarterly = derive_quarterly_results(ablation_combos, all_results)
        if not derived_quarterly.empty:
            all_results = pd.concat([all_results, derived_quarterly], ignore_index=True)
        panel_clusters = (
            sorted(all_results["cluster_id"].unique(), key=str)
            if not all_results.empty else []
        )

        col_metric, col_cluster = st.columns([1, 1])
        with col_metric:
            metric = st.selectbox("Metric", ABLATION_METRICS, key="ablation_metric")
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

        # Cards/barras/heatmap/tabela abaixo usam a visão agregada 'ALL'
        # (todos os trimestres combinados) — o detalhamento por trimestre
        # isolado fica no bloco "Per-quarter snapshot" logo abaixo, sem
        # precisar de um seletor pra trocar de trimestre.
        panel_season = "ALL"
        by_season = _resolve_all_season(all_results)
        if by_season.empty:
            st.info(
                "No combination has data yet — pipelines currently differ "
                "in season coverage (e.g. MLP only reports 'ALL' today).",
                icon="ℹ️",
            )

        by_cluster = by_season
        if panel_cluster_choice != "All clusters (weighted avg)":
            chosen_cluster = panel_cluster_choice.removeprefix("Cluster ")
            by_cluster = by_season[by_season["cluster_id"].astype(str) == chosen_cluster]

        selected_rows = _ablation_select_split(by_cluster)
        summary_df = _ablation_aggregate(selected_rows, ["pipeline", "arm"])

        # ── Best per cluster (barras + mapa espacial) — lado a lado pra
        # economizar espaço vertical. O 1º mostra, por cluster, a MELHOR
        # combinação pipeline×configuração entre TODAS as testadas — não é
        # filtrado por "Cluster" acima (esse seletor foca 1 cluster; este
        # gráfico já compara todos). O 2º é a versão espacial do 1º: em vez
        # de barra, mostra o campo interpolado do erro do modelo (|Predito −
        # INMET observado|, só onde há observação real), "costurando" o
        # resultado de cada cluster com o combo que venceu ali.
        col_best_cluster, col_best_map = st.columns(2)
        with col_best_cluster:
            _best_cluster_fig = build_best_per_cluster_bars(all_results, metric)
            if not _best_cluster_fig.data:
                st.caption("No data yet.")
            else:
                st.plotly_chart(
                    _best_cluster_fig, use_container_width=True, key="best_per_cluster_chart",
                )
        with col_best_map:
            _all_winners = _best_combo_per_cluster(all_results, metric)
            _best_map_sdf = _sdf_from_cluster_winners(
                ablation_combos, _all_winners, "Error (|Pred − Obs|)", "Mean",
            )
            if _best_map_sdf.empty or len(_best_map_sdf) < 2:
                st.caption("No data yet.")
            else:
                st.plotly_chart(
                    build_interp_map(_best_map_sdf, "IDW (original)", False, height=380),
                    use_container_width=True, key="best_per_cluster_map",
                )
        st.divider()

        # ── Best per cluster × trimestre — mapa espacial ─────────────────
        # Mesma ideia do mapa acima, mas 1 mapa por trimestre (DJF/MAM/JJA/
        # SON), cada um "costurado" com o vencedor daquele cluster NAQUELE
        # trimestre especificamente (não o vencedor do ano inteiro).
        _quarters_for_maps = [
            s for s in ("DJF", "MAM", "JJA", "SON") if s in all_results["season"].unique()
        ]
        if not _quarters_for_maps:
            st.caption("No per-quarter rows synced yet (only 'ALL' available).")
        else:
            # Calcula os 4 sdf's ANTES de desenhar, pra poder compartilhar a
            # mesma escala de cor (cmin/cmax) e mostrar só 1 colorbar — mesmo
            # padrão já usado no comparativo de 5 painéis da Seção 2.
            _quarter_sdfs = []
            for quarter in _quarters_for_maps:
                q_winners = _best_combo_per_cluster(all_results, metric, season=quarter)
                q_sdf = _sdf_from_cluster_winners(
                    ablation_combos, q_winners, "Error (|Pred − Obs|)", "Mean", season=quarter,
                )
                _quarter_sdfs.append((quarter, q_sdf))

            _quarter_vals = pd.concat(
                [s["value"] for _, s in _quarter_sdfs if s is not None and not s.empty],
                ignore_index=True,
            )
            _q_cmin = float(_quarter_vals.min()) if not _quarter_vals.empty else None
            _q_cmax = float(_quarter_vals.max()) if not _quarter_vals.empty else None
            _q_last_valid = max(
                (i for i, (_, s) in enumerate(_quarter_sdfs) if s is not None and len(s) >= 2),
                default=-1,
            )

            quarter_map_cols = st.columns(len(_quarters_for_maps))
            for i, (q_col, (quarter, q_sdf)) in enumerate(zip(quarter_map_cols, _quarter_sdfs)):
                with q_col:
                    st.caption(quarter)
                    if q_sdf.empty or len(q_sdf) < 2:
                        st.caption("No data.")
                    else:
                        st.plotly_chart(
                            build_interp_map(
                                q_sdf, "IDW (original)", False, height=340,
                                cmin_override=_q_cmin, cmax_override=_q_cmax,
                                show_colorbar=(i == _q_last_valid),
                            ),
                            use_container_width=True, key=f"best_per_cluster_map_{quarter}",
                        )
        st.divider()

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

        _quarter_pipeline_fig = build_ablation_bars_by_quarter(all_results, metric)
        if _quarter_pipeline_fig.data:
            st.plotly_chart(
                _quarter_pipeline_fig, use_container_width=True,
                key="ablation_bars_by_quarter_chart",
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

        with st.expander("ML Models — top 5 per configuration"):
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

with tab_global:
    _render_tab_global()

# ── Seção 2: Spatial & Temporal Error Inspector ───────────────────────────────

@st.fragment
def _render_tab_inspector():
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
            ("ML Models", _lazy_sdf),
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

        # Junta o valor de cada estação em TODOS os painéis num único texto
        # de hover, pra comparar Observado/ERA5/LazyPredict/MLP/LSTM sem
        # precisar passar o mouse painel por painel.
        _values_by_station: dict[str, dict[str, float]] = {}
        for label, sdf in multi_panels:
            if sdf is None or sdf.empty:
                continue
            for est, v in zip(sdf["estacao"], sdf["value"]):
                _values_by_station.setdefault(est, {})[label] = v
        _hover_extra_by_station = {
            est: "<br>".join(
                f"{label}: {values[label]:.2f} m/s"
                for label, _ in multi_panels
                if label in values and np.isfinite(values[label])
            )
            for est, values in _values_by_station.items()
        }

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
                            hover_extra_by_station=_hover_extra_by_station,
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
                st.info("Select ML Models and/or LSTM to see their residual distribution.", icon="ℹ️")

with tab_inspector:
    _render_tab_inspector()

# ── Seção 3: Model Diagnostics & Explainability ───────────────────────────────

@st.fragment
def _render_tab_diag():
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

with tab_diag:
    _render_tab_diag()

# ── Seção 4: Corrected Grid Explorer ──────────────────────────────────────────

@st.fragment
def _render_tab_grid():
    _grid_bounds = corrected_grid_date_bounds()
    if _grid_bounds is None:
        st.info(
            "No corrected grid synced yet. Run "
            "`scripts/sync_corrected_grid_to_dashboard.py` from the research "
            "repo after `src/dataset/creation/grid_generator.py` finishes.",
            icon="ℹ️",
        )
    else:
        _gmin, _gmax = _grid_bounds

        # Seletor de estação PRÓPRIO (não usa o global_station da barra
        # lateral): esse é escopado ao cluster selecionado ali (relevante só
        # pra matriz de ablation), enquanto o grid v1 é cluster-agnóstico —
        # reusar global_station faria a seleção "sumir" sempre que a estação
        # escolhida não pertencesse ao cluster atual da barra lateral.
        _grid_all_stations = stations_geo_df["estacao"].tolist()
        col_gdate, col_gstation = st.columns([1, 2])
        with col_gdate:
            grid_date = st.date_input(
                "Data do mapa", value=pd.Timestamp("2020-05-15"),
                min_value=_gmin.date(), max_value=_gmax.date(),
                key="grid_map_date",
            )
        with col_gstation:
            grid_station_choice = st.selectbox(
                "Estação", ["(none)"] + _grid_all_stations,
                key="grid_station_select",
            )
        grid_station = None if grid_station_choice == "(none)" else grid_station_choice

        grid_snapshot = corrected_grid_snapshot(str(grid_date))

        col_map1, col_map2 = st.columns(2)
        with col_map1:
            st.plotly_chart(
                build_grid_map(
                    grid_snapshot, "ws_original", stations_geo_df,
                    grid_station, "ERA5 original", show_colorbar=False,
                ),
                use_container_width=True, key="grid_map_original",
            )
        with col_map2:
            st.plotly_chart(
                build_grid_map(
                    grid_snapshot, "rajada_max_corrigida", stations_geo_df,
                    grid_station, "ERA5 corrigido", show_colorbar=True,
                ),
                use_container_width=True, key="grid_map_corrected",
            )

        if grid_station is None:
            st.info("Select a station above to see its time series.", icon="ℹ️")
        else:
            _st_row = stations_geo_df[stations_geo_df["estacao"] == grid_station]
            if _st_row.empty:
                st.warning(f"Station {grid_station} not found in stations_metadata.csv.")
            else:
                _st_lat = float(_st_row["latitude"].iloc[0])
                _st_lon = float(_st_row["longitude"].iloc[0])
                grid_series = corrected_grid_station_series(_st_lat, _st_lon)
                inmet_series = load_inmet_station_series(grid_station)
                st.plotly_chart(
                    build_grid_station_timeseries(
                        grid_station, grid_series, inmet_series, grid_date.year,
                    ),
                    use_container_width=True, key="grid_station_timeseries",
                )

        st.divider()
        extreme_metric = st.selectbox(
            "Métrica", list(INTERP_METRICS),
            index=list(INTERP_METRICS).index("P95"),
            key="grid_extreme_metric",
        )

        _grid_pct = grid_station_percentile_values(extreme_metric, stations_geo_df).dropna(
            subset=["era5_original", "era5_corrected"],
        )

        if len(_grid_pct) < 2:
            st.info("Not enough stations to build the maps.", icon="ℹ️")
        else:
            # Valor de vento (não erro) das duas bases no percentil escolhido —
            # mais interpretativo: mostra diretamente "quanto vento" cada base
            # captura na cauda extrema, em vez de uma métrica abstrata de erro.
            _val_orig_sdf = _grid_pct[["estacao", "latitude", "longitude", "cluster_id", "era5_original"]].rename(
                columns={"era5_original": "value"},
            )
            _val_corr_sdf = _grid_pct[["estacao", "latitude", "longitude", "cluster_id", "era5_corrected"]].rename(
                columns={"era5_corrected": "value"},
            )
            _val_all = pd.concat([_val_orig_sdf["value"], _val_corr_sdf["value"]], ignore_index=True)
            _val_cmin, _val_cmax = float(_val_all.min()), float(_val_all.max())

            col_val1, col_val2 = st.columns(2)
            with col_val1:
                st.markdown(f"**ERA5 original — {extreme_metric}**")
                st.plotly_chart(
                    build_interp_map(
                        _val_orig_sdf, "IDW (original)", False,
                        cmin_override=_val_cmin, cmax_override=_val_cmax,
                        height=380, show_colorbar=False,
                    ),
                    use_container_width=True, key="grid_extreme_value_original",
                )
            with col_val2:
                st.markdown(f"**ERA5 corrigido — {extreme_metric}**")
                st.plotly_chart(
                    build_interp_map(
                        _val_corr_sdf, "IDW (original)", False,
                        cmin_override=_val_cmin, cmax_override=_val_cmax,
                        height=380, show_colorbar=True,
                    ),
                    use_container_width=True, key="grid_extreme_value_corrected",
                )


with tab_grid:
    _render_tab_grid()
