"""
Apuração — os números que a narrativa mostra.

Regra desta camada: **nenhum número da tela é digitado à mão.** Tudo aqui é
lido dos artefatos (`results.parquet`, `run_meta.json`, `*_cluster_results.csv`)
no momento em que a página carrega, para que o painel não passe a mentir
quando a pipeline for re-executada com dados novos.

Quando um número não existe no artefato, a função devolve `None` e a tela
descreve a lacuna — nunca preenche com um valor plausível.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ABLATION_DIR = Path("artifacts/ablation")
PIPELINES = ["lazy", "mlp", "lstm"]
ARMS = ["original", "synthetic", "newfeatures", "all", "basin", "all_basin"]

# Como chamar cada coisa em português, na tela.
NOME_PIPELINE = {
    "lazy": "Modelos clássicos",
    "mlp": "Rede neural (MLP)",
    "lstm": "Rede recorrente (LSTM)",
}
NOME_ARM = {
    "original": "Base",
    "synthetic": "Base + dados sintéticos",
    "newfeatures": "Variáveis novas",
    "all": "Tudo",
    "basin": "Base + bacia",
    "all_basin": "Tudo + bacia",
}
NOME_METRICA = {
    "R2": "R²",
    "RMSE": "REQM",
    "Bias": "Viés",
    "Bias_P90": "Viés no extremo (P90)",
    "RMSE_P90": "REQM no extremo (P90)",
}


# ── Leitura bruta ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def combos_existentes() -> list[tuple[str, str]]:
    """Pares (pipeline, configuração) que têm `results.parquet` no disco."""
    achados = []
    for pipeline in PIPELINES:
        for arm in ARMS:
            if (ABLATION_DIR / pipeline / arm / "results.parquet").exists():
                achados.append((pipeline, arm))
    return achados


@st.cache_data(show_spinner=False)
def _resultados(pipeline: str, arm: str) -> pd.DataFrame:
    caminho = ABLATION_DIR / pipeline / arm / "results.parquet"
    return pd.read_parquet(caminho) if caminho.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _estacoes_por_area() -> dict:
    """Quantas estações cada área tem, segundo a abordagem que reporta mais.

    Serve só para ponderar; o número que a tela mostra vem de
    `contagem_de_estacoes()`, que não esconde a divergência entre as fontes.
    """
    melhor: dict = {}
    for pipeline, arm in combos_existentes():
        df = _resultados(pipeline, arm)
        if df.empty or "n_stations" not in df.columns:
            continue
        mapa = (
            df.drop_duplicates("cluster_id")
            .set_index("cluster_id")["n_stations"]
            .to_dict()
        )
        if sum(mapa.values()) > sum(melhor.values()):
            melhor = mapa
    return melhor


@st.cache_data(show_spinner=False)
def contagem_de_estacoes() -> dict:
    """As três contagens de estação que os artefatos trazem — de propósito
    separadas, porque elas **não** são iguais.

    - `rede`: quantas estações o projeto cataloga (`stations_metadata.csv`);
    - `por_abordagem`: quantas cada abordagem efetivamente usou, que é menor
      porque nem toda estação tem série utilizável no recorte de cada
      experimento;
    - `sem_registro`: abordagens que não gravam a contagem (o LSTM não grava).

    Achatar isso numa média daria um número que não existe em lugar nenhum.
    """
    rede = None
    for pipeline, arm in combos_existentes():
        caminho = ABLATION_DIR / pipeline / arm / "stations_metadata.csv"
        if caminho.exists():
            rede = int(len(pd.read_csv(caminho)))
            break
    por_abordagem, sem_registro = {}, []
    for pipeline in PIPELINES:
        usadas = set()
        registra = False
        for arm in ARMS:
            df = _resultados(pipeline, arm)
            if df.empty or "n_stations" not in df.columns:
                continue
            registra = True
            linhas = df.drop_duplicates("cluster_id")
            if linhas["cluster_id"].nunique() >= len(_estacoes_por_area()):
                usadas.add(int(linhas["n_stations"].sum()))
        if not registra:
            sem_registro.append(NOME_PIPELINE.get(pipeline, pipeline))
        elif usadas:
            por_abordagem[NOME_PIPELINE.get(pipeline, pipeline)] = sorted(usadas)
    usados = [n for vs in por_abordagem.values() for n in vs]
    return {
        "rede": rede,
        "por_abordagem": por_abordagem,
        "sem_registro": sem_registro,
        "faixa_usada": (min(usados), max(usados)) if usados else None,
    }


def _recorte_teste(df: pd.DataFrame) -> pd.DataFrame:
    """Linhas do split de teste; se o experimento não tem teste, usa o que tem.

    O LazyPredict não grava `test` em todos os casos, e nenhum experimento
    grava a mesma combinação de `season`: MLP só tem 'ALL', LSTM só tem os
    quatro trimestres. Agrega-se 'ALL' quando existe; senão, os trimestres
    ponderados pelo número de amostras.
    """
    if df.empty:
        return df
    splits = set(df["split"].unique())
    df = df[df["split"] == "test"] if "test" in splits else df
    estacoes_do_ano = set(df["season"].unique())
    return df[df["season"] == "ALL"] if "ALL" in estacoes_do_ano else df


# ── Quadro comparativo dos experimentos ───────────────────────────────────────

@st.cache_data(show_spinner=False)
def quadro_experimentos() -> pd.DataFrame:
    """Uma linha por experimento, com as métricas ponderadas pelo número de
    amostras e a cobertura real (quantas das áreas o experimento cobriu)."""
    por_area = _estacoes_por_area()
    total_areas = len(por_area)
    linhas = []
    for pipeline, arm in combos_existentes():
        df = _recorte_teste(_resultados(pipeline, arm))
        if df.empty:
            continue
        peso = df["n_samples"].astype(float)
        areas = sorted(df["cluster_id"].unique())
        linhas.append({
            "pipeline": pipeline,
            "arm": arm,
            "Abordagem": NOME_PIPELINE.get(pipeline, pipeline),
            "Configuração": NOME_ARM.get(arm, arm),
            "areas": len(areas),
            "estacoes": int(sum(por_area.get(a, 0) for a in areas)),
            "completo": len(areas) == total_areas,
            "R2": float(np.average(df["R2"], weights=peso)),
            "RMSE": float(np.average(df["RMSE"], weights=peso)),
            "Bias": float(np.average(df["Bias"], weights=peso)),
            "Bias_P90": float(np.average(df["Bias_P90"], weights=peso)),
            "RMSE_P90": float(np.average(df["RMSE_P90"], weights=peso)),
        })
    return pd.DataFrame(linhas)


@st.cache_data(show_spinner=False)
def baseline_era5() -> dict | None:
    """Erro do ERA5 bruto — o ponto de partida que a correção tenta consertar.

    Só os experimentos do MLP e dos modelos clássicos gravam as colunas
    `ERA5_*`; o LSTM não grava. Como o ERA5 é a entrada e não a saída de
    nenhum modelo, o valor não depende de qual experimento o leu.
    """
    for pipeline, nome_csv in (("mlp", "mlp_cluster_results.csv"),
                               ("lazy", "lazy_cluster_results.csv")):
        for arm in ARMS:
            caminho = ABLATION_DIR / pipeline / arm / nome_csv
            if not caminho.exists():
                continue
            df = pd.read_csv(caminho)
            if "ERA5_test_Bias_P90" not in df.columns or "n_stations" not in df.columns:
                continue
            peso = df["n_stations"].astype(float)
            def med(coluna):
                return float(np.average(df[coluna], weights=peso)) if coluna in df else None
            return {
                "fonte": f"{pipeline}/{arm}",
                "areas": int(len(df)),
                "estacoes": int(df["n_stations"].sum()),
                "Bias_P90": med("ERA5_test_Bias_P90"),
                "RMSE_P90": med("ERA5_test_RMSE_P90"),
                "Bias": med("ERA5_test_Bias"),
                "RMSE": med("ERA5_test_RMSE"),
                "R2": med("ERA5_test_R2"),
                "pior_area_Bias_P90": float(df["ERA5_test_Bias_P90"].min()),
                "melhor_area_Bias_P90": float(df["ERA5_test_Bias_P90"].max()),
            }
    return None


# ── O que distingue cada configuração (lido dos run_meta) ─────────────────────

@st.cache_data(show_spinner=False)
def fichas_das_configuracoes() -> pd.DataFrame:
    """O que cada configuração de fato mudou, medido em `run_meta.json`.

    Não se confia no campo `feature_groups` (ele se repete entre
    configurações que deveriam diferir); conta-se `active_features`, que é a
    lista efetivamente usada pelo modelo.
    """
    quadro = quadro_experimentos()
    linhas = []
    for arm in ARMS:
        variaveis, sinteticos, coberturas = set(), None, []
        n_variaveis = []
        for pipeline in PIPELINES:
            caminho = ABLATION_DIR / pipeline / arm / "run_meta.json"
            if not caminho.exists():
                continue
            meta = json.loads(caminho.read_text())
            ativas = meta.get("active_features") or meta.get("features") or []
            if ativas:
                variaveis |= set(ativas)
                n_variaveis.append(len(ativas))
            if meta.get("synthetic_csv"):
                sinteticos = True
            elif sinteticos is None:
                sinteticos = False
            sub = quadro[(quadro["pipeline"] == pipeline) & (quadro["arm"] == arm)]
            if not sub.empty:
                coberturas.append(bool(sub.iloc[0]["completo"]))
        if not n_variaveis:
            continue
        linhas.append({
            "arm": arm,
            "Configuração": NOME_ARM.get(arm, arm),
            "Variáveis": (
                str(n_variaveis[0]) if len(set(n_variaveis)) == 1
                else f"{min(n_variaveis)}–{max(n_variaveis)}"
            ),
            "Dados sintéticos": "sim" if sinteticos else "não",
            "Cobertura completa": all(coberturas) if coberturas else None,
        })
    return pd.DataFrame(linhas)


@st.cache_data(show_spinner=False)
def configuracoes_identicas() -> list[tuple[str, str]]:
    """Pares de configurações cujo conjunto de variáveis é idêntico.

    Existe porque isso foi medido no acervo atual e muda como a comparação
    deve ser lida: duas configurações com a mesma lista de variáveis não
    testam hipóteses diferentes.
    """
    assinaturas: dict[str, tuple] = {}
    for arm in ARMS:
        caminho = ABLATION_DIR / "mlp" / arm / "run_meta.json"
        if not caminho.exists():
            continue
        meta = json.loads(caminho.read_text())
        ativas = meta.get("active_features") or meta.get("features") or []
        if ativas:
            # Duas configurações só são a mesma coisa quando coincidem em
            # variáveis E em dados de treino: 'Tudo' usa as mesmas variáveis
            # de 'Base', mas acrescenta amostras sintéticas, então testa uma
            # hipótese diferente e não conta como duplicata.
            assinaturas[arm] = (frozenset(ativas), bool(meta.get("synthetic_csv")))
    iguais = []
    nomes = list(assinaturas)
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            if assinaturas[a] == assinaturas[b]:
                iguais.append((a, b))
    return iguais


@st.cache_data(show_spinner=False)
def experimentos_incompletos() -> pd.DataFrame:
    """Experimentos que não cobriram todas as áreas — não são comparáveis de
    igual para igual com os demais, e a tela precisa dizer isso."""
    quadro = quadro_experimentos()
    if quadro.empty:
        return quadro
    return quadro[~quadro["completo"]][
        ["Abordagem", "Configuração", "areas", "estacoes"]
    ]


# ── Leitura de quem venceu ────────────────────────────────────────────────────

DIRECAO = {"R2": "maior", "RMSE": "menor", "Bias": "zero",
           "Bias_P90": "zero", "RMSE_P90": "menor"}


def campeao(metrica: str, apenas_completos: bool = True) -> pd.Series | None:
    """Melhor experimento numa métrica. Por padrão ignora os incompletos, que
    venceriam por cobrir só as áreas fáceis."""
    quadro = quadro_experimentos()
    if quadro.empty:
        return None
    if apenas_completos and quadro["completo"].any():
        quadro = quadro[quadro["completo"]]
    if quadro.empty:
        return None
    direcao = DIRECAO.get(metrica, "maior")
    if direcao == "maior":
        return quadro.loc[quadro[metrica].idxmax()]
    if direcao == "menor":
        return quadro.loc[quadro[metrica].idxmin()]
    return quadro.loc[quadro[metrica].abs().idxmin()]


@st.cache_data(show_spinner=False)
def panorama() -> dict:
    """A linha de números do topo do painel."""
    quadro = quadro_experimentos()
    por_area = _estacoes_por_area()
    base = baseline_era5()
    melhor_extremo = campeao("Bias_P90")
    reducao = None
    if base and base["Bias_P90"] and melhor_extremo is not None:
        antes, depois = abs(base["Bias_P90"]), abs(melhor_extremo["Bias_P90"])
        if antes:
            reducao = (antes - depois) / antes * 100
    estacoes = contagem_de_estacoes()
    return {
        "estacoes": estacoes["rede"] or int(sum(por_area.values())),
        "estacoes_faixa": estacoes["faixa_usada"],
        "areas": len(por_area),
        "experimentos": int(len(quadro)),
        "viés_era5": base["Bias_P90"] if base else None,
        "viés_melhor": float(melhor_extremo["Bias_P90"]) if melhor_extremo is not None else None,
        "reducao_pct": reducao,
    }


@st.cache_data(show_spinner=False)
def modelos_rastreados() -> int | None:
    """Quantos tipos de modelo diferentes o screening de modelos clássicos
    chegou a avaliar."""
    for arm in ARMS:
        caminho = ABLATION_DIR / "lazy" / arm / "lazy_cluster_results.csv"
        if caminho.exists():
            df = pd.read_csv(caminho)
            if "Model" in df.columns:
                return int(df["Model"].nunique())
    return None
