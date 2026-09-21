"""
Apuração, os números que a narrativa mostra.

Regra desta camada: **nenhum número da tela é digitado à mão.** Tudo aqui é
lido dos artefatos (`results.parquet`, `run_meta.json`, `*_cluster_results.csv`)
no momento em que a página carrega, para que o painel não passe a mentir
quando a pipeline for re-executada com dados novos.

Quando um número não existe no artefato, a função devolve `None` e a tela
descreve a lacuna, nunca preenche com um valor plausível.
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
    "lazy": "Modelos clássicos", "mlp": "Rede neural (MLP)", "lstm": "Rede recorrente (LSTM)",
}
NOME_ARM = {
    "original": "Base", "synthetic": "Base + dados sintéticos", "newfeatures": "Variáveis novas", "all": "Tudo", "basin": "Base + bacia", "all_basin": "Tudo + bacia",
}
# O desenho do experimento, como a pipeline o declara: duas perguntas
# cruzadas, mais variáveis × extremos sintéticos. Espelhado aqui para que a
# tela possa CONFERIR se cada artefato publicado corresponde ao braço que diz
# ser, em vez de confiar no nome da pasta.
DESENHO_ABLACAO = {
    "original":    {"grupos": "original", "sinteticos": False},
    "synthetic":   {"grupos": "original", "sinteticos": True},
    "newfeatures": {"grupos": "original,era5_18z,bt55", "sinteticos": False},
    "all":         {"grupos": "original,era5_18z,bt55", "sinteticos": True},
    "basin":       {"grupos": "original,era5_basin", "sinteticos": False},
    "all_basin":   {"grupos": "original,era5_18z,bt55,era5_basin", "sinteticos": True},
}
# Como explicar cada grupo de variáveis para quem não é da área.
NOME_GRUPO = {
    "original": "vento, temperatura, pressão, chuva e histórico recente", "era5_18z": "estado da atmosfera às 18Z (instabilidade, cisalhamento)", "bt55": "nuvem fria vista por satélite (temperatura de brilho)", "era5_basin": "condições agregadas de toda a bacia, não só do ponto",
}
# Grupos que existem só em parte do território.
GRUPOS_REGIONAIS = ("era5_18z", "bt55")

NOME_METRICA = {
    "R2": "R²", "RMSE": "REQM", "Bias": "Viés", "Bias_P90": "Viés no extremo (P90)", "RMSE_P90": "REQM no extremo (P90)",
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
    """As três contagens de estação que os artefatos trazem, de propósito
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
    """Erro do ERA5 bruto, o ponto de partida que a correção tenta consertar.

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
                "fonte": f"{pipeline}/{arm}", "areas": int(len(df)),
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
                else f"{min(n_variaveis)}-{max(n_variaveis)}"
            ),
            "Dados sintéticos": "sim" if sinteticos else "não", "Cobertura completa": all(coberturas) if coberturas else None,
        })
    return pd.DataFrame(linhas)


@st.cache_data(show_spinner=False)
def conferencia_do_desenho() -> list[dict]:
    """Confere cada artefato publicado contra o braço que ele diz ser.

    Compara o campo `feature_groups` gravado em `run_meta.json` com o que
    `DESENHO_ABLACAO` declara para aquele braço. Existe porque a comparação
    inteira depende disso: se um braço rodou com um conjunto de variáveis
    diferente do que seu nome promete, todo delta medido contra ele aponta
    para a referência errada.

    `None` gravado é tratado como divergência, e não como ausência de
    informação: na pipeline, não especificar grupo nenhum significa usar
    **todas** as variáveis disponíveis, o oposto de uma linha de base.
    """
    divergencias = []
    for pipeline in PIPELINES:
        for arm, esperado in DESENHO_ABLACAO.items():
            caminho = ABLATION_DIR / pipeline / arm / "run_meta.json"
            if not caminho.exists():
                continue
            meta = json.loads(caminho.read_text())
            gravado = meta.get("feature_groups")
            if gravado == esperado["grupos"]:
                continue
            ativas = meta.get("active_features") or meta.get("features") or []
            divergencias.append({
                "Abordagem": NOME_PIPELINE.get(pipeline, pipeline),
                "Configuração": NOME_ARM.get(arm, arm),
                "arm": arm,
                "Declarado": esperado["grupos"],
                "Gravado": "nenhum (= todas as variáveis)" if not gravado else gravado,
                "Variáveis usadas": len(ativas),
            })
    return divergencias


@st.cache_data(show_spinner=False)
def bracos_gemeos() -> list[tuple[str, str, str]]:
    """Pares de braços que, na prática, viraram a mesma execução.

    Devolve `(abordagem, braço A, braço B)` quando os dois usaram exatamente
    a mesma lista de variáveis e o mesmo tipo de dado de treino, situação em
    que comparar um com o outro não responde a pergunta nenhuma.
    """
    gemeos = []
    for pipeline in PIPELINES:
        assinaturas: dict[str, tuple] = {}
        for arm in ARMS:
            caminho = ABLATION_DIR / pipeline / arm / "run_meta.json"
            if not caminho.exists():
                continue
            meta = json.loads(caminho.read_text())
            ativas = meta.get("active_features") or meta.get("features") or []
            if ativas:
                assinaturas[arm] = (frozenset(ativas), bool(meta.get("synthetic_csv")))
        nomes = list(assinaturas)
        for i, a in enumerate(nomes):
            for b in nomes[i + 1:]:
                if assinaturas[a] == assinaturas[b]:
                    gemeos.append((NOME_PIPELINE.get(pipeline, pipeline),
                                   NOME_ARM.get(a, a), NOME_ARM.get(b, b)))
    return gemeos


@st.cache_data(show_spinner=False)
def cobertura_parcial() -> pd.DataFrame:
    """Experimentos que cobriram só parte das áreas, com a leitura do porquê.

    Não é o mesmo que execução abortada. Alguns grupos de variáveis existem
    só em parte do território, e a pipeline tem um mecanismo documentado que
    descarta as estações sem cobertura real em vez de preencher o vazio por
    imputação, o que produz exatamente este efeito, de propósito.

    O `run_meta.json` **não registra** se esse mecanismo estava ligado, então
    a coluna "Explicação provável" descreve a hipótese, sem afirmá-la.
    """
    quadro = quadro_experimentos()
    if quadro.empty:
        return quadro
    parciais = quadro[~quadro["completo"]].copy()
    if parciais.empty:
        return parciais

    def _explica(arm: str) -> str:
        grupos = DESENHO_ABLACAO.get(arm, {}).get("grupos", "")
        regionais = [g for g in GRUPOS_REGIONAIS if g in grupos]
        if not regionais:
            return "Sem explicação nos metadados."
        return (
            "Pede variáveis que só existem em parte do território ("
            + ", ".join(NOME_GRUPO.get(g, g) for g in regionais)
            + ")."
        )

    parciais["Explicação provável"] = parciais["arm"].map(_explica)
    return parciais[["Abordagem", "Configuração", "areas", "estacoes", "Explicação provável"]]


@st.cache_data(show_spinner=False)
def estabilidade_deploy() -> pd.DataFrame:
    """R² medido em janelas mensais, e não uma vez no ano inteiro.

    A pipeline de modelos clássicos grava, para os melhores modelos de cada
    área, o R² recalculado mês a mês (`R2_deploy_mean/std/min`), que replica
    o cenário real, em que o modelo roda sobre um mês de cada vez. Um R²
    anual único esconde o mês ruim; estas colunas o mostram.

    Só existe para os modelos clássicos: é a única abordagem que grava esse
    recorte.
    """
    linhas = []
    for arm in ARMS:
        caminho = ABLATION_DIR / "lazy" / arm / "lazy_cluster_results.csv"
        if not caminho.exists():
            continue
        df = pd.read_csv(caminho)
        if "R2_deploy_mean" not in df.columns:
            continue
        df = df[df["R2_deploy_mean"].notna()]
        if df.empty:
            continue
        # Uma linha por área: o melhor modelo daquela área naquele braço.
        melhores = df.sort_values("R-Squared", ascending=False).groupby("cluster_id").head(1)
        for _, linha in melhores.iterrows():
            linhas.append({
                "arm": arm,
                "Configuração": NOME_ARM.get(arm, arm),
                "area": linha["cluster_id"],
                "Modelo": linha["Model"],
                "R2_anual": float(linha["R-Squared"]),
                "R2_mes_medio": float(linha["R2_deploy_mean"]),
                "R2_mes_desvio": float(linha["R2_deploy_std"]),
                "R2_mes_pior": float(linha["R2_deploy_min"]),
            })
    return pd.DataFrame(linhas)


# ── Leitura de quem venceu ────────────────────────────────────────────────────

DIRECAO = {"R2": "maior", "RMSE": "menor", "Bias": "zero", "Bias_P90": "zero", "RMSE_P90": "menor"}


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


@st.cache_data(show_spinner=False)
def estabilidade_por_configuracao() -> pd.DataFrame:
    """Resumo da estabilidade de cada configuração, para comparar entre elas.

    Responde uma pergunta que a visão por área não responde: as configurações
    testadas chegam a mexer na estabilidade, ou só no acerto médio?
    """
    dados = estabilidade_deploy()
    if dados.empty:
        return dados
    dados = dados.copy()
    dados["queda"] = dados["R2_mes_medio"] - dados["R2_mes_pior"]
    resumo = dados.groupby(["arm", "Configuração"], as_index=False).agg(
        areas=("area", "nunique"),
        mes_medio=("R2_mes_medio", "mean"),
        pior_mes=("R2_mes_pior", "min"),
        queda_media=("queda", "mean"),
    )
    return resumo.sort_values("mes_medio", ascending=False)


@st.cache_data(show_spinner=False)
def area_mais_fragil() -> dict | None:
    """A área que aparece como o pior mês no maior número de configurações.

    Só devolve resultado quando há repetição real: uma área que é a pior em
    uma única configuração é ruído, não padrão.
    """
    dados = estabilidade_deploy()
    if dados.empty:
        return None
    piores = (
        dados.loc[dados.groupby("arm")["R2_mes_pior"].idxmin()]["area"]
        .value_counts()
    )
    if piores.empty or int(piores.iloc[0]) < 2:
        return None
    area = piores.index[0]
    recorte = dados[dados["area"] == area]
    return {
        "area": area,
        "em_quantas": int(piores.iloc[0]),
        "de_um_total": int(dados["arm"].nunique()),
        "pior_mes": float(recorte["R2_mes_pior"].min()),
        "mes_medio": float(recorte["R2_mes_medio"].mean()),
    }


# Linhagem de produção do estudo de interpolação, os métodos que já estavam
# corrigindo o ERA5 antes deste projeto existir. Não são candidatos aqui; são
# o ponto de partida que motivou trocar de abordagem.
METODOS_EM_PRODUCAO = ("V2", "V3", "V4")
REFERENCIA_INTERPOLACAO = Path("dados_referencia_interpolacao.csv")


@st.cache_data(show_spinner=False)
def baseline_interpolacao(percentil: str = "p99") -> pd.DataFrame:
    """Desempenho dos métodos de interpolação no extremo, por validação
    deixa-uma-estação-de-fora.

    Vem do **estudo companheiro de interpolação**, não deste projeto: é a
    correção que já existia, e é contra a limitação dela que este trabalho
    foi proposto. Está aqui para que a seção 1 possa mostrar o problema com
    número medido em vez de afirmação.

    A régua é outra (percentil, conjunto de estações e desenho de validação
    diferem dos experimentos deste painel), então serve para enunciar o
    problema, nunca para montar placar contra os resultados das seções 3 e 5.
    """
    if not REFERENCIA_INTERPOLACAO.exists():
        return pd.DataFrame()
    df = pd.read_csv(REFERENCIA_INTERPOLACAO)
    df = df[df["pct"] == percentil].copy()
    if df.empty:
        return df
    df["em_producao"] = df["method"].str.startswith(METODOS_EM_PRODUCAO)
    # Encurta o rótulo sem fundir métodos diferentes: "MSP1 (F-madograma)"
    # e "MSP1 (Verossimilhanca composta)" são dois métodos, e colapsar os
    # dois em "MSP1" empilhava barras de coisas distintas na mesma linha.
    df["Método"] = (
        df["method"]
        .str.replace("Verossimilhanca", "Veross.", regex=False)
        .str.replace(r"\s*\(IDW p=2, k=15\)", "", regex=True)
        .str.replace(r"\s*\(Gaussiano sigma=2\.0, k=15\)", "", regex=True)
        .str.replace(r"\s*\(calibrado\)", "", regex=True)
        .str.strip()
    )
    return df.sort_values("bias")


@st.cache_data(show_spinner=False)
def teto_da_interpolacao(percentil: str = "p99") -> dict | None:
    """Resume o que a linhagem em produção erra no extremo, e quanto a melhor
    alternativa do mesmo estudo consegue."""
    df = baseline_interpolacao(percentil)
    if df.empty:
        return None
    producao = df[df["em_producao"]]
    alternativas = df[~df["em_producao"]]
    if producao.empty:
        return None
    # Escolhida por REQM, não por viés: viés perto de zero pode ser média de
    # erros grandes que se cancelam, e é o erro típico que interessa aqui.
    melhor_alt = (
        alternativas.loc[alternativas["rmse"].idxmin()]
        if not alternativas.empty else None
    )
    return {
        "percentil": percentil,
        "n_metodos_producao": int(len(producao)),
        "pior_vies": float(producao["bias"].min()),
        "melhor_vies_producao": float(producao["bias"].max()),
        "estacoes": int(producao["n"].iloc[0]),
        "melhor_alternativa": None if melhor_alt is None else melhor_alt["Método"],
        "vies_melhor_alternativa": None if melhor_alt is None else float(melhor_alt["bias"]),
        "reqm_melhor_alternativa": None if melhor_alt is None else float(melhor_alt["rmse"]),
        "reqm_pior_producao": float(producao["rmse"].max()),
    }


SIGLA_ARM = {
    "original": "Base", "synthetic": "Base+S", "newfeatures": "Novas", "all": "Tudo", "basin": "Bacia", "all_basin": "Tudo+B",
}
SIGLA_GRUPO = {
    "original": "base", "era5_18z": "18Z", "bt55": "satélite", "era5_basin": "bacia",
}


@st.cache_data(show_spinner=False)
def erro_era5_por_estacao(metrica: str = "P90") -> pd.DataFrame:
    """Viés do ERA5 em cada estação: o que o ERA5 diz menos o que foi medido.

    Sai com sinal, de propósito, o mapa usa escala divergente e "erra para
    menos" não pode virar a mesma cor de "erra para mais".
    """
    from engine import aggregate_station_values  # tardio: evita ciclo de import

    for pipeline, arm in combos_existentes():
        caminho = ABLATION_DIR / pipeline / arm / "predictions_by_station.csv"
        if not caminho.exists():
            continue
        try:
            era5 = aggregate_station_values(str(caminho.parent), "ERA5 raw", metrica)
            obs = aggregate_station_values(str(caminho.parent), "Observed (INMET)", metrica)
        except Exception:
            continue
        if era5.empty or obs.empty:
            continue
        junto = era5.merge(
            obs[["estacao", "value"]], on="estacao", suffixes=("_era5", "_obs"),
        )
        junto["value"] = junto["value_era5"] - junto["value_obs"]
        return junto[["estacao", "latitude", "longitude", "cluster_id", "value"]]
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def celulas_do_experimento() -> list[dict]:
    """A matriz do experimento pronta para virar grade, uma célula por braço.

    O estado de cada célula é apurado, não declarado: divergente quando os
    grupos gravados não batem com o desenho, parcial quando a execução não
    cobriu todas as áreas, ausente quando não há artefato.
    """
    quadro = quadro_experimentos()
    divergentes = {d["arm"] for d in conferencia_do_desenho()}
    parciais = set()
    if not quadro.empty:
        parciais = set(quadro[~quadro["completo"]]["arm"])
    existentes = {arm for _, arm in combos_existentes()}

    celulas = []
    for arm, ficha in DESENHO_ABLACAO.items():
        rotulos = [SIGLA_GRUPO.get(g.strip(), g.strip()) for g in ficha["grupos"].split(", ")]
        linha = " + ".join(rotulos)
        coluna = "Com extremos inventados" if ficha["sinteticos"] else "Sem extremos inventados"
        if arm not in existentes:
            estado, detalhe = "ausente", "Combinação não executada."
        elif arm in divergentes:
            estado, detalhe = (
                "divergente", "Os grupos de variáveis gravados não batem com o desenho, "
                "esta execução não é a linha de base que o nome promete.",
            )
        elif arm in parciais:
            estado, detalhe = (
                "parcial", "Cobriu só parte das áreas: pede variáveis que existem "
                "apenas em parte do território.",
            )
        else:
            estado, detalhe = "ok", "Execução íntegra."
        celulas.append({
            "linha": linha, "coluna": coluna, "estado": estado,
            "sigla": SIGLA_ARM.get(arm, arm), "detalhe": f"{NOME_ARM.get(arm, arm)}, {detalhe}",
        })

    # Célula vazia onde o desenho não tem par, para a grade mostrar a lacuna.
    linhas = {c["linha"] for c in celulas}
    colunas = {"Sem extremos inventados", "Com extremos inventados"}
    presentes = {(c["linha"], c["coluna"]) for c in celulas}
    for linha in linhas:
        for coluna in colunas:
            if (linha, coluna) not in presentes:
                celulas.append({
                    "linha": linha, "coluna": coluna, "estado": "ausente", "sigla": "", "detalhe": "Combinação não executada.",
                })
    return celulas


REFERENCIA_NACIONAL = Path("dados_referencia_inmet_era5.csv")
NIVEIS_DE_EXTREMO = {"p95": "P95", "p99": "P99", "max": "Máximo"}
EXPLICA_NIVEL = {
    "p95": "de cada 100 dias, os 5 de vento mais forte",
    "p99": "de cada 100 dias, o mais ventoso",
    "max": "a maior rajada registrada na estação em toda a série",
}


@st.cache_data(show_spinner=False)
def comparacao_nacional() -> pd.DataFrame:
    """O que cada estação do INMET mediu, contra o ponto de ERA5 mais próximo.

    Cobre o país inteiro, não só a área do experimento, porque o viés do ERA5
    não é particularidade de uma região. Uma linha por estação, com o mesmo
    dia a dia resumido em três níveis de extremo.
    """
    if not REFERENCIA_NACIONAL.exists():
        return pd.DataFrame()
    return pd.read_csv(REFERENCIA_NACIONAL)


@st.cache_data(show_spinner=False)
def vies_por_nivel() -> pd.DataFrame:
    """Para cada nível, o que o INMET mediu, o que o ERA5 estimou, e a
    diferença entre os dois.

    A diferença é subtração direta, estação por estação: valor do ERA5 menos
    valor do INMET no mesmo nível. Negativa quer dizer que o ERA5 ficou
    abaixo do que o instrumento registrou.
    """
    df = comparacao_nacional()
    if df.empty:
        return df
    linhas = []
    for chave, rotulo in NIVEIS_DE_EXTREMO.items():
        inmet, era5 = df[f"inmet_{chave}"], df[f"era5_{chave}"]
        diferenca = era5 - inmet
        linhas.append({
            "nivel": chave,
            "Nível": rotulo,
            "explicacao": EXPLICA_NIVEL[chave],
            "inmet_medio": float(inmet.mean()),
            "era5_medio": float(era5.mean()),
            "vies_medio": float(diferenca.mean()),
            "vies_mediano": float(diferenca.median()),
            "frac_subestima": float((diferenca < 0).mean()),
            "estacoes": int(len(df)),
        })
    return pd.DataFrame(linhas)
