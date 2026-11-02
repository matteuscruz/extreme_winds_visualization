"""
Grade v1.2, previsões de cada modelo sobre a malha do ERA5.

Compartilhado pela Seção 6 e pelo Explorador. Mostra o que os arquivos
trazem e a regra que vale para ler cada dia: dias que o modelo viu no
treino ou na validação não provam previsão.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

import engine as en

GRAFICO = {"responsive": True}
NIVEIS = ["P90", "P95", "P99", "Máximo"]
DIA_PADRAO = pd.Timestamp("2025-03-10")


def _enquadramento(regiao: str) -> tuple[tuple[float, float], float]:
    """Centro e zoom do mapa a partir da extensão da malha da região."""
    ds = en.load_grade_v12(regiao)
    lat0, lat1 = float(ds.lat.min()), float(ds.lat.max())
    lon0, lon1 = float(ds.lon.min()), float(ds.lon.max())
    folga = max(lat1 - lat0, lon1 - lon0) * 1.35
    return ((lat0 + lat1) / 2, (lon0 + lon1) / 2), round(math.log2(492 / folga), 1)


def _estacoes_da_regiao(regiao: str) -> pd.DataFrame:
    """Estações das áreas que a região cobre e cujo ponto cai numa célula com
    dado. A caixa da malha também pega estações de outras áreas, sem previsão."""
    ds = en.load_grade_v12(regiao)
    areas = np.unique(ds["cluster_id"].values)
    df = en.stations_geo_df
    df = df[df["cluster_id"].isin(areas[areas >= 0])]
    com_dado = [
        int(ds["cluster_id"].sel(lat=r.latitude, lon=r.longitude, method="nearest")) >= 0
        for r in df.itertuples()
    ]
    return df[com_dado]


def _escala(*tabelas: pd.DataFrame) -> float:
    pico = max((float(t[c].max()) for t in tabelas if not t.empty
                for c in ("ws_original", "rajada_max_corrigida")), default=en.GRID_VMAX)
    return max(en.GRID_VMAX, math.ceil(pico))


def _par_de_mapas(tabela: pd.DataFrame, regiao: str, rotulo: str, chave: str,
                  estacao: str | None = None):
    centro, zoom = _enquadramento(regiao)
    estacoes = _estacoes_da_regiao(regiao)
    vmax = _escala(tabela)
    esquerda, direita = st.columns(2)
    with esquerda:
        st.plotly_chart(
            en.build_grid_map(tabela, "ws_original", estacoes, estacao, "ERA5",
                              show_colorbar=False, vmax=vmax, centro=centro, zoom=zoom),
            width="stretch", config=GRAFICO, key=f"{chave}_era5",
        )
    with direita:
        st.plotly_chart(
            en.build_grid_map(tabela, "rajada_max_corrigida", estacoes, estacao, rotulo,
                              vmax=vmax, centro=centro, zoom=zoom),
            width="stretch", config=GRAFICO, key=f"{chave}_modelo",
        )


def desenhar(prefixo: str, com_estacao: bool = False):
    regioes = en.grade_v12_disponivel()
    if not regioes:
        st.info("A grade v1.2 não está publicada neste painel.")
        return

    esquerda, direita = st.columns(2)
    regiao = esquerda.selectbox("Região", list(regioes), key=f"{prefixo}_regiao")
    modelo = direita.selectbox(
        "Modelo", list(en.GRADE_V12_MODELOS), format_func=en.GRADE_V12_MODELOS.get,
        key=f"{prefixo}_modelo",
    )
    rotulo = en.GRADE_V12_MODELOS[modelo]

    limites = en.grade_v12_limites(regiao)
    if limites is None:
        st.info("Sem dado para esta região.")
        return

    dia = st.date_input(
        "Dia", value=min(max(DIA_PADRAO, limites[0]), limites[1]),
        min_value=limites[0].date(), max_value=limites[1].date(), key=f"{prefixo}_dia",
    )
    tabela = en.grade_v12_snapshot(str(dia), regiao, modelo)
    if tabela.empty:
        st.info("Sem valor neste dia.")
        return

    estacao = None
    if com_estacao:
        nomes = _estacoes_da_regiao(regiao)["estacao"].tolist()
        escolha = st.selectbox("Estação", [en.NENHUMA] + nomes, key=f"{prefixo}_estacao")
        estacao = None if escolha == en.NENHUMA else escolha

    _par_de_mapas(tabela, regiao, rotulo, f"{prefixo}_dia_mapa", estacao)
    if en.grade_v12_dia_fora_da_amostra(str(dia), regiao):
        st.caption(f"{dia} · mesma escala de cor nos dois mapas. Dia fora da amostra: "
                   "nenhum modelo o viu no treino nem na validação.")
    else:
        st.warning(
            f"{dia} está dentro da amostra: nas células com estação, os modelos viram "
            "este dia no treino ou na validação. O acerto aqui é otimista. "
            "Fora da amostra: janeiro, abril, julho e outubro de 2000 a 2024, e todo 2025."
        )

    if estacao is not None:
        linha = en.stations_geo_df[en.stations_geo_df["estacao"] == estacao].iloc[0]
        serie = en.grade_v12_serie_ponto(
            float(linha["latitude"]), float(linha["longitude"]), regiao, modelo)
        st.plotly_chart(
            en.build_grid_station_timeseries(
                estacao, serie, en.load_inmet_station_series(estacao), dia.year),
            width="stretch", config=GRAFICO, key=f"{prefixo}_serie",
        )
        no_ano = serie[serie["time"].dt.year == dia.year]
        if not no_ano.empty:
            st.caption(f"{dia.year}: {no_ano['fora_da_amostra'].mean():.0%} dos dias "
                       "fora da amostra.")

    st.divider()
    nivel = st.radio("Nível", NIVEIS, horizontal=True, key=f"{prefixo}_nivel")
    niveis = en.grade_v12_nivel(regiao, modelo, nivel)
    if niveis.empty:
        st.info("Sem dado fora da amostra para montar este nível.")
        return
    _par_de_mapas(niveis, regiao, rotulo, f"{prefixo}_nivel_mapa")
    n_fora = int(en.grade_v12_fora_da_amostra(en.load_grade_v12(regiao).time.values).sum())
    st.caption(f"{nivel} de cada célula, calculado só nos {n_fora} dias fora da amostra, "
               "mesma escala de cor nos dois mapas.")


def panorama() -> dict:
    """Números da grade v1.2 lidos dos arquivos: período, dias, horas e áreas."""
    areas, dias, inicio, fim = set(), 0, None, None
    for regiao in en.grade_v12_disponivel():
        ds = en.load_grade_v12(regiao)
        ids = np.unique(ds["cluster_id"].values)
        areas |= {int(i) for i in ids if i >= 0}
        limites = en.grade_v12_limites(regiao)
        dias = max(dias, len(ds.time))
        inicio = limites[0] if inicio is None else min(inicio, limites[0])
        fim = limites[1] if fim is None else max(fim, limites[1])
    return {
        "areas": sorted(areas), "dias": dias, "horas": dias * 24,
        "inicio": inicio, "fim": fim,
    }


def celulas() -> pd.DataFrame:
    """Células com previsão de todas as regiões, com a área de cada uma."""
    partes = []
    for regiao in en.grade_v12_disponivel():
        ids = en.load_grade_v12(regiao)["cluster_id"]
        df = ids.to_dataframe().reset_index()
        df = df[df["cluster_id"] >= 0].rename(
            columns={"lat": "latitude", "lon": "longitude", "cluster_id": "area"})
        partes.append(df.assign(regiao=regiao)[["regiao", "area", "latitude", "longitude"]])
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(
        columns=["regiao", "area", "latitude", "longitude"])


def ficha_das_regioes() -> pd.DataFrame:
    """Uma linha por região: grade, células com previsão e estações."""
    linhas = []
    cel = celulas()
    for regiao in en.grade_v12_disponivel():
        ds = en.load_grade_v12(regiao)
        areas = sorted(cel.loc[cel["regiao"] == regiao, "area"].unique())
        linhas.append({
            "Região": regiao,
            "Grade": f"{ds.sizes['lat']} × {ds.sizes['lon']}",
            "Células com previsão": int((cel["regiao"] == regiao).sum()),
            "Estações": len(_estacoes_da_regiao(regiao)),
            "Áreas": ", ".join(str(a) for a in areas),
        })
    return pd.DataFrame(linhas).set_index("Região")
