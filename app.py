"""
Painel de resultados, correção de viés de rajada de vento extremo.

A tela é montada em quatro camadas, cada uma num arquivo:

- `app.py` (aqui): a casca, configuração da página, navegação e barra lateral.
- `narrativa.py`: as cinco seções que contam a história, para quem chega sem
  contexto nenhum.
- `explorador.py`: o painel de consulta técnica, as quatro telas originais,
  preservadas, para quem é do projeto.
- `engine.py` / `apuracao.py`: o motor. Carregam os artefatos, calculam as
  métricas e devolvem as figuras. Nenhum número da tela é digitado à mão.

A navegação é estado (`st.radio` na barra lateral), não `st.tabs`: aba não
pode ser selecionada por código, e os botões da página inicial precisam levar
a algum lugar.
"""

from __future__ import annotations

import streamlit as st

# `set_page_config` tem de ser o primeiro comando Streamlit do processo, e
# importar o motor já dispara carregamento de dado em cache, por isso a
# configuração vem antes dos imports do projeto, fora da ordem habitual.
st.set_page_config(
    page_title="Rajada extrema, correção de viés",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import apuracao as ap  # noqa: E402
import explorador  # noqa: E402
import narrativa  # noqa: E402
from engine import (  # noqa: E402
    ABLATION_ARMS,
    ABLATION_PIPELINES,
    ABLATION_SEASONS_ORDER,
    PERIODOS,
    PIPELINE_LABELS,
    load_ablation_results,
    ablation_combos,
    stations_geo_df,
    CLUSTER_IDS_ALL,
)

# Esconde o controle de atribuição (texto "© CARTO, © OpenStreetMap
# contributors" + botão "ⓘ") que a MapLibre/Mapbox GL desenha sobre os
# mapas, a pedido explícito, ciente de que isso normalmente vai contra os
# termos de uso desses provedores de tile gratuitos.
st.markdown(
    """
    <style>
    .maplibregl-ctrl-attrib.mapboxgl-ctrl-attrib {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Navegação ─────────────────────────────────────────────────────────────────

PAGINAS = (
    [("hub", "Início")]
    + [(chave, titulo) for chave, titulo, _ in narrativa.SECOES]
    + [("explorador", "Explorador (consulta técnica)")]
)
ROTULOS = dict(PAGINAS)
CHAVES = [chave for chave, _ in PAGINAS]

if "navegacao" not in st.session_state:
    st.session_state["navegacao"] = "hub"


def ir_para(destino: str) -> None:
    """Navega a partir de um botão.

    Precisa rodar como `on_click`, nunca no corpo do script: escrever em
    `st.session_state` de um widget que já foi instanciado neste ciclo
    levanta exceção, e o rádio da barra lateral é criado antes dos botões.
    O callback roda antes do ciclo seguinte, quando nenhum widget existe
    ainda.
    """
    st.session_state["navegacao"] = destino


# O clique no mapa do Explorador não pode escrever direto em
# st.session_state["global_station"] no mesmo ciclo em que o widget da barra
# lateral já foi instanciado. O clique grava um valor pendente, processado
# aqui, antes de o widget existir.
if "_pending_station" in st.session_state:
    st.session_state["global_station"] = st.session_state.pop("_pending_station")

for chave, padrao in (
    ("global_pipelines", ["mlp"]),
    ("global_arms", ["original"]),
    ("global_station", None),
):
    st.session_state.setdefault(chave, padrao)


# ── Barra lateral ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Rajada extrema")
    st.caption("Correção de viés do ERA5 no Sul do Brasil")
    st.divider()

    st.radio(
        "Seções", CHAVES,
        format_func=lambda chave: ROTULOS[chave],
        key="navegacao",
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("Onde ficam os dados e o código"):
        st.markdown(
            "**Dados brutos e artefatos dos experimentos:** máquina 3 do "
            "cluster de pesquisa, em `/home/publico/vendaval_ai`.\n\n"
            "**Este painel** carrega apenas resultado já calculado, nenhum "
            "modelo é treinado aqui."
        )
        st.caption(
            f"Treino {PERIODOS['treino'][0]} a {PERIODOS['treino'][1]} · "
            f"validação {PERIODOS['validacao'][0]} a {PERIODOS['validacao'][1]} · "
            f"teste {PERIODOS['teste'][0]} a {PERIODOS['teste'][1]}."
        )

    # Os controles de recorte só existem para o Explorador, não fazem sentido
    # para quem está lendo a narrativa e atrapalhariam a leitura.
    if st.session_state["navegacao"] == "explorador":
        st.divider()
        st.caption("**Recorte da consulta**")
        st.multiselect(
            "Abordagem", ABLATION_PIPELINES,
            format_func=lambda p: PIPELINE_LABELS.get(p, p),
            key="global_pipelines",
        )
        st.multiselect(
            "Configuração", ABLATION_ARMS,
            format_func=lambda a: ap.NOME_ARM.get(a, a),
            key="global_arms",
        )

        escolhidos = [
            c for c in ablation_combos
            if c["pipeline"] in st.session_state["global_pipelines"]
            and c["arm"] in st.session_state["global_arms"]
        ]
        if len(escolhidos) > 4:
            st.warning("Muitas combinações deixam os gráficos sobrepostos ilegíveis.")
        elif not escolhidos:
            st.info("Escolha ao menos uma abordagem e uma configuração.", icon="ℹ️")

        areas_disponiveis: set = set()
        for combo in escolhidos:
            resultados = load_ablation_results(combo["dir"])
            if not resultados.empty:
                areas_disponiveis |= set(resultados["cluster_id"].unique().tolist())
        opcoes_area = sorted(areas_disponiveis, key=str) or CLUSTER_IDS_ALL

        if opcoes_area:
            st.selectbox(
                "Área", opcoes_area,
                format_func=lambda c: f"Área {c}",
                key="global_cluster",
            )
        st.selectbox("Trimestre", ABLATION_SEASONS_ORDER, key="global_season")

        area_atual = st.session_state.get("global_cluster")
        estacoes = (
            stations_geo_df[stations_geo_df["cluster_id"] == area_atual]["estacao"].tolist()
            if area_atual is not None and not stations_geo_df.empty else []
        )
        opcoes_estacao = [explorador.NENHUMA] + estacoes
        if st.session_state["global_station"] not in opcoes_estacao:
            st.session_state["global_station"] = explorador.NENHUMA
        st.selectbox("Estação", opcoes_estacao, key="global_station")


# ── Corpo ─────────────────────────────────────────────────────────────────────

pagina = st.session_state["navegacao"]

if pagina == "hub":
    narrativa.hub(ir_para)
elif pagina == "problema":
    narrativa.problema()
elif pagina == "experimento":
    narrativa.experimento()
elif pagina == "resultado":
    narrativa.resultado()
elif pagina == "prova":
    narrativa.prova()
elif pagina == "estabilidade":
    narrativa.estabilidade()
elif pagina == "entrega":
    narrativa.entrega(ir_para)
elif pagina == "explorador":
    st.title("Explorador")
    st.caption(
        "As quatro telas de consulta do projeto. Use o recorte na barra "
        "lateral para escolher o que comparar."
    )
    TELAS = {
        "Comparação geral": explorador._render_tab_global,
        "Erro no espaço e no tempo": explorador._render_tab_inspector,
        "Diagnóstico dos modelos": explorador._render_tab_diag,
        "Mapa corrigido": explorador._render_tab_grid,
    }
    # Rádio em vez de `st.tabs`: o Streamlit monta aba escondida sem largura
    # utilizável, e figura que depende da largura para se dimensionar nasce
    # encolhida ali dentro.
    tela = st.radio(
        "Tela", list(TELAS),
        horizontal=True, key="tela_explorador", label_visibility="collapsed",
    )
    st.divider()
    TELAS[tela]()

# Navegação de rodapé, para não obrigar a voltar à barra lateral.
if pagina != "hub":
    st.divider()
    indice = CHAVES.index(pagina)
    anterior, proxima = st.columns(2)
    if indice > 0:
        anterior.button(
            f"← {ROTULOS[CHAVES[indice - 1]]}", key="nav_anterior",
            on_click=ir_para, args=(CHAVES[indice - 1],), width="stretch",
        )
    if indice < len(CHAVES) - 1:
        proxima.button(
            f"{ROTULOS[CHAVES[indice + 1]]} →", key="nav_proxima",
            on_click=ir_para, args=(CHAVES[indice + 1],), width="stretch",
        )
