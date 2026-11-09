"""
Painel de resultados do Vendaval IA, correção de viés da rajada extrema do ERA5.

`app.py` é a casca: configuração da página, navegação e barra lateral.
`secoes.py` tem as seções, na ordem em que contam a história.

A navegação é estado (`st.radio` na barra lateral), não `st.tabs`: aba não
pode ser selecionada por código, e os botões da página inicial precisam levar
a algum lugar.
"""

from __future__ import annotations

import streamlit as st

# `set_page_config` tem de ser o primeiro comando Streamlit do processo.
st.set_page_config(
    page_title="Vendaval IA",
    layout="wide",
    initial_sidebar_state="expanded",
)

import secoes  # noqa: E402

# O texto dos avisos usa a cor de tinta do tema, sempre legível, e o corpo do
# texto é maior para quem olha de longe.
st.markdown(
    """
    <style>
    [data-testid^="stAlertContent"] [data-testid="stMarkdownContainer"],
    [data-testid^="stAlertContent"] [data-testid="stMarkdownContainer"] * {
        color: #f2f0eb !important;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid^="stAlertContent"] { font-size: 1.15rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGINAS = [("hub", "Início")] + [(chave, titulo) for chave, titulo, _ in secoes.SECOES]
ROTULOS = dict(PAGINAS)
CHAVES = [chave for chave, _ in PAGINAS]

if "navegacao" not in st.session_state:
    st.session_state["navegacao"] = "hub"


def ir_para(destino: str) -> None:
    """Navega a partir de um botão. Roda como `on_click`, nunca no corpo do
    script: escrever em `st.session_state` de um widget já instanciado neste
    ciclo levanta exceção, e o rádio da barra lateral nasce antes dos botões."""
    st.session_state["navegacao"] = destino


with st.sidebar:
    st.title("Vendaval IA")
    st.caption("Correção da rajada extrema do ERA5 na bacia do vendaval")
    st.divider()
    st.radio(
        "Seções", CHAVES, format_func=lambda chave: ROTULOS[chave],
        key="navegacao", label_visibility="collapsed",
    )

pagina = st.session_state["navegacao"]
if pagina == "hub":
    secoes.hub(ir_para)
elif pagina == "contexto":
    secoes.contexto()
elif pagina == "area":
    secoes.area()
elif pagina == "experimento":
    secoes.experimento()
elif pagina == "resultados":
    secoes.resultados()
elif pagina == "entrega":
    secoes.entrega(ir_para)

# Navegação de rodapé, para não obrigar a voltar à barra lateral.
if pagina != "hub":
    st.divider()
    indice = CHAVES.index(pagina)
    anterior, proxima = st.columns(2)
    if indice > 0:
        anterior.button(f"← {ROTULOS[CHAVES[indice - 1]]}", key="nav_anterior",
                        on_click=ir_para, args=(CHAVES[indice - 1],), width="stretch")
    if indice < len(CHAVES) - 1:
        proxima.button(f"{ROTULOS[CHAVES[indice + 1]]} →", key="nav_proxima",
                       on_click=ir_para, args=(CHAVES[indice + 1],), width="stretch")
