"""Seções do painel, na ordem em que contam a história."""

from __future__ import annotations

import streamlit as st

import clusters as cl

GRAFICO = {"responsive": True}

SECOES = [
    ("area", "1 · Área de estudo", "Onde o projeto olha?"),
    ("contexto", "2 · Contexto e problema", "Por que corrigir a rajada do ERA5?"),
    ("experimento", "3 · Experimento", "O que foi feito, passo a passo?"),
    ("resultados", "4 · Resultados", "O que funcionou melhor?"),
    ("entrega", "5 · Entrega", "O que sai disso e com que ressalva?"),
]


SIGLAS = [
    ("ERA5", "Base de clima global calculada por modelo, em pontos a cada 0,25 grau, hora a hora."),
    ("INMET", "Instituto Nacional de Meteorologia. Suas estações medem a rajada de verdade."),
    ("Cluster", "Cada uma das 14 divisões da bacia do vendaval."),
    ("Rajada", "O vento mais forte registrado num curto intervalo. O painel usa o máximo de cada dia."),
    ("Viés", "Previsto menos medido, em média. Negativo quer dizer que a previsão fica abaixo."),
    ("RMSE", "Erro médio, em m/s, que dá mais peso aos erros grandes. Menor é melhor."),
    ("P95, P99", "Valores acima dos quais ficam 5% ou 1% dos dias de uma estação. Dia extremo é o que passa deles."),
    ("POD", "Fração dos dias extremos medidos que o modelo também aponta como extremos."),
    ("FAR", "Fração dos alarmes do modelo que eram falsos."),
    ("LSTM", "Rede neural que lê a sequência das horas, em vez de um instante só."),
]


def _versao(*arquivos) -> tuple:
    """Data de modificação dos arquivos. Entra como argumento das leituras em
    cache, para que trocar um arquivo invalide a tabela antiga, em vez de o
    servidor seguir servindo a leitura de antes."""
    return tuple(p.stat().st_mtime_ns for p in arquivos)


@st.cache_data(show_spinner=False)
def _bacia(versao):
    return cl.carregar()


@st.cache_data(show_spinner=False)
def _estacoes(versao):
    return cl.carregar_estacoes()


@st.cache_data(show_spinner=False)
def _celulas(versao):
    return cl.carregar_celulas()


def _lista(numeros) -> str:
    nomes = [str(n) for n in numeros]
    return ", ".join(nomes[:-1]) + " e " + nomes[-1]


def _em_construcao(titulo: str):
    st.header(titulo)
    st.info("Seção em reescrita.")


def hub(ir_para):
    st.title("Vendaval IA: corrigir a rajada extrema do ERA5")
    st.markdown(
        "O ERA5 é uma base de clima global usada no mundo todo, mas **suaviza a "
        "rajada forte**: o vendaval real passa do que ele registra. As estações "
        "do INMET medem a rajada de verdade, só que em poucos pontos. Este projeto "
        "treina modelos de aprendizado de máquina que usam as variáveis do "
        "próprio ERA5, hora a hora, para estimar a rajada máxima do dia que a "
        "estação mediria, e aplica isso à bacia do vendaval, no Sul e Sudeste do Brasil."
    )
    g = _bacia(_versao(cl.SHP))
    est = _estacoes(_versao(cl.ESTACOES))
    cel = _celulas(_versao(cl.CELULAS, cl.ESTACOES))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters na bacia", len(g))
    m2.metric("Clusters com modelo", int(cel["cluster"].nunique()))
    m3.metric("Estações do INMET no modelo", int(est["modelada"].sum()))
    m4.metric("Pontos do ERA5 no modelo", len(cel))
    st.info(
        "O painel cobre o modelo de aprendizado de máquina que usa o ERA5 hora a hora. A interpolação, "
        "uma tentativa anterior feita em outro estudo, aparece só como contexto, na seção 2."
    )
    with st.expander("Siglas e termos usados no painel"):
        st.table({"Termo": [s[0] for s in SIGLAS], "O que quer dizer": [s[1] for s in SIGLAS]})
    st.divider()
    for linha in range(0, len(SECOES), 3):
        for coluna, (chave, titulo, pergunta) in zip(st.columns(3), SECOES[linha:linha + 3]):
            with coluna.container(border=True):
                st.markdown(f"**{titulo}**")
                st.caption(pergunta)
                st.button("Abrir", key=f"ir_{chave}", on_click=ir_para,
                          args=(chave,), width="stretch")


def contexto():
    import contexto as ctx
    ctx.render()


def area():
    st.header("1 · Área de estudo")
    st.markdown(
        "A **bacia do vendaval** cobre o Sul e o Sudeste do Brasil e está dividida em "
        "**14 clusters**, as divisões da própria bacia."
    )

    g = _bacia(_versao(cl.SHP))
    est = _estacoes(_versao(cl.ESTACOES))
    cel = _celulas(_versao(cl.CELULAS, cl.ESTACOES))

    st.plotly_chart(cl.mapa_da_bacia(g, est), width="stretch", config=cl.GRAFICO_MAPA,
                    key="mapa_bacia")

    st.markdown("#### Estações do INMET por cluster")
    st.plotly_chart(cl.mapa_das_estacoes(g, est), width="stretch", config=cl.GRAFICO_MAPA,
                    key="mapa_estacoes")

    st.markdown("#### Pontos do ERA5 e estações do INMET")
    st.markdown(
        "O ERA5 é uma grade de pontos a cada 0,25 grau, cerca de 25 km. Só nos pontos que "
        "têm uma estação do INMET dentro dá para comparar o ERA5 com a rajada medida. "
        "A grade está desenhada para os clusters do estudo: "
        f"{_lista(sorted(cel['cluster'].unique()))}."
    )
    opcoes = cl.vistas(cel)
    escolha = st.radio("Aproximar", list(opcoes), horizontal=True, key="vista_grade")
    st.plotly_chart(cl.mapa_da_grade(g, cel, est, opcoes[escolha]), width="stretch",
                    config=cl.GRAFICO_MAPA, key="mapa_grade")


def experimento():
    import metodo
    metodo.render()


def resultados():
    import resultados as res
    res.render()


def entrega(ir_para):
    import entrega as ent
    ent.render()
