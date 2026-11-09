"""Seção 1, o contexto: o problema, a primeira tentativa e a hipótese.

Fontes:
- `dataset/vies_era5.csv`: quanto o ERA5 erra, em média, nos dias de teste e nos
  dias extremos (rajada medida acima do P95 ou do P99 da estação, limiar dos
  dias de treino). Calculado dos arquivos de previsão dos modelos.
- `dados_referencia_interpolacao.csv`: validação cruzada dos métodos de
  interpolação, com 615 estações do país. É um estudo à parte, do projeto
  Vendaval, e não do Vendaval IA. As definições dos métodos vêm da documentação
  desse estudo (`docs/research/interpolacoes_extremos/`, documentos 000, 001, 003 e 005).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from metodo import _virgula

TINTA = "#f2f0eb"
TINTA_FRACA = "#b4b1a9"
FUNDO = "#17181a"
GRAFICO = {"responsive": True, "displaylogo": False}
VIES = "dataset/vies_era5.csv"
INTERPOLACAO = "dados_referencia_interpolacao.csv"

# método (nome no arquivo): (nome na tela, família, o que faz em uma frase)
METODOS = {
    "V2 (IDW p=2, k=15)": (
        "V2", "media",
        "Média das 15 estações mais próximas, com peso maior para as mais perto."),
    "V3 (Gaussiano sigma=2.0, k=15)": (
        "V3", "media",
        "A mesma média, com o peso em forma de sino e um alcance de cerca de 200 km."),
    "V4 (Brown-Resnick calibrado)": (
        "V4", "media",
        "A mesma média, com o alcance medido nos dados, cerca de 10 km."),
    "MSP1 (F-madograma)": (
        "MSP1, F-madograma", "outro",
        "Processo max-estável, um modelo de como os extremos de lugares diferentes se relacionam. "
        "Na versão aproximada, que olha duas estações por vez."),
    "MSP1 (Verossimilhanca composta)": (
        "MSP1, verossimilhança", "outro",
        "O mesmo processo, com a dependência entre as estações estimada de outro jeito."),
    "EVK": (
        "EVK", "outro",
        "Krigagem de valores extremos: ajusta uma distribuição de extremos em cada estação e "
        "espalha os parâmetros pelo mapa."),
    "Quantile Kriging": (
        "Quantile Kriging", "outro",
        "Krigagem do percentil alto, P95 ou P99, em vez da média."),
    "RBF Multiquadric": (
        "RBF multiquadrática", "outro",
        "Interpolação exata por funções da distância, sem base estatística."),
    "IDW Adaptativo": (
        "IDW adaptativo", "outro",
        "A média pelo inverso da distância, com o expoente ajustado a cada região."),
}
COR_FAMILIA = {"media": "#b8932f", "outro": "#3d7ea6"}
NOME_FAMILIA = {"media": "Média ponderada das vizinhas (V2, V3 e V4)", "outro": "Outros métodos"}


def figura_do_vies(vies: pd.DataFrame) -> go.Figure:
    """Rajada do ERA5 menos a medida, em média, por cluster e por recorte."""
    recortes = [("todos", "Todos os dias", "#9aa3ad"), ("P95", "Dias extremos (P95)", "#3d7ea6"),
                ("P99", "Dias extremos (P99)", "#b8932f")]
    cl_ids = sorted(vies["cluster"].unique())
    figura = go.Figure()
    for chave, nome, cor in recortes:
        s = vies[vies["recorte"] == chave].set_index("cluster").loc[cl_ids]
        figura.add_trace(go.Bar(
            x=[f"Cluster {c}" for c in cl_ids], y=s["vies_era5"], name=nome, marker_color=cor,
            text=[_virgula(v, 1) for v in s["vies_era5"]], textposition="outside",
            textfont=dict(color=TINTA),
            hovertemplate="%{x}<br>ERA5 menos o INMET: %{y:.1f} m/s<extra>" + nome + "</extra>"))
    figura.add_hline(y=0, line=dict(color="#9aa3ad", width=1))
    figura.update_layout(
        barmode="group", height=400, margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=TINTA)),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        xaxis=dict(tickfont=dict(color=TINTA), showgrid=False),
        yaxis=dict(title="ERA5 menos INMET (m/s)", gridcolor="#26292d",
                   tickfont=dict(color=TINTA_FRACA), automargin=True))
    return figura


def figura_dos_metodos(interp: pd.DataFrame, nivel: str, metrica: str) -> go.Figure:
    """O viés ou o erro de cada método de interpolação, na validação cruzada."""
    sub = interp[interp["pct"] == nivel.lower()].set_index("method")
    ordem = [m for m in METODOS if m in sub.index]
    valores = [float(sub.loc[m, metrica]) for m in ordem]
    figura = go.Figure()
    for fam in ("media", "outro"):
        idx = [i for i, m in enumerate(ordem) if METODOS[m][1] == fam]
        figura.add_trace(go.Bar(
            x=[valores[i] for i in idx], y=[METODOS[ordem[i]][0] for i in idx], orientation="h",
            name=NOME_FAMILIA[fam], marker_color=COR_FAMILIA[fam],
            text=[_virgula(valores[i], 1) for i in idx], textposition="outside",
            textfont=dict(color=TINTA), cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.1f} m/s<extra></extra>"))
    figura.add_vline(x=0, line=dict(color="#9aa3ad", width=1))
    titulo = "Viés no extremo (m/s), negativo é subestimar" if metrica == "bias" else "Erro (RMSE, m/s), menor é melhor"
    figura.update_layout(
        height=430, margin=dict(l=10, r=40, t=10, b=40), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=TINTA)),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        xaxis=dict(title=titulo, gridcolor="#26292d", tickfont=dict(color=TINTA_FRACA), zeroline=False),
        yaxis=dict(tickfont=dict(color=TINTA), showgrid=False, automargin=True, autorange="reversed"))
    return figura


def render():
    vies = pd.read_csv(VIES)
    interp = pd.read_csv(INTERPOLACAO)

    st.header("1 · Contexto e problema")

    st.markdown("#### O problema")
    p99 = vies[vies["recorte"] == "P99"]
    todos = vies[vies["recorte"] == "todos"]
    st.markdown(
        "O ERA5 é uma base de clima calculada, em pontos a cada 25 km, sem instrumento no lugar. "
        "As estações do INMET medem a rajada de verdade, mas em poucos pontos. Na média dos dias o "
        f"ERA5 quase não tem viés, de {_virgula(todos['vies_era5'].abs().min(), 1)} a "
        f"{_virgula(todos['vies_era5'].abs().max(), 1)} m/s. O problema está no **extremo**: nos dias de "
        "rajada muito forte, ele fica bem abaixo do medido."
    )
    st.plotly_chart(figura_do_vies(vies), width="stretch", config=GRAFICO, key="ctx_vies")
    st.markdown(
        f"Nos dias extremos pelo P99, a rajada do ERA5 fica de {_virgula(abs(p99['vies_era5']).min(), 1)} a "
        f"{_virgula(abs(p99['vies_era5']).max(), 1)} m/s abaixo da medida, conforme o cluster."
    )
    st.caption(
        "Cada barra é a rajada máxima do dia do ERA5 menos a medida pelo INMET, na média, nos meses "
        "de teste. Negativo quer dizer que o ERA5 fica abaixo. Dia extremo é o dia em que a rajada medida "
        "passa do P95 ou do P99 da própria estação, calculado nos dias de treino. Parte da diferença vem "
        "de escolher os dias pelo valor medido: o ERA5 e o INMET nunca coincidem, e pegar os maiores "
        "medidos destaca os casos em que o ERA5 ficou abaixo."
    )

    st.markdown("#### A primeira tentativa: interpolar")
    st.info(
        "Esta parte é um estudo à parte, do projeto Vendaval, feito antes. Não é o Vendaval IA. "
        "Entra aqui para mostrar o que já se tentou e onde parou."
    )
    st.markdown(
        "Já existia uma correção por interpolação. Mede-se o erro do ERA5 em cada estação, espalha-se "
        "esse erro pelo mapa e soma-se ao ERA5. As versões **V2, V3 e V4** mudam só o jeito de "
        "espalhar, e todas fazem uma **média ponderada das estações vizinhas**. Uma média nunca passa "
        "do maior valor que entrou na conta: se só uma estação mediu o vendaval, as vizinhas, que não "
        "o viram, diluem o valor dela."
    )
    st.markdown("Para ver se outros métodos escapam desse limite, foram testados mais seis:")
    st.table({"Método": [v[0] for v in METODOS.values()], "O que faz": [v[2] for v in METODOS.values()]})
    c1, c2 = st.columns(2)
    nivel = c1.radio("Extremo", ["P99", "P95"], horizontal=True, key="ctx_nivel")
    metrica = c2.radio("Mostrar", ["Viés", "Erro (RMSE)"], horizontal=True, key="ctx_metrica")
    chave = "bias" if metrica == "Viés" else "rmse"
    st.plotly_chart(figura_dos_metodos(interp, nivel, chave), width="stretch", config=GRAFICO,
                    key="ctx_metodos")
    sub = interp[interp["pct"] == nivel.lower()].set_index("method")
    media = [m for m in METODOS if METODOS[m][1] == "media"]
    outros = [m for m in METODOS if METODOS[m][1] == "outro"]
    melhor = min(outros, key=lambda m: sub.loc[m, "rmse"])
    n = int(sub["n"].iloc[0])
    st.markdown(
        f"As médias ponderadas (V2, V3 e V4) ficam de {_virgula(abs(sub.loc[media, 'bias']).min(), 1)} a "
        f"{_virgula(abs(sub.loc[media, 'bias']).max(), 1)} m/s abaixo do medido no {nivel}: o limite da "
        "média, na prática. Os outros métodos reduzem o viés, e alguns o zeram, mas o erro continua alto: o melhor, "
        f"o {METODOS[melhor][0]}, erra {_virgula(float(sub.loc[melhor, 'rmse']), 1)} m/s. Nenhum resolve."
    )
    st.caption(
        f"Validação cruzada: tira-se uma estação, prevê-se o valor dela com as outras e compara-se. Foram "
        f"{n} estações do país. Os números são deste estudo, com régua própria, e não se comparam com os "
        "erros da seção Resultados."
    )

    st.markdown("#### Por que um modelo de aprendizado de máquina pode apresentar melhores resultados?")
    st.markdown(
        "A interpolação só enxerga as estações vizinhas. Se nenhuma viu o vendaval, ele não aparece no "
        "mapa. O modelo olha o **estado da atmosfera no próprio ponto**, hora a hora: instabilidade, "
        "vento em altitude, pressão. Nos pontos com estação, ele aprende a levar a rajada do ERA5 até a "
        "medida. Depois, vale para todos os pontos do cluster, tenham estação ou não."
    )
    st.info(
        "**A pergunta do projeto:** um modelo que aprende das variáveis atmosféricas do próprio ponto "
        "captura o vendaval que a interpolação não captura? É uma hipótese, e as seções seguintes a testam."
    )
