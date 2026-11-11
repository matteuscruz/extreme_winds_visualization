"""Seção 5, a entrega: o que sai do projeto, como abrir e com que ressalva.

Fontes: a documentação do pipeline, para o conteúdo dos arquivos e os limites
do estudo; a auditoria do projeto; e os números calculados nas seções 1 e 4
deste painel. Tudo o que é número
vem dos NetCDFs ou dos CSVs, nada é digitado aqui.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import clusters as cl
from metodo import ERRO_POR_CLUSTER, NOMES_MODELO, _milhar, _virgula
from resultados import (CLUSTERS_NC, CONJUNTO, DETECCAO, FUNDO, GRAFICO, PASTA_NC, TINTA,
                        TINTA_FRACA, _abrir, _nc_disponivel, _variacao, _versao_nc)

COR_MODELO = "#e8833a"
MODELOS = [m for m in NOMES_MODELO if m != "era5"]

# O que cada variável do arquivo guarda. Duas famílias: a grade e as estações.
VARIAVEIS = [
    ("Grade do ERA5", "time, lat, lon", "era5",
     "Rajada do ERA5 sem correção, na hora do pico do dia"),
    ("Grade do ERA5", "time, lat, lon", "ml_local, ml_global_*, lstm_local, lstm_global_anual",
     "Rajada máxima do dia prevista por cinco modelos corrigidos"),
    ("Grade do ERA5", "lat, lon", "celula_no_cluster",
     "1 se o ponto da grade está dentro do cluster, 0 se está fora (vazio)"),
    ("Estações", "estacao, time", "est_inmet",
     "Rajada máxima do dia medida pelo INMET, a referência"),
    ("Estações", "estacao, time", "est_era5 e est_*",
     "O mesmo do ERA5 e dos seis modelos corrigidos, na posição de cada estação"),
    ("Estações", "estacao", "estacao_lat, estacao_lon",
     "Posição de cada estação"),
    ("Dia", "time", "conjunto",
     "Para que o dia serviu: treino, validação, teste, 2025 ou sem dado"),
]

CODIGO_ERRO = '''import xarray as xr

ds = xr.open_dataset("predicoes_rajada_cluster3.nc")
teste = ds["conjunto"] == 2          # 0 treino, 1 validação, 2 teste, 3 = 2025
erro = (ds["est_lstm_global_anual"] - ds["est_inmet"]).where(teste)
print(float((erro ** 2).mean() ** 0.5))   # erro (RMSE) em m/s'''

CODIGO_ESTACAO = '''import xarray as xr

ds = xr.open_dataset("predicoes_rajada_cluster3.nc")

serie = ds.sel(estacao="A801")[["est_inmet", "est_era5", "est_lstm_global_anual"]]
serie = serie.where(ds["conjunto"] >= 2)       # só dias que o modelo não viu
serie.to_dataframe().dropna().head()'''

CODIGO_PONTO = '''import xarray as xr

ds = xr.open_dataset("predicoes_rajada_cluster3.nc")

# o ponto da grade mais próximo de uma coordenada qualquer
ponto = ds.sel(lat=-30.0, lon=-51.2, method="nearest")
rajada_ms = ponto["lstm_global_anual"].sel(time="2025-07-14")
print(float(rajada_ms) * 3.6, "km/h")'''


# ── Dados ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _arquivos(versao) -> pd.DataFrame:
    """Uma linha por arquivo: tamanho, grade, pontos, estações e dias de cada conjunto."""
    linhas = []
    for c in CLUSTERS_NC:
        ds = _abrir(c, versao)
        conj = ds["conjunto"].values
        linha = {
            "cluster": c,
            "tamanho": (PASTA_NC / f"predicoes_rajada_cluster{c}.nc").stat().st_size / 1e6,
            "grade": f"{ds.sizes['lat']} × {ds.sizes['lon']}",
            "pontos": int(ds["celula_no_cluster"].values.sum()),
            "estacoes": int(ds.sizes["estacao"]),
        }
        for k, nome in CONJUNTO.items():
            linha[nome] = int((conj == k).sum())
        linha["sem dado"] = int((conj == -1).sum())
        linhas.append(linha)
    return pd.DataFrame(linhas)


@st.cache_data(show_spinner=False)
def _estacoes_do_cluster(cluster: int, versao) -> list[str]:
    ds = _abrir(cluster, versao)
    vistas = np.isfinite(ds["est_inmet"].values)[:, ds["conjunto"].values >= 2].any(axis=1)
    return [str(e) for e, ok in zip(ds["estacao"].values, vistas) if ok]


@st.cache_data(show_spinner=False)
def _serie(cluster: int, estacao: str, versao) -> pd.DataFrame:
    ds = _abrir(cluster, versao).sel(estacao=estacao)
    quadro = pd.DataFrame({"inmet": ds["est_inmet"].values, "era5": ds["est_era5"].values,
                           **{m: ds[f"est_{m}"].values for m in MODELOS},
                           "conjunto": ds["conjunto"].values},
                          index=pd.DatetimeIndex(ds["time"].values))
    return quadro[quadro["conjunto"] >= 2].dropna(subset=["inmet"])


def _rmse(a: pd.Series, b: pd.Series) -> float:
    return float(np.sqrt(np.nanmean((a.values - b.values) ** 2)))


# ── Figuras ───────────────────────────────────────────────────────────────────

def figura_da_estacao(serie: pd.DataFrame, modelo: str) -> go.Figure:
    """Rajada medida, ERA5 e modelo ao longo de um ano, só nos dias que o modelo não viu."""
    # Os meses de teste são quatro por ano: sem quebrar a linha entre eles, ela cruzaria o vazio.
    quadro = serie.asfreq("D")
    f = go.Figure()
    f.add_trace(go.Scatter(x=quadro.index, y=quadro["era5"], name="ERA5 sem correção",
                           mode="lines", line=dict(color="#9aa3ad", width=1.4),
                           hovertemplate="ERA5: %{y:.1f} m/s<extra></extra>"))
    f.add_trace(go.Scatter(x=quadro.index, y=quadro[modelo], name=NOMES_MODELO[modelo],
                           mode="lines", line=dict(color=COR_MODELO, width=1.8),
                           hovertemplate="Modelo: %{y:.1f} m/s<extra></extra>"))
    f.add_trace(go.Scatter(x=quadro.index, y=quadro["inmet"], name="Medida pelo INMET",
                           mode="markers", marker=dict(color=TINTA, size=5),
                           hovertemplate="INMET: %{y:.1f} m/s<extra></extra>"))
    f.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14), hovermode="x unified",
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(color=TINTA)),
        xaxis=dict(gridcolor="#26292d", tickfont=dict(color=TINTA_FRACA)),
        yaxis=dict(title="Rajada máxima do dia (m/s)", gridcolor="#26292d",
                   tickfont=dict(color=TINTA_FRACA), rangemode="tozero", automargin=True))
    return f


# ── Página ────────────────────────────────────────────────────────────────────

def _arquivos_e_conteudo(arq: pd.DataFrame, regiao: dict):
    st.markdown("#### Os seis arquivos")
    tabela = pd.DataFrame({
        "Cluster": arq["cluster"],
        "Região": arq["cluster"].map(regiao),
        "Grade (lat × lon)": arq["grade"],
        "Pontos no cluster": arq["pontos"],
        "Estações": arq["estacoes"],
        "Tamanho": arq["tamanho"].map(lambda v: f"{_virgula(v, 0)} MB"),
    })
    st.dataframe(tabela, hide_index=True, width="stretch")

    st.markdown("#### O que há em cada arquivo")
    st.table({"Parte": [v[0] for v in VARIAVEIS], "Dimensões": [v[1] for v in VARIAVEIS],
              "Variável": [v[2] for v in VARIAVEIS], "O que guarda": [v[3] for v in VARIAVEIS]})
    st.markdown(
        "A rajada está em m/s, e multiplicar por 3,6 dá km/h. Os valores já são finais, com o "
        "ajuste de quantis aplicado."
    )

    st.markdown("#### Quais dias provam previsão")
    st.markdown(
        "A variável `conjunto` diz para que cada dia serviu. **Só teste e 2025 provam previsão**: "
        "treino e validação foram vistos pelo modelo."
    )
    dias = arq.set_index("cluster")[list(CONJUNTO.values()) + ["sem dado"]].astype(int)
    dias.index = [f"Cluster {c}" for c in dias.index]
    dias.columns = [c.capitalize() for c in dias.columns]
    st.dataframe(dias.map(_milhar), width="stretch")
    st.caption(
        "Dias de 2000 a 2025. Um dia conta como teste, treino ou validação se alguma estação do "
        "cluster o usou assim. O cluster 5 não tem 2025."
    )


def _exemplo():
    st.markdown("#### Veja uma estação")
    st.markdown(
        "O que o arquivo entrega numa estação do INMET, só nos dias que o modelo não viu. "
        "Cada ponto branco é a rajada medida; as linhas são o ERA5 e o modelo escolhido."
    )
    versao = _versao_nc()
    a, b, c = st.columns(3)
    cluster = a.selectbox("Cluster", CLUSTERS_NC, index=2, key="entrega_cluster",
                          format_func=lambda x: f"Cluster {x}")
    estacoes = _estacoes_do_cluster(cluster, versao)
    estacao = b.selectbox("Estação", estacoes, key=f"entrega_estacao_{cluster}")
    modelo = c.selectbox("Modelo", MODELOS, index=MODELOS.index("lstm_global_anual"),
                         key="entrega_modelo", format_func=NOMES_MODELO.get)
    serie = _serie(cluster, estacao, versao)
    anos = sorted(serie.index.year.unique())
    ano_forte = int(serie["inmet"].groupby(serie.index.year).max().idxmax())
    ano = st.select_slider("Ano", options=anos, value=ano_forte, key=f"entrega_ano_{cluster}_{estacao}")
    recorte = serie[serie.index.year == ano]

    st.plotly_chart(figura_da_estacao(recorte, modelo), width="stretch", config=GRAFICO)
    e5, md = _rmse(recorte["era5"], recorte["inmet"]), _rmse(recorte[modelo], recorte["inmet"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Dias de teste no ano", _milhar(len(recorte)))
    m2.metric("Erro do ERA5", f"{_virgula(e5, 2)} m/s")
    m3.metric("Erro do modelo", f"{_virgula(md, 2)} m/s",
              delta=f"{_virgula((md - e5) / e5 * 100, 1)}%", delta_color="inverse")
    st.caption("Erro (RMSE) só nos dias mostrados. Uma estação num ano é pouco para concluir: "
               "o resultado do cluster está em Resultados.")


def _como_usar():
    st.markdown("#### Como abrir")
    st.markdown("Os arquivos abrem com `xarray`. Três receitas:")
    a, b, c = st.tabs(["Erro de um modelo", "Série de uma estação", "Valor num ponto"])
    with a:
        st.code(CODIGO_ERRO, language="python")
        st.caption("Reproduz o erro da LSTM global anual no cluster 3 da seção Resultados.")
    with b:
        st.code(CODIGO_ESTACAO, language="python")
    with c:
        st.code(CODIGO_PONTO, language="python")


def _qual_modelo(erro: pd.DataFrame, var: pd.DataFrame):
    st.markdown("#### Qual modelo usar")
    por_modelo = var.groupby("modelo")["variacao"].mean()
    lstm, ml = por_modelo["lstm_global_anual"], por_modelo["ml_global_anual"]
    trimestrais = var[var["modelo"].isin(["ml_global_trimestral", "lstm_global_trimestral"])]
    n_pior, n_caso = int((trimestrais["variacao"] > 0).sum()), len(trimestrais)

    # O melhor de cada cluster, lido da tabela de erros.
    erros = erro[erro["modelo"] != "era5"].copy()
    era5 = erro[erro["modelo"] == "era5"].set_index("cluster")["rmse_teste"]
    melhor = erros.loc[erros.groupby("cluster")["rmse_teste"].idxmin()].set_index("cluster")
    lstm_anual = erros[erros["modelo"] == "lstm_global_anual"].set_index("cluster")["rmse_teste"]
    tabela = pd.DataFrame({
        "ERA5 sem correção": era5.map(lambda v: f"{_virgula(v, 2)} m/s"),
        "LSTM global anual": lstm_anual.map(lambda v: f"{_virgula(v, 2)} m/s"),
        "Menos erro que o ERA5": ((lstm_anual - era5) / era5 * 100).map(lambda v: f"{_virgula(abs(v), 1)}%"),
        "Menor erro do cluster": melhor["modelo"].map(NOMES_MODELO),
    })
    tabela.index = [f"Cluster {c}" for c in tabela.index]
    outros = melhor[melhor["modelo"] != "lstm_global_anual"]

    st.markdown(
        f"A **LSTM global anual** é a recomendada: em média {_virgula(abs(lstm), 1)}% abaixo do erro "
        f"do ERA5 nos seis clusters, e a de menor erro em {len(melhor) - len(outros)} deles. O **ML global "
        f"anual**, um CatBoost, chega a {_virgula(abs(ml), 1)}% e não precisa de GPU nem de janelas de "
        "24 horas: serve a um ambiente simples."
    )
    st.dataframe(tabela, width="stretch")
    if len(outros):
        c = int(outros.index[0])
        st.caption(
            f"No cluster {c}, {NOMES_MODELO[outros['modelo'].iloc[0]]} fica à frente por "
            f"{_virgula(lstm_anual[c] - outros['rmse_teste'].iloc[0], 3)} m/s: um empate, que não muda a recomendação."
        )
    st.markdown(
        f"Os dois modelos globais por trimestre ficam acima do erro do ERA5 em {n_pior} dos {n_caso} "
        "casos (cluster e modelo), e por isso não são os indicados."
    )


def _ressalvas(est: pd.DataFrame, det: pd.DataFrame):
    st.markdown("#### Com que ressalva")
    pod = det[(det["nivel"] == "P99") & (det["modelo"] == "lstm_global_anual")]
    c3_est = est[est["modelada"] & (est["cluster"] == 3)]
    ressalvas = [
        f"**Dias de vendaval.** O modelo reduz o erro médio, mas detecta de {pod['pod'].min():.0%} a "
        f"{pod['pod'].max():.0%} dos dias extremos, e de {pod['far'].min():.0%} a {pod['far'].max():.0%} "
        "dos alarmes são falsos. Não é um sistema de alerta.",
        "**Estações novas.** O painel não mostra como o modelo se sai numa estação que ficou fora do "
        "treino, porque o teste usa as mesmas estações do treino, em meses que o modelo não viu.",
        "**Só onde há estação no treino.** Segundo a documentação do pipeline, sem estações da região no "
        "treino nenhum modelo melhora o ERA5. Um cluster novo precisa de estações próprias.",
        f"**Cluster 3.** Das {len(c3_est)} estações, só {int(c3_est['com_serie'].sum())} têm rajada medida "
        "até 2024. O cluster 5 não tem 2025, e a vizinhança do ERA5 dele não é completa.",
        "**Verão.** O teste cobre só janeiro de cada ano, o único mês de verão. Nele, o ML local e o "
        "ML global por trimestre não melhoram o ERA5, e os dois modelos globais anuais mantêm o ganho "
        "dos outros meses de teste.",
        "**Uma semente.** Cada ajuste usa uma semente só. Os campeões de aprendizado de máquina vieram "
        "do cluster 3 e não foram retreinados por região.",
    ]
    for r in ressalvas:
        st.markdown(f"- {r}")


def render():
    est = cl.carregar_estacoes()
    erro = pd.read_csv(ERRO_POR_CLUSTER)
    det = pd.read_csv(DETECCAO)
    var = _variacao(erro)

    st.header("5 · Entrega")
    st.markdown(
        "Para cada cluster com modelo, **um arquivo** com a rajada máxima do dia prevista, de 2000 a "
        "2025, em cada ponto do ERA5 e em cada estação do INMET, com o ERA5 e os modelos corrigidos lado a lado. "
        "A previsão da estação vem ao lado da rajada medida, para quem quiser calcular as próprias métricas. "
        "A LSTM global por trimestre só existe nas estações."
    )

    if _nc_disponivel():
        arq = _arquivos(_versao_nc())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Arquivos", len(arq))
        c2.metric("Pontos do ERA5", _milhar(int(arq["pontos"].sum())))
        c3.metric("Estações", _milhar(int(arq["estacoes"].sum())))
        c4.metric("Tamanho", f"{_virgula(arq['tamanho'].sum(), 0)} MB")
        regiao = {c: cl.REGIAO[c] for c in CLUSTERS_NC}
        _arquivos_e_conteudo(arq, regiao)
        _exemplo()
    else:
        st.info("Os arquivos de previsão não estão neste ambiente, por isso as tabelas e o exemplo "
                "não aparecem.")
    _como_usar()
    _qual_modelo(erro, var)
    _ressalvas(est, det)
