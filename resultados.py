"""Seção 4, os resultados: o que funcionou, em quatro perguntas e um mapa.

Fontes dos números, todos calculados dos arquivos de previsão
(`predicoes_rajada_cluster*.nc`) ou do estudo do cluster 3:
- `dataset/erro_por_cluster.csv`: erro (RMSE) de cada modelo no teste, por cluster.
- `dataset/deteccao_extremos.csv`: POD e FAR dos dias extremos. O limiar de cada
  estação é o P95 ou o P99 da rajada medida nos dias de treino, avaliado nos
  meses de teste. É um cálculo próprio do painel.
- `dataset/vies_modelos_extremos.csv`: previsto menos medido nos dias extremos,
  pelo mesmo critério (P95 ou P99 de cada estação nos dias de treino, comparação
  com `>=`, mínimo de 100 dias de treino). Para o ERA5 reproduz `vies_era5.csv`.
- `dataset/estudo_c3_importancia.csv`: poder preditivo das variáveis, cluster 3.
- `artifacts/predicoes/*.nc`: os mapas. Ficam fora do git por enquanto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import clusters as cl
import escala_vento as ev
from metodo import NOMES_MODELO, IMPORTANCIA, ERRO_POR_CLUSTER, CONFIGURACOES, _milhar, _virgula

TINTA = "#f2f0eb"
TINTA_FRACA = "#b4b1a9"
FUNDO = "#17181a"
GRAFICO = {"responsive": True, "displaylogo": False}
DETECCAO = "dataset/deteccao_extremos.csv"
VIES_MODELOS = "dataset/vies_modelos_extremos.csv"
PASTA_NC = Path("artifacts/predicoes")
CLUSTERS_NC = (1, 2, 3, 4, 5, 12)
MODELOS_GRADE = ("era5", "ml_local", "ml_global_trimestral", "ml_global_anual",
                 "lstm_local", "lstm_global_anual")
CORES_GRUPO = {"grupo1": "#3d7ea6", "grupo2": "#d9a441", "grupo3": "#9085e9", "grupo4": "#9aa3ad"}
NOME_GRUPO = {"grupo1": "Grupo 1 · vento e camada limite", "grupo2": "Grupo 2 · convecção",
              "grupo3": "Grupo 3 · sinótico", "grupo4": "Grupo 4 · relevo"}
# Fronteiras das faixas de cor, uma escala por estatística, iguais nos três mapas.
# Um dia: a faixa mais escura, de 20 m/s para cima, é a de vendaval.
LIMITES_MAPA = {
    "dia": [0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30],
    50: [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    90: [0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
    95: [0, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    99: [0, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
}
ESTATISTICAS = {"Um dia": "dia", "Mediana (P50)": 50, "P90": 90, "P95": 95, "P99": 99}
CONJUNTO = {0: "treino", 1: "validação", 2: "teste", 3: "2025"}


def _pct(v: float) -> str:
    return f"{v:+.1f}".replace(".", ",").replace("-", "−") + "%"


# ── Dados ─────────────────────────────────────────────────────────────────────

def _variacao(erro: pd.DataFrame) -> pd.DataFrame:
    """Variação do erro de cada modelo contra o ERA5, em %, por cluster. Negativo é melhor."""
    era5 = erro[erro["modelo"] == "era5"].set_index("cluster")["rmse_teste"]
    sub = erro[erro["modelo"] != "era5"].copy()
    sub["variacao"] = (sub["rmse_teste"] - sub["cluster"].map(era5)) / sub["cluster"].map(era5) * 100
    return sub


@st.cache_resource(show_spinner=False)
def _abrir(cluster: int, versao):
    import xarray as xr
    return xr.open_dataset(PASTA_NC / f"predicoes_rajada_cluster{cluster}.nc")


def _versao_nc() -> tuple:
    return tuple(p.stat().st_mtime_ns for p in sorted(PASTA_NC.glob("*.nc")))


def _nc_disponivel() -> bool:
    return all((PASTA_NC / f"predicoes_rajada_cluster{c}.nc").exists() for c in CLUSTERS_NC)


@st.cache_data(show_spinner=False)
def _dia_padrao(versao) -> pd.Timestamp:
    """O dia de teste com mais estações com rajada medida de 20 m/s ou mais."""
    total = None
    for c in CLUSTERS_NC:
        ds = _abrir(c, versao)
        obs = ds["est_inmet"].values >= 20
        conta = obs.sum(axis=0) * (ds["conjunto"].values == 2)
        total = conta if total is None else total + conta
    return pd.Timestamp(ds["time"].values[int(np.argmax(total))])


# ── Figuras ───────────────────────────────────────────────────────────────────

def figura_das_variaveis(imp: pd.DataFrame) -> go.Figure:
    """As variáveis mantidas, pelo poder preditivo, coloridas pelo grupo."""
    m = imp[imp["decisao"] == "manter"].sort_values("d_rmse_medio", ascending=True)
    nomes = [f"{v} (retirada depois)" if v == "anor" else v for v in m["variavel"]]
    figura = go.Figure(go.Bar(
        x=m["d_rmse_medio"], y=nomes, orientation="h",
        marker_color=[CORES_GRUPO[g] for g in m["grupo"]], showlegend=False,
        text=[f"{v:.3f}".replace(".", ",") for v in m["d_rmse_medio"]], textposition="outside",
        textfont=dict(color=TINTA), cliponaxis=False,
        hovertemplate="%{y}<br>Piora do erro: %{x:.3f} m/s<extra></extra>"))
    for g, cor in CORES_GRUPO.items():
        figura.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=NOME_GRUPO[g],
                                    marker=dict(size=11, color=cor, symbol="square")))
    figura.update_layout(
        height=430, margin=dict(l=10, r=40, t=10, b=40), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=TINTA, size=13)),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        xaxis=dict(title="Quanto o erro piora ao embaralhar a variável (m/s)", gridcolor="#26292d",
                   tickfont=dict(color=TINTA_FRACA)),
        yaxis=dict(tickfont=dict(color=TINTA), showgrid=False, automargin=True))
    return figura


def figura_lstm_vs_arvores(cfg: pd.DataFrame) -> go.Figure:
    """O erro de cinco configurações do cluster 3, nas árvores e na LSTM."""
    ordem = [("base", "Só vento e camada limite"), ("full", "Todos os grupos"),
             ("sel__val12", "12 variáveis selecionadas"), ("drop_grp__grupo1", "Todos menos o grupo 1")]
    figura = go.Figure()
    for origem, nome, cor in (("arvores", "Árvores (média de 5 modelos e 5 sementes)", "#3d7ea6"),
                              ("lstm", "Rede LSTM (uma semente)", "#d9a441")):
        sub = cfg[cfg["origem"] == origem].set_index("arm")["rmse_teste"]
        vals = [float(sub[a]) for a, _ in ordem]
        figura.add_trace(go.Bar(
            x=[n for _, n in ordem], y=vals, name=nome, marker_color=cor,
            text=[_virgula(v) for v in vals], textposition="outside", textfont=dict(color=TINTA),
            hovertemplate="%{x}<br>Erro: %{y:.2f} m/s<extra>" + nome + "</extra>"))
    era5 = float(cfg[cfg["origem"] == "referencia"]["rmse_teste"].iloc[0])
    figura.add_hline(y=era5, line=dict(color="#9aa3ad", dash="dash", width=1),
                     annotation_text="ERA5 sem correção", annotation_position="top left", annotation_font_color=TINTA_FRACA)
    figura.update_layout(
        barmode="group", height=380, margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=TINTA)),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        xaxis=dict(tickfont=dict(color=TINTA), showgrid=False),
        yaxis=dict(title="Erro no teste (RMSE, m/s)", range=[0, 3.1], gridcolor="#26292d",
                   tickfont=dict(color=TINTA_FRACA), automargin=True))
    return figura


def figura_da_variacao(erro: pd.DataFrame, em_pct: bool = True) -> go.Figure:
    """Cada modelo em cada cluster: o erro contra o do ERA5 (em %) ou o erro em m/s."""
    v = _variacao(erro)
    modelos = [m for m in NOMES_MODELO if m != "era5"] if em_pct else list(NOMES_MODELO)
    cl_ids = sorted(erro["cluster"].unique())
    if em_pct:
        z = [[float(v[(v.cluster == c) & (v.modelo == m)]["variacao"].iloc[0]) for m in modelos] for c in cl_ids]
        texto = [[_pct(x) for x in linha] for linha in z]
        lim = max(abs(np.array(z).min()), abs(np.array(z).max()))
        extra = dict(zmin=-lim, zmax=lim, colorscale=[[0, "#3d7ea6"], [0.5, "#26292d"], [1, "#b8932f"]],
                     colorbar=dict(title=dict(text="Erro do modelo<br>contra o do ERA5", font=dict(color=TINTA)),
                                   tickfont=dict(color=TINTA), thickness=14, outlinewidth=0, ticksuffix="%"))
        dica = "%{y}<br>%{x}<br>Erro %{text} contra o do ERA5<extra></extra>"
    else:
        z = [[float(erro[(erro.cluster == c) & (erro.modelo == m)]["rmse_teste"].iloc[0]) for m in modelos] for c in cl_ids]
        texto = [[_virgula(x) for x in linha] for linha in z]
        extra = dict(zmin=0, colorscale=[[0, "#1d2c3a"], [1, "#6fb0dc"]],
                     colorbar=dict(title=dict(text="Erro<br>(m/s)", font=dict(color=TINTA)),
                                   tickfont=dict(color=TINTA), thickness=14, outlinewidth=0))
        dica = "%{y}<br>%{x}<br>Erro médio: %{text} m/s<extra></extra>"
    figura = go.Figure(go.Heatmap(
        x=[NOMES_MODELO[m] for m in modelos], y=[f"Cluster {c}" for c in cl_ids], z=z,
        text=texto, texttemplate="%{text}", textfont=dict(color=TINTA, size=14),
        xgap=3, ygap=3, hovertemplate=dica, **extra))
    figura.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        xaxis=dict(side="top", tickfont=dict(color=TINTA), showgrid=False, automargin=True),
        yaxis=dict(autorange="reversed", tickfont=dict(color=TINTA), showgrid=False, automargin=True))
    return figura


def figura_da_deteccao(det: pd.DataFrame, nivel: str, modelo: str) -> go.Figure:
    """POD e FAR por cluster, o ERA5 e um modelo."""
    sub = det[det["nivel"] == nivel]
    cl_ids = sorted(sub["cluster"].unique())
    n_ext = sub[sub["modelo"] == "era5"].set_index("cluster")["dias_extremos_medidos"]
    nomes = [f"Cluster {c}<br>{int(n_ext[c])} dias extremos" for c in cl_ids]
    figura = go.Figure()
    for m, cor in (("era5", "#9aa3ad"), (modelo, "#3d7ea6")):
        s = sub[sub["modelo"] == m].set_index("cluster").loc[cl_ids]
        figura.add_trace(go.Bar(
            x=nomes, y=s["pod"], name=f"{NOMES_MODELO[m]}", marker_color=cor,
            text=[f"{v:.0%}" for v in s["pod"]], textposition="outside", textfont=dict(color=TINTA),
            hovertemplate="%{x}<br>Dias extremos detectados: %{y:.0%}<extra>" + NOMES_MODELO[m] + "</extra>"))
    figura.update_layout(
        barmode="group", height=360, margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=TINTA)),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        xaxis=dict(tickfont=dict(color=TINTA), showgrid=False),
        yaxis=dict(title="Dias extremos detectados (POD)", range=[0, 1], tickformat=".0%",
                   gridcolor="#26292d", tickfont=dict(color=TINTA_FRACA), automargin=True))
    return figura


def figura_do_vies_extremo(vm: pd.DataFrame, nivel: str, modelo: str) -> go.Figure:
    """Previsto menos medido, em média, nos dias extremos: o ERA5 e um modelo, por cluster."""
    sub = vm[vm["nivel"] == nivel]
    cl_ids = sorted(sub["cluster"].unique())
    figura = go.Figure()
    for m, cor in (("era5", "#9aa3ad"), (modelo, "#3d7ea6")):
        s = sub[sub["modelo"] == m].set_index("cluster").loc[cl_ids]
        figura.add_trace(go.Bar(
            x=[f"Cluster {c}" for c in cl_ids], y=s["vies"], name=NOMES_MODELO[m], marker_color=cor,
            text=[_virgula(v, 1) for v in s["vies"]], textposition="outside", textfont=dict(color=TINTA),
            hovertemplate="%{x}<br>Previsto menos medido: %{y:.1f} m/s<extra>" + NOMES_MODELO[m] + "</extra>"))
    figura.add_hline(y=0, line=dict(color="#9aa3ad", width=1))
    figura.update_layout(
        barmode="group", height=360, margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=TINTA)),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        xaxis=dict(tickfont=dict(color=TINTA), showgrid=False),
        yaxis=dict(title="Previsto menos medido (m/s)", gridcolor="#26292d",
                   tickfont=dict(color=TINTA_FRACA), automargin=True))
    return figura


def figura_da_generalizacao(erro: pd.DataFrame) -> go.Figure:
    """A variação do erro contra o ERA5 nos clusters 3 e 12, modelo a modelo."""
    v = _variacao(erro)
    modelos = [m for m in NOMES_MODELO if m != "era5"]
    figura = go.Figure()
    for c, cor in ((3, "#9aa3ad"), (12, "#3d7ea6")):
        s = v[v["cluster"] == c].set_index("modelo").loc[modelos]
        figura.add_trace(go.Bar(
            x=[NOMES_MODELO[m] for m in modelos], y=s["variacao"],
            name=f"Cluster {c}" + (", onde as variáveis foram escolhidas" if c == 3 else ", longe do Sul"),
            marker_color=cor, text=[_pct(x) for x in s["variacao"]], textposition="outside",
            textfont=dict(color=TINTA),
            hovertemplate="%{x}<br>Erro: %{text} contra o ERA5<extra>Cluster " + str(c) + "</extra>"))
    figura.add_hline(y=0, line=dict(color="#9aa3ad", width=1))
    figura.update_layout(
        barmode="group", height=400, margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=TINTA)),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        xaxis=dict(tickfont=dict(color=TINTA), showgrid=False),
        yaxis=dict(title="Erro contra o ERA5 (menor é melhor)", gridcolor="#26292d",
                   ticksuffix="%", tickfont=dict(color=TINTA_FRACA), automargin=True))
    return figura


@st.cache_data(show_spinner=False)
def _campo_grade(cluster: int, modelo: str, q, so_fora: bool, versao):
    """Um dia, ou o percentil q da rajada máxima diária ao longo dos dias, em cada célula."""
    ds = _abrir(cluster, versao)
    mascara = ds["celula_no_cluster"].values == 1
    if q == "dia":
        raise ValueError("Um dia não passa por aqui.")
    conj = ds["conjunto"].values
    dias = (conj == 2) | (conj == 3) if so_fora else (conj >= 0)
    z = ds[modelo].values[dias].astype(float)
    campo = np.nanpercentile(z, q, axis=0)
    return np.where(mascara, campo, np.nan), ds["lon"].values, ds["lat"].values


@st.cache_data(show_spinner=False)
def _campo_estacoes(cluster: int, q, so_fora: bool, versao):
    """O percentil q da rajada medida em cada estação, nos mesmos dias, com 30 dias ou mais."""
    ds = _abrir(cluster, versao)
    conj = ds["conjunto"].values
    dias = (conj == 2) | (conj == 3) if so_fora else (conj >= 0)
    obs = ds["est_inmet"].values[:, dias].astype(float)
    valido = np.isfinite(obs).sum(axis=1) >= 30
    val = np.full(obs.shape[0], np.nan)
    if valido.any():
        val[valido] = np.nanpercentile(obs[valido], q, axis=1)
    return val, ds["estacao_lat"].values, ds["estacao_lon"].values, [str(e) for e in ds["estacao"].values]


def mapa(g, painel: str, estatistica, dia: pd.Timestamp | None, so_fora: bool,
         titulo: str, com_barra: bool) -> go.Figure:
    """Um dos três mapas: o INMET (só estações), o ERA5 ou um modelo (grade e estações).

    `estatistica` é "dia" ou o percentil (50, 90, 95, 99). Os três mapas usam a
    mesma escala de cor, para dar para comparar.
    """
    versao = _versao_nc()
    esc = ev.escala(LIMITES_MAPA[estatistica], cinza_abaixo=True)
    figura = go.Figure()
    for linha in g.itertuples():
        xs, ys = cl._contorno(linha.geometry)
        figura.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                    line=dict(color="#4a4f57", width=1)))
    est_lat, est_lon, est_val, est_nome = [], [], [], []
    for c in CLUSTERS_NC:
        ds = _abrir(c, versao)
        if estatistica == "dia":
            i = int(np.searchsorted(ds["time"].values, np.datetime64(dia)))
            v = ds["est_inmet"].isel(time=i).values.astype(float)
            la, lo = ds["estacao_lat"].values, ds["estacao_lon"].values
            nomes = [str(e) for e in ds["estacao"].values]
        else:
            v, la, lo, nomes = _campo_estacoes(c, estatistica, so_fora, versao)
        ok = np.isfinite(v)
        est_lat += list(la[ok]); est_lon += list(lo[ok]); est_val += list(v[ok])
        est_nome += [n for n, o in zip(nomes, ok) if o]
        if painel == "inmet":
            continue
        if estatistica == "dia":
            z = ds[painel].isel(time=i).values.astype(float)
            z = np.where(ds["celula_no_cluster"].values == 1, z, np.nan)
            x, y = ds["lon"].values, ds["lat"].values
        else:
            z, x, y = _campo_grade(c, painel, estatistica, so_fora, versao)
        gx, gy = np.meshgrid(x, y)
        ok = np.isfinite(z)
        figura.add_trace(go.Scatter(
            x=gx[ok], y=gy[ok], mode="markers", showlegend=False,
            marker=dict(size=5, color=z[ok], coloraxis="coloraxis"),
            hovertemplate="%{marker.color:.1f} m/s<extra>Cluster " + str(c) + "</extra>"))
    if painel == "inmet":
        figura.add_trace(go.Scatter(
            x=est_lon, y=est_lat, mode="markers", showlegend=False, text=est_nome, customdata=est_val,
            marker=dict(size=10, color=est_val, coloraxis="coloraxis", line=dict(color="#ffffff", width=1.2)),
            hovertemplate="Estação %{text}<br>Medido: %{customdata:.1f} m/s<extra></extra>"))
    barra = dict(esc["colorbar"], title=dict(text="m/s", font=dict(color=TINTA)),
                 tickfont=dict(color=TINTA), thickness=14, outlinewidth=0, len=0.7)
    figura.update_layout(coloraxis=dict(colorscale=esc["colorscale"], cmin=esc["cmin"], cmax=esc["cmax"],
                                        showscale=com_barra, colorbar=barra))
    minx, miny, maxx, maxy = g.total_bounds
    xlim, ylim = cl._caixa([miny, maxy], [minx, maxx], margem=0.4)
    figura = cl._plano(figura, xlim, ylim, altura=520)
    figura.update_layout(title=dict(text=titulo, font=dict(color=TINTA, size=15), x=0.02, y=0.98),
                         margin=dict(l=45, r=5, t=40, b=30))
    figura.update_xaxes(tickfont=dict(size=10))
    figura.update_yaxes(tickfont=dict(size=10))
    return figura


# ── Seção ─────────────────────────────────────────────────────────────────────

def render():
    imp = pd.read_csv(IMPORTANCIA)
    erro = pd.read_csv(ERRO_POR_CLUSTER)
    det = pd.read_csv(DETECCAO)
    cfg = pd.read_csv(CONFIGURACOES)
    var = _variacao(erro)

    st.header("4 · Resultados")
    st.markdown(
        "Quatro perguntas, na ordem em que se respondem: quais variáveis importam, qual modelo "
        "erra menos, se ele acerta os dias de vendaval e se vale fora do cluster onde foi escolhido. "
        "No fim, o mapa do modelo aplicado."
    )

    # 4.1
    st.markdown("#### 1. Quais variáveis funcionam melhor")
    st.markdown(
        "Cada barra mostra **quanto o erro do modelo piora quando uma variável é embaralhada**: "
        "quanto maior a barra, mais o modelo depende dela. As cores são os grupos. Estudo feito no "
        "cluster 3, só com as variáveis mantidas."
    )
    m = imp[imp["decisao"] == "manter"].sort_values("d_rmse_medio", ascending=False)
    total = float(m["d_rmse_medio"].sum())
    top = m.iloc[0]
    por_grupo = m.groupby("grupo")["d_rmse_medio"].sum() / total
    st.plotly_chart(figura_das_variaveis(imp), width="stretch", config=GRAFICO, key="res_variaveis")
    st.markdown(
        f"Das {len(imp)} variáveis estudadas, {len(m)} foram mantidas. A rajada do próprio ERA5, "
        f"`{top['variavel']}`, pesa sozinha {top['d_rmse_medio'] / total:.0%} da piora somada das "
        f"mantidas, e o grupo 1 inteiro, {por_grupo['grupo1']:.0%}. Convecção e pressão somam "
        f"{por_grupo.get('grupo2', 0) + por_grupo.get('grupo3', 0):.0%}. O relevo (`anor`) é forte, "
        "mas foi retirado depois: segundo o estudo, ele identifica a estação mais do que descreve o terreno."
    )

    l = cfg[cfg["origem"] == "lstm"].set_index("arm")["rmse_teste"]
    a = cfg[cfg["origem"] == "arvores"].set_index("arm")["rmse_teste"]
    st.markdown("##### E com a rede LSTM?")
    st.markdown(
        "A LSTM é uma rede neural que lê a sequência das horas até o pico, em vez de olhar um "
        "instante só. Ela foi testada nas mesmas configurações, para ver se a conclusão das árvores "
        "de decisão se repete."
    )
    st.plotly_chart(figura_lstm_vs_arvores(cfg), width="stretch", config=GRAFICO, key="res_lstm")
    st.caption(
        "Cada barra é o erro médio no teste, em m/s, e menor é melhor. A linha tracejada é o ERA5 sem "
        "correção. As árvores são a média de cinco modelos de árvores de decisão, com cinco sementes; "
        "a LSTM, uma semente só."
    )
    st.markdown(
        f"O grupo 1 sozinho já basta nas duas. Nas árvores, juntar os outros grupos não muda o erro "
        f"({_virgula(float(a['base']))} em todas). Na LSTM, juntar tudo piora, de {_virgula(float(l['base']))} "
        f"para {_virgula(float(l['full']))}, e as 12 variáveis ({_virgula(float(l['sel__val12']))}) também "
        "ficam atrás do grupo 1 sozinho. A seleção foi feita com árvores, e a rede, com poucas linhas "
        "de treino, sobreajusta as colunas a mais."
    )

    # 4.2
    st.markdown("#### 2. Qual modelo erra menos")
    st.markdown(
        "Cada quadrado compara o erro de um modelo com o erro do **ERA5 sem correção**, no mesmo "
        "cluster. Azul é erro menor que o do ERA5, dourado é maior. Por exemplo, −12,6% quer dizer "
        "um erro 12,6% menor."
    )
    visao = st.radio("Mostrar", ["Em relação ao ERA5", "Erro médio em m/s"], horizontal=True,
                     key="res_visao")
    st.plotly_chart(figura_da_variacao(erro, visao == "Em relação ao ERA5"), width="stretch",
                    config=GRAFICO, key="res_variacao")
    melhor_por_cluster = var.loc[var.groupby("cluster")["rmse_teste"].idxmin()]
    n_lstm = int((melhor_por_cluster["modelo"] == "lstm_global_anual").sum())
    pior = var.loc[var["variacao"].idxmax()]
    n_acima = int((var["variacao"] > 0).sum())
    st.markdown(
        f"A **LSTM global anual** erra menos em {n_lstm} dos {var['cluster'].nunique()} clusters. "
        f"Em {n_acima} das {len(var)} combinações de modelo e cluster o erro fica acima do ERA5; "
        f"a pior é o {NOMES_MODELO[pior['modelo']]} no cluster {int(pior['cluster'])}, "
        f"com {_pct(pior['variacao'])[1:]} a mais."
    )

    # 4.3
    st.markdown("#### 3. E os dias de vendaval")
    st.markdown(
        "Um **dia extremo** é um dia em que a rajada medida passa do P99 (ou do P95) da própria "
        "estação. A pergunta é: o modelo também avisa esse dia?"
    )
    c1, c2 = st.columns([1, 2])
    nivel = c1.radio("Dia extremo", ["P99", "P95"], horizontal=True, key="res_nivel",
                     help="Rajada medida acima do P99 (ou P95) da própria estação nos dias de treino.")
    modelo = c2.selectbox("Modelo", [m for m in NOMES_MODELO if m != "era5"], index=5,
                          format_func=lambda x: NOMES_MODELO[x], key="res_modelo_det")
    st.plotly_chart(figura_da_deteccao(det, nivel, modelo), width="stretch", config=GRAFICO,
                    key="res_deteccao")
    sub = det[(det["nivel"] == nivel) & (det["modelo"] == modelo)]
    st.markdown(
        f"O modelo detecta de {sub['pod'].min():.0%} a {sub['pod'].max():.0%} dos dias extremos, e "
        f"de {sub['far'].min():.0%} a {sub['far'].max():.0%} dos alarmes que dá são falsos. Reduzir o "
        "erro médio não basta para o vendaval: o modelo ainda perde a maior parte dos dias extremos e "
        "erra boa parte dos alarmes."
    )
    st.caption(
        "A barra é o POD: a fração dos dias extremos medidos pelo INMET que o modelo também põe "
        "acima do limiar. O FAR é a fração dos alarmes que eram falsos. O limiar de cada estação é o "
        "P99 (ou P95) da rajada medida nos dias de treino, e a conta é feita nos meses de teste. "
        "Abaixo do nome de cada cluster, quantos dias extremos o INMET mediu."
    )

    st.markdown("##### E o quanto a rajada fica abaixo da medida?")
    vm = pd.read_csv(VIES_MODELOS)
    e5 = vm[(vm["nivel"] == nivel) & (vm["modelo"] == "era5")].set_index("cluster")["vies"]
    md = vm[(vm["nivel"] == nivel) & (vm["modelo"] == modelo)].set_index("cluster")["vies"]
    corte = (1 - md.abs() / e5.abs()) * 100
    melhor_c, pior_c = int(corte.idxmax()), int(corte.idxmin())
    n_menor = int((corte > 0).sum())

    def _lado(x: float) -> str:
        return f"{abs(x):.0f}% a menos" if x >= 0 else f"{abs(x):.0f}% a mais"

    continuam = "continuam subestimados em todos" if (md < 0).all() else "continuam subestimados em parte deles"
    st.markdown(
        "Na seção 2, o ERA5 ficava de "
        f"{_virgula(e5.abs().min(), 1)} a {_virgula(e5.abs().max(), 1)} m/s abaixo da rajada medida nos dias "
        f"extremos pelo {nivel}. Com o modelo escolhido, fica de {_virgula(md.abs().min(), 1)} a "
        f"{_virgula(md.abs().max(), 1)} m/s abaixo. O déficit é menor que o do ERA5 em {n_menor} dos "
        f"{len(corte)} clusters, de {_lado(corte.min())} (cluster {pior_c}) a {_lado(corte.max())} "
        f"(cluster {melhor_c}), e os dias extremos {continuam}."
    )
    st.plotly_chart(figura_do_vies_extremo(vm, nivel, modelo), width="stretch", config=GRAFICO,
                    key="res_vies_extremo")
    st.caption(
        "Cada barra é a rajada prevista menos a medida pelo INMET, em média, nos dias extremos dos meses de "
        "teste. Negativo quer dizer que a previsão fica abaixo. Vale a mesma ressalva da seção 2: escolher "
        "os dias pelo valor medido já puxa a previsão para baixo, tanto no ERA5 quanto no modelo, então o "
        "que se compara é a diferença entre as barras."
    )

    # 4.4
    st.markdown("#### 4. Vale fora do cluster onde foi escolhido")
    st.markdown(
        "O cluster 3 é onde as variáveis foram escolhidas, e o 12, longe do Sul, é a conferência. "
        "Cada barra é o erro do modelo **em relação ao ERA5 sem correção** no mesmo cluster: abaixo de "
        "zero é melhor que o ERA5."
    )
    st.plotly_chart(figura_da_generalizacao(erro), width="stretch", config=GRAFICO, key="res_generaliz")
    v3 = var[var["cluster"] == 3].set_index("modelo")["variacao"]
    v12 = var[var["cluster"] == 12].set_index("modelo")["variacao"]
    ambos = int(((v3 < 0) & (v12 < 0)).sum())
    acima = [NOMES_MODELO[x] for x in v3.index if v3[x] >= 0 and v12[x] >= 0]
    if len(acima) == 1:
        resto = f" O que fica acima nos dois é o {acima[0]}."
    elif acima:
        resto = f" Os que ficam acima nos dois são {' e '.join(acima)}."
    else:
        resto = ""
    st.markdown(
        f"{ambos} dos {len(v3)} modelos ficam abaixo do ERA5 nos dois clusters.{resto} As variáveis "
        "escolhidas no cluster 3 também reduzem o erro no cluster 12, no Sudeste, onde o vento tem "
        "outro regime. É uma conferência das variáveis, com estações do próprio cluster no treino."
    )

    # 4.5
    st.markdown("#### O modelo aplicado")
    if not _nc_disponivel():
        st.info("Os arquivos de previsão ainda não estão publicados neste painel.")
        return
    versao = _versao_nc()
    g = cl.carregar()
    c1, c2 = st.columns([3, 2])
    escolha = c1.radio("O que mostrar", list(ESTATISTICAS), horizontal=True, key="res_estat")
    estatistica = ESTATISTICAS[escolha]
    modelo_mapa = c2.selectbox("Modelo", list(MODELOS_GRADE[1:]), index=4,
                               format_func=lambda x: NOMES_MODELO[x], key="res_modelo_mapa")
    dia, so_fora = None, True
    if estatistica == "dia":
        padrao = _dia_padrao(versao)
        dia = pd.Timestamp(st.date_input(
            "Dia", value=padrao.date(), min_value=pd.Timestamp("2000-01-01").date(),
            max_value=pd.Timestamp("2025-12-31").date(), key="res_dia", format="DD/MM/YYYY",
            help="Começa no dia de teste com mais estações em vendaval."))
        ds0 = _abrir(CLUSTERS_NC[0], versao)
        conj = int(ds0["conjunto"].values[int(np.searchsorted(ds0["time"].values, np.datetime64(dia)))])
        if conj == 2:
            st.caption("Este dia está no teste, que o modelo não viu. Vale como previsão.")
        elif conj == 3:
            st.caption("Este dia é de 2025, ano que o modelo não viu. Vale como previsão.")
        else:
            st.caption(
                f"Este dia está no {CONJUNTO.get(conj, 'treino')}: o modelo o viu nas células com estação. "
                "Para ver uma previsão de verdade, escolha um dia de janeiro, abril, julho ou outubro, "
                "ou de 2025.")
        st.markdown("A rajada máxima do dia, em cada ponto do ERA5 e em cada estação.")
    else:
        so_fora = st.radio("Dias considerados", ["Só os que o modelo não viu (teste e 2025)", "Todos os dias"],
                           horizontal=True, key="res_dias") != "Todos os dias"
        contagens = []
        for c in CLUSTERS_NC:
            conj = _abrir(c, versao)["conjunto"].values
            contagens.append(int(((conj == 2) | (conj == 3)).sum()) if so_fora else int((conj >= 0).sum()))
        n_dias = (f"{_milhar(min(contagens))} a {_milhar(max(contagens))}"
                  if min(contagens) != max(contagens) else _milhar(contagens[0]))
        explica = ("a rajada máxima do dia típico: metade dos dias passa dela" if estatistica == 50
                   else f"a rajada máxima do dia que só é superada em {100 - estatistica}% dos dias")
        st.markdown(
            f"Em cada ponto, o **{escolha}** é {explica}. Calculado sobre {n_dias} dias, conforme o cluster"
            + (", que o modelo não viu." if so_fora else ", inclusive os de treino, que o modelo viu.")
        )
    c_in, c_era, c_mod = st.columns(3)
    with c_in:
        st.plotly_chart(mapa(g, "inmet", estatistica, dia, so_fora, "INMET, medido", False),
                        width="stretch", config=cl.GRAFICO_MAPA, key="res_mapa_inmet")
    with c_era:
        st.plotly_chart(mapa(g, "era5", estatistica, dia, so_fora, "ERA5 sem correção", False),
                        width="stretch", config=cl.GRAFICO_MAPA, key="res_mapa_era5")
    with c_mod:
        st.plotly_chart(mapa(g, modelo_mapa, estatistica, dia, so_fora, NOMES_MODELO[modelo_mapa], True),
                        width="stretch", config=cl.GRAFICO_MAPA, key="res_mapa_modelo")
    st.caption(
        "Os três mapas usam a mesma escala de cor, em m/s. Cada círculo do mapa do INMET é uma estação; "
        "ficam de fora as estações com menos de 30 dias medidos no período. O ERA5 e o modelo usam todos "
        "os dias do período, e o INMET só os dias em que a estação mediu."
    )
