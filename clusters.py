"""A bacia do vendaval, seus 14 clusters, as estações e as células do ERA5.

Os polígonos vêm de `dataset/shp/shp_vento.shp`, um por cluster. As estações
do INMET dentro da bacia (271, as mesmas da tabela de clusters do repositório do
pipeline) estão em `dataset/estacoes_bacia.csv`. Das 271, 117 aparecem nos
arquivos de previsão e, dessas, 98 têm rajada medida até 2024 (`dias_ate_2024`
de 365 ou mais); as outras só começaram a reportar em 2025. As células de 0,25 grau do ERA5 dos seis clusters
com modelo (821) estão em `dataset/celulas_era5_modeladas.csv`, só as
coordenadas, lidas dos arquivos de previsão do pipeline.

Os mapas são desenhados num plano de longitude e latitude, sem mapa de fundo:
o leitor vê só a bacia, e pode dar zoom e arrastar. A região de cada cluster é
a da documentação do pipeline. A situação de cada cluster (com modelo, em
obtenção, pendente) precisa ser revista quando mudar.
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

SHP = Path("dataset/shp/shp_vento.shp")
ESTACOES = Path("dataset/estacoes_bacia.csv")
CELULAS = Path("dataset/celulas_era5_modeladas.csv")

TINTA = "#f2f0eb"
TINTA_FRACA = "#b4b1a9"
FUNDO = "#17181a"
GRADE = "#26292d"
PASSO = 0.25  # grau, a grade do ERA5

ESTUDO = "Estudo das variáveis"
VALIDACAO = "Validação longe do Sul"
MODELADO = "Modelo treinado"

# cluster: região, papel no experimento.
MODELADOS = {
    1: ("Sul do RS", MODELADO),
    2: ("Oeste e noroeste do RS", MODELADO),
    3: ("Leste do RS", ESTUDO),
    4: ("Norte do RS e SC", MODELADO),
    5: ("Norte de SC, PR, SP e sul do RJ", MODELADO),
    12: ("Leste de MG, ES e RJ", VALIDACAO),
}
EM_OBTENCAO = {6}

# Cores calmas sobre o fundo escuro: o que tem modelo ganha cor, o resto recua.
SITUACOES = {
    "modelado": ("Com modelo treinado", "#3d7ea6"),
    "obtencao": ("Dados em obtenção", "#b8932f"),
    "pendente": ("Ainda não trabalhado", "#3a3f47"),
}
CONTORNO = "#aeb4bb"

# Uma cor por cluster. A ordem dos matizes foi escolhida por busca, para que
# dois clusters vizinhos nunca fiquem com cores parecidas: entre todos os pares
# que fazem fronteira, a menor distância é de 3 passos em 14 na roda de matizes
# (77 graus), e a luminosidade alterna entre tons fortes e claros.
COR_DO_CLUSTER = {
    1: "#96eeb8", 2: "#9f96ee", 3: "#eed296", 4: "#5de458", 5: "#5885e4",
    6: "#ee96a5", 7: "#a358e4", 8: "#d5e458", 9: "#eb96ee", 10: "#96d8ee",
    11: "#e47b58", 12: "#bfee96", 13: "#58e4cb", 14: "#e458ad",
}
COR_ESTACAO = "#ffffff"

GRAFICO_MAPA = {"responsive": True, "scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


def situacao(cluster: int) -> str:
    if cluster in MODELADOS:
        return "modelado"
    return "obtencao" if cluster in EM_OBTENCAO else "pendente"


# ── Leitura ───────────────────────────────────────────────────────────────────

def carregar() -> gpd.GeoDataFrame:
    """Os 14 polígonos, em graus (WGS 84), simplificados para o mapa."""
    g = gpd.read_file(SHP).to_crs(4326)
    g["cluster"] = g["cluster"].astype(int)
    g["geometry"] = g.geometry.simplify(0.01)
    return g.sort_values("cluster").reset_index(drop=True)


def carregar_estacoes() -> pd.DataFrame:
    """Todas as estações dentro da bacia. `modelada` marca as 117 dos arquivos de
    previsão, e `com_serie` as que têm 365 dias ou mais de rajada medida até 2024."""
    est = pd.read_csv(ESTACOES)
    est["com_serie"] = est["dias_ate_2024"].fillna(0) >= 365
    return est


def carregar_celulas() -> pd.DataFrame:
    """Células do ERA5, com a marca de quais têm estação do INMET dentro."""
    cel = pd.read_csv(CELULAS)
    est = carregar_estacoes().query("modelada and com_serie")
    cel["com_estacao"] = [
        bool(((est["cluster"] == r.cluster)
              & ((est["lat"] - r.lat).abs() <= PASSO / 2 + 1e-6)
              & ((est["lon"] - r.lon).abs() <= PASSO / 2 + 1e-6)).any())
        for r in cel.itertuples()
    ]
    return cel


# ── Desenho ───────────────────────────────────────────────────────────────────

def _contorno(geometria) -> tuple[list, list]:
    """Anéis externos de um polígono ou multipolígono, separados por vazio."""
    partes = geometria.geoms if geometria.geom_type == "MultiPolygon" else [geometria]
    xs, ys = [], []
    for p in partes:
        x, y = p.exterior.xy
        xs += list(x) + [None]
        ys += list(y) + [None]
    return xs, ys


def _ponto_do_rotulo(geometria):
    """Um ponto garantidamente dentro do polígono, na maior parte dele."""
    if geometria.geom_type == "MultiPolygon":
        geometria = max(geometria.geoms, key=lambda p: p.area)
    return geometria.representative_point()


def _poligono(linha, cor: str, texto: str, **extra) -> go.Scatter:
    xs, ys = _contorno(linha.geometry)
    return go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself", fillcolor=cor,
        line=dict(color=CONTORNO, width=1), hoveron="fills", text=texto,
        hovertemplate="%{text}<extra></extra>", **extra,
    )


def _numeros(g: gpd.GeoDataFrame, rotulos: dict | None = None) -> go.Scatter:
    """O número do cluster, e opcionalmente uma segunda linha, sobre cada polígono."""
    pontos = [_ponto_do_rotulo(geom) for geom in g.geometry]
    textos = []
    for c in g["cluster"]:
        extra = (rotulos or {}).get(c)
        textos.append(f"<b>{c}</b>" + (f"<br>{extra}" if extra else ""))
    return go.Scatter(
        x=[p.x for p in pontos], y=[p.y for p in pontos], mode="text", text=textos,
        textfont=dict(color=TINTA, size=15), showlegend=False, hoverinfo="skip",
    )


def _transparente(hexa: str, alfa: float) -> str:
    r, g, b = (int(hexa[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alfa})"


def _caixa(lat, lon, margem: float = 0.4) -> tuple[list, list]:
    return ([min(lon) - margem, max(lon) + margem], [min(lat) - margem, max(lat) + margem])


def _plano(figura: go.Figure, xlim, ylim, altura: int = 680) -> go.Figure:
    """Plano de longitude e latitude com a proporção certa, sem mapa de fundo."""
    lat_media = sum(ylim) / 2
    figura.update_layout(
        height=altura, margin=dict(l=55, r=10, t=10, b=35),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=FUNDO,
        font=dict(color=TINTA, size=14), dragmode="pan",
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(20,20,19,0.85)", font=dict(color=TINTA, size=13),
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
    )
    # Marcações em graus com a letra do hemisfério: -50 vira 50°O, -25 vira 25°S.
    passo = 1 if (xlim[1] - xlim[0]) < 8 else 5
    lons = [v for v in range(-70, -20, passo)]
    lats = [v for v in range(-40, -5, passo)]
    figura.update_xaxes(range=xlim, showgrid=True, gridcolor=GRADE, zeroline=False,
                        tickvals=lons, ticktext=[f"{-v}°O" for v in lons],
                        tickfont=dict(color=TINTA_FRACA, size=12), linecolor=GRADE)
    figura.update_yaxes(range=ylim, showgrid=True, gridcolor=GRADE, zeroline=False,
                        tickvals=lats, ticktext=[f"{-v}°S" for v in lats],
                        tickfont=dict(color=TINTA_FRACA, size=12), linecolor=GRADE,
                        scaleanchor="x", scaleratio=1 / math.cos(math.radians(lat_media)))
    return figura


def _janela_da_bacia(g: gpd.GeoDataFrame):
    minx, miny, maxx, maxy = g.total_bounds
    return _caixa([miny, maxy], [minx, maxx], margem=0.5)


# ── Figuras ───────────────────────────────────────────────────────────────────

def _texto_do_hover(cluster: int, n_total: dict, n_modelo: dict) -> str:
    if cluster in MODELADOS:
        regiao, papel = MODELADOS[cluster]
        return (f"<b>Cluster {cluster}</b> · {regiao}<br>{papel}<br>"
                f"{n_total.get(cluster, 0)} estações do INMET, "
                f"{n_modelo.get(cluster, 0)} com modelo")
    return (f"<b>Cluster {cluster}</b><br>{SITUACOES[situacao(cluster)][0]}<br>"
            f"{n_total.get(cluster, 0)} estações do INMET")


def mapa_da_bacia(g: gpd.GeoDataFrame, est: pd.DataFrame) -> go.Figure:
    """Os 14 clusters, coloridos pela situação, com o número em cima."""
    n_total = est.groupby("cluster").size().to_dict()
    n_modelo = est[est["modelada"]].groupby("cluster").size().to_dict()
    figura = go.Figure()
    mostradas = set()
    for linha in g.itertuples():
        chave = situacao(linha.cluster)
        nome, cor = SITUACOES[chave]
        figura.add_trace(_poligono(
            linha, cor, _texto_do_hover(linha.cluster, n_total, n_modelo),
            name=nome, legendgroup=chave, showlegend=chave not in mostradas,
        ))
        mostradas.add(chave)
    figura.add_trace(_numeros(g))
    return _plano(figura, *_janela_da_bacia(g))


def mapa_das_estacoes(g: gpd.GeoDataFrame, est: pd.DataFrame) -> go.Figure:
    """As estações do INMET, uma cor por cluster, com a contagem na legenda."""
    n_total = est.groupby("cluster").size().to_dict()
    n_modelo = est[est["modelada"]].groupby("cluster").size().to_dict()
    figura = go.Figure()
    for linha in g.itertuples():
        cor = COR_DO_CLUSTER[linha.cluster]
        xs, ys = _contorno(linha.geometry)
        figura.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself", fillcolor=_transparente(cor, 0.13),
            line=dict(color=_transparente(cor, 0.7), width=1.2), hoverinfo="skip",
            legendgroup=f"c{linha.cluster}", showlegend=False))
    for c in sorted(n_total):
        sub = est[est["cluster"] == c]
        figura.add_trace(go.Scatter(
            x=sub["lon"], y=sub["lat"], mode="markers", legendgroup=f"c{c}",
            name=f"Cluster {c} · {n_total[c]}",
            marker=dict(size=7, color=COR_DO_CLUSTER[c], line=dict(color="#141413", width=0.6)),
            text=sub["estacao"],
            hovertemplate=f"Estação %{{text}}<br>Cluster {c} · {n_total[c]} estações"
                          f"<extra></extra>"))
    figura.add_trace(_numeros(g))
    return _plano(figura, *_janela_da_bacia(g))


def vistas(cel: pd.DataFrame) -> dict:
    """Atalhos de zoom da figura da grade: a bacia inteira e cada cluster com ERA5."""
    out = {"Bacia inteira": None}
    for c in sorted(cel["cluster"].unique()):
        out[f"Cluster {c}"] = c
    return out


def mapa_da_grade(g: gpd.GeoDataFrame, cel: pd.DataFrame, est: pd.DataFrame,
                  vista: int | None = None) -> go.Figure:
    """A bacia inteira com os pontos do ERA5, por cluster, e as estações.

    Cada cluster tem a sua cor. O ponto com estação dentro é maior, com borda
    branca, e a estação é um ponto branco. `vista` aproxima a figura de um
    cluster; zoom e arrasto continuam livres.
    """
    est = est[est["modelada"]]
    figura = go.Figure()
    for linha in g.itertuples():
        cor = COR_DO_CLUSTER[linha.cluster]
        xs, ys = _contorno(linha.geometry)
        tem = linha.cluster in set(cel["cluster"])
        figura.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor=_transparente(cor, 0.12 if tem else 0.05),
            line=dict(color=_transparente(cor, 0.7 if tem else 0.3), width=1.2),
            hoverinfo="skip", legendgroup=f"c{linha.cluster}", showlegend=False))
    tamanho = 7 if vista is not None else 5
    for c in sorted(cel["cluster"].unique()):
        sub = cel[cel["cluster"] == c]
        sem, com = sub[~sub["com_estacao"]], sub[sub["com_estacao"]]
        cor = COR_DO_CLUSTER[c]
        figura.add_trace(go.Scatter(
            x=sem["lon"], y=sem["lat"], mode="markers", legendgroup=f"c{c}",
            name=f"Cluster {c} · {len(sub)} pontos", hoverinfo="skip",
            marker=dict(size=tamanho, color=cor, opacity=0.9)))
        figura.add_trace(go.Scatter(
            x=com["lon"], y=com["lat"], mode="markers", legendgroup=f"c{c}",
            showlegend=False, hoverinfo="skip",
            marker=dict(size=tamanho + 4, color=cor, line=dict(color="#ffffff", width=1.6))))
    _estacoes_do_cluster(figura, est, max(tamanho - 2, 3))
    # Só para a legenda: como se lê o ponto com estação.
    figura.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="Ponto com estação (maior, borda branca)",
        marker=dict(size=11, color="#6d7279", line=dict(color="#ffffff", width=1.6))))
    figura.add_trace(_numeros(g))
    if vista is None:
        xlim, ylim = _janela_da_bacia(g)
    else:
        sub = cel[cel["cluster"] == vista]
        xlim, ylim = _caixa(list(sub["lat"]), list(sub["lon"]), margem=0.6)
    return _plano(figura, xlim, ylim)


# ── Figuras de um cluster só (seção Experimento) ──────────────────────────────

def _distancia_km(lat0, lon0, lat, lon):
    """Distância na esfera (haversine), em km."""
    r = 6371.0
    p0, p1 = math.radians(lat0), [math.radians(v) for v in lat]
    dl = [math.radians(v - lon0) for v in lon]
    out = []
    for p, d in zip(p1, dl):
        a = math.sin((p - p0) / 2) ** 2 + math.cos(p0) * math.cos(p) * math.sin(d / 2) ** 2
        out.append(2 * r * math.asin(math.sqrt(a)))
    return out


NOME_ESTACAO = "Estação com rajada medida até 2024"
NOME_ESTACAO_NOVA = "Estação com dados só a partir de 2025"
COR_ESTACAO_NOVA = "#8d939b"


def _estacoes_do_cluster(figura, e, tamanho: int):
    """Estações do cluster: brancas as com série até 2024, cinza as que só começaram em 2025."""
    com, sem = e[e["com_serie"]], e[~e["com_serie"]]
    figura.add_trace(go.Scatter(
        x=com["lon"], y=com["lat"], mode="markers", name=NOME_ESTACAO, text=com["estacao"],
        hovertemplate="Estação %{text}<extra></extra>",
        marker=dict(size=tamanho, color=COR_ESTACAO)))
    if len(sem):
        figura.add_trace(go.Scatter(
            x=sem["lon"], y=sem["lat"], mode="markers", name=NOME_ESTACAO_NOVA, text=sem["estacao"],
            hovertemplate="Estação %{text}<extra></extra>",
            marker=dict(size=tamanho, color=COR_ESTACAO_NOVA)))


def mapa_do_cluster(g: gpd.GeoDataFrame, cel: pd.DataFrame, est: pd.DataFrame,
                    cluster: int) -> go.Figure:
    """Um cluster de perto: contorno, pontos do ERA5 e estações do INMET."""
    cor = COR_DO_CLUSTER[cluster]
    linha = next(g[g["cluster"] == cluster].itertuples())
    sub = cel[cel["cluster"] == cluster]
    e = est[est["modelada"] & (est["cluster"] == cluster)]
    xs, ys = _contorno(linha.geometry)
    figura = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself", fillcolor=_transparente(cor, 0.12),
        line=dict(color=_transparente(cor, 0.7), width=1.2), hoverinfo="skip", showlegend=False))
    sem, com = sub[~sub["com_estacao"]], sub[sub["com_estacao"]]
    figura.add_trace(go.Scatter(
        x=sem["lon"], y=sem["lat"], mode="markers", name="Ponto do ERA5", hoverinfo="skip",
        marker=dict(size=8, color=cor, opacity=0.9)))
    figura.add_trace(go.Scatter(
        x=com["lon"], y=com["lat"], mode="markers", name="Ponto com estação", hoverinfo="skip",
        marker=dict(size=13, color=cor, line=dict(color="#ffffff", width=1.6))))
    _estacoes_do_cluster(figura, e, 6)
    xlim, ylim = _caixa(list(sub["lat"]), list(sub["lon"]), margem=0.5)
    return _plano(figura, xlim, ylim, altura=520)


def mapa_do_entorno(g: gpd.GeoDataFrame, est: pd.DataFrame, cluster: int = 3) -> go.Figure:
    """O entorno de uma estação: os pontos do ERA5 a menos de 75 km e a menos de 250 km."""
    cor = COR_DO_CLUSTER[cluster]
    linha = next(g[g["cluster"] == cluster].itertuples())
    e = est[est["modelada"] & est["com_serie"] & (est["cluster"] == cluster)].reset_index(drop=True)
    centro = _ponto_do_rotulo(linha.geometry)
    d = _distancia_km(centro.y, centro.x, list(e["lat"]), list(e["lon"]))
    foco = e.iloc[int(min(range(len(d)), key=d.__getitem__))]

    lat0, lon0 = float(foco["lat"]), float(foco["lon"])
    folga = 250 / 111.19 + 0.3
    lats = [round(v * PASSO, 2) for v in range(int((lat0 - folga) / PASSO), int((lat0 + folga) / PASSO) + 2)]
    lons = [round(v * PASSO, 2) for v in range(int((lon0 - folga) / PASSO), int((lon0 + folga) / PASSO) + 2)]
    pts = [(la, lo) for la in lats for lo in lons]
    dist = _distancia_km(lat0, lon0, [p[0] for p in pts], [p[1] for p in pts])
    perto = [(p, k) for p, k in zip(pts, dist) if k <= 75]
    meio = [(p, k) for p, k in zip(pts, dist) if 75 < k <= 250]

    def circulo(raio):
        ang = [i * 2 * math.pi / 120 for i in range(121)]
        return ([lon0 + raio / (111.19 * math.cos(math.radians(lat0))) * math.cos(a) for a in ang],
                [lat0 + raio / 111.19 * math.sin(a) for a in ang])

    xs, ys = _contorno(linha.geometry)
    figura = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself", fillcolor=_transparente(cor, 0.10),
        line=dict(color=_transparente(cor, 0.6), width=1.2), hoverinfo="skip", showlegend=False))
    for raio, tom in ((250, "#6f8fb3"), (75, "#E69F00")):
        cx, cy = circulo(raio)
        figura.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name=f"Raio de {raio} km",
                                    line=dict(color=tom, width=1.6, dash="dash"), hoverinfo="skip"))
    figura.add_trace(go.Scatter(
        x=[p[0][1] for p in meio], y=[p[0][0] for p in meio], mode="markers",
        name="Pontos do ERA5 entre 75 e 250 km", hoverinfo="skip",
        marker=dict(size=6, color="#6f8fb3", opacity=0.85)))
    figura.add_trace(go.Scatter(
        x=[p[0][1] for p in perto], y=[p[0][0] for p in perto], mode="markers",
        name="Pontos do ERA5 a menos de 75 km", hoverinfo="skip",
        marker=dict(size=8, color="#E69F00")))
    outras = e[e["estacao"] != foco["estacao"]]
    figura.add_trace(go.Scatter(
        x=outras["lon"], y=outras["lat"], mode="markers", name="Outras estações do estudo",
        text=outras["estacao"], hovertemplate="Estação %{text}<extra></extra>",
        marker=dict(size=6, color=COR_ESTACAO)))
    figura.add_trace(go.Scatter(
        x=[lon0], y=[lat0], mode="markers", name=f"Estação {foco['estacao']}",
        marker=dict(size=13, color=COR_ESTACAO, line=dict(color="#E69F00", width=3)),
        hovertemplate=f"Estação {foco['estacao']}<extra></extra>"))
    cx, cy = circulo(250)
    xlim, ylim = _caixa(cy, cx, margem=0.2)
    return _plano(figura, xlim, ylim, altura=620)


def distancia_entre_clusters(g: gpd.GeoDataFrame, a: int, b: int) -> float:
    """Distância em km, arredondada para a centena, entre os pontos centrais de dois clusters."""
    pa = _ponto_do_rotulo(g[g["cluster"] == a].geometry.iloc[0])
    pb = _ponto_do_rotulo(g[g["cluster"] == b].geometry.iloc[0])
    return round(_distancia_km(pa.y, pa.x, [pb.y], [pb.x])[0], -2)
