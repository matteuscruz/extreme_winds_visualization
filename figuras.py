"""
Figuras da narrativa.

Existe porque a primeira versão das seções explicava por escrito o que um
gráfico mostra melhor — 81 blocos de texto contra 3 figuras, medido. Aqui
ficam as figuras que substituem parágrafo.

Cor segue o trabalho que ela faz, não o gosto:
- **divergente** (duas cores + cinza no meio) quando o sinal importa, como
  num viés que pode ser para mais ou para menos;
- **categórica** em ordem fixa quando a cor é identidade;
- **de estado** (reservada) quando a cor diz se algo está certo, suspeito ou
  quebrado — e nesse caso nunca vai sozinha: sempre com rótulo escrito.

Os palettes usados foram validados por script (banda de luminosidade, croma,
separação sob daltonismo, piso de visão normal e contraste contra o fundo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from theme import diff_colorscale

# Tinta para texto — nunca a cor da série.
TINTA = "#0b0b0b"
TINTA_FRACA = "#52514e"
SUPERFICIE = "#fcfcfb"

# Paleta de estado (reservada; nunca reaproveitada como "série 4").
ESTADO = {
    "ok": "#0ca30c",
    "atencao": "#fab219",
    "grave": "#ec835a",
    "critico": "#d91f11",
    "neutro": "#9a9894",
}

GRAFICO = {"responsive": True}

# Rampa sequencial de uma cor só (azul), clara -> escura. Usada quando o dado
# não muda de sinal e a cor precisa dizer "quanto", não "para que lado".
SEQUENCIAL_AZUL = [
    [0.00, "#0d366b"], [0.20, "#184f95"], [0.40, "#256abf"],
    [0.60, "#3987e5"], [0.80, "#86b6ef"], [1.00, "#cde2fb"],
]


def _base(figura: go.Figure, altura: int = 380, titulo_y: str = "",
          esquerda: int = 70, baixo: int = 50, topo: int = 60) -> go.Figure:
    """Margens generosas por padrão: renderizar e olhar mostrou rótulo de eixo
    cortado e título de eixo por cima dos valores em todas as figuras."""
    figura.update_layout(
        height=altura,
        margin=dict(l=esquerda, r=30, t=topo, b=baixo),
        plot_bgcolor=SUPERFICIE,
        paper_bgcolor=SUPERFICIE,
        font=dict(color=TINTA),
        yaxis_title=titulo_y,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    figura.update_xaxes(showgrid=False, linecolor="#d8d6d1")
    figura.update_yaxes(gridcolor="#eceae5", zerolinecolor="#d8d6d1")
    return figura


# ── Seção 1 ───────────────────────────────────────────────────────────────────

def mapa_do_erro(sdf: pd.DataFrame, titulo: str) -> go.Figure:
    """Onde o ERA5 erra mais, estação por estação.

    O valor é um viés com sinal, então a escala é divergente com cinza no
    zero — assim "erra para menos" e "erra para mais" não viram a mesma cor.
    """
    figura = go.Figure()
    if sdf.empty:
        return _base(figura)

    minimo, maximo = float(sdf["value"].min()), float(sdf["value"].max())
    mesmo_sinal = minimo * maximo > 0
    if mesmo_sinal:
        # Todos os valores caem do mesmo lado do zero: isso é magnitude, não
        # polaridade. Uma escala divergente simétrica jogaria fora metade do
        # intervalo de cor e comprimiria o dado todo num tom só — aqui vai um
        # degradê de uma cor só, cobrindo a faixa que existe de verdade.
        escala = SEQUENCIAL_AZUL if minimo < 0 else SEQUENCIAL_AZUL[::-1]
        faixa = (minimo, maximo)
    else:
        limite = float(np.nanmax(np.abs(sdf["value"]))) or 1.0
        escala, faixa = diff_colorscale(1.6), (-limite, limite)

    figura.add_trace(go.Scattermap(
        lat=sdf["latitude"], lon=sdf["longitude"],
        mode="markers",
        marker=dict(
            size=11, color=sdf["value"], colorscale=escala,
            cmin=faixa[0], cmax=faixa[1], opacity=0.9,
            colorbar=dict(title=dict(text="m/s", side="right"), thickness=12),
        ),
        customdata=np.stack([sdf["estacao"], sdf["value"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.1f} m/s<extra></extra>",
        name="Estações",
    ))
    figura.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=float(sdf["latitude"].mean()), lon=float(sdf["longitude"].mean())),
            zoom=4.1,
        ),
        height=460, margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text=titulo, font=dict(size=14, color=TINTA)),
        paper_bgcolor=SUPERFICIE, showlegend=False,
    )
    return figura


def barras_da_interpolacao(dados: pd.DataFrame) -> go.Figure:
    """Viés de cada método de interpolação no pico anual.

    Duas categorias com significado de estado — a linhagem que estava em
    produção e as alternativas testadas — então a cor vem da paleta de
    estado e vem acompanhada de legenda e do número escrito na barra.
    """
    figura = go.Figure()
    if dados.empty:
        return _base(figura)
    dados = dados.sort_values("bias")
    for em_producao, rotulo, cor in (
        (True, "Linhagem em produção", ESTADO["grave"]),
        (False, "Alternativas avaliadas", ESTADO["neutro"]),
    ):
        recorte = dados[dados["em_producao"] == em_producao]
        if recorte.empty:
            continue
        figura.add_bar(
            name=rotulo, y=recorte["Método"], x=recorte["bias"],
            orientation="h", marker_color=cor,
            text=[f"{v:+.1f}" for v in recorte["bias"]],
            textposition="outside", textfont=dict(color=TINTA_FRACA, size=11),
            hovertemplate="%{y}<br>viés %{x:.2f} m/s<extra></extra>",
        )
    figura.add_vline(x=0, line_width=1, line_color="#d8d6d1")
    figura = _base(figura, altura=430, esquerda=200, baixo=70)
    figura.update_layout(
        xaxis_title="Viés no pico anual (m/s) — negativo = subestima",
        yaxis_title="", bargap=0.35,
        yaxis=dict(categoryorder="array", categoryarray=list(dados["Método"])),
    )
    figura.update_xaxes(title_standoff=20, automargin=True)
    figura.update_yaxes(automargin=True)
    return figura


def esquema_do_teto() -> go.Figure:
    """Esquema: por que espalhar o erro dos vizinhos não alcança o extremo.

    Não são dados medidos — é o mecanismo desenhado. Uma média ponderada
    fica sempre dentro da faixa dos valores que entraram nela.
    """
    vizinhos = [2.1, 3.4, 2.8, 4.0, 3.1]
    x = list(range(1, len(vizinhos) + 1))
    piso, teto = min(vizinhos), max(vizinhos)
    extremo_real = 9.2

    figura = go.Figure()
    figura.add_hrect(
        y0=piso, y1=teto, fillcolor="#2a78d6", opacity=0.10, line_width=0,
        annotation_text="tudo que a média ponderada pode devolver",
        annotation_position="bottom left",
        annotation_font=dict(size=11, color=TINTA_FRACA),
    )
    figura.add_trace(go.Scatter(
        x=x, y=vizinhos, mode="markers", name="Erro medido nas estações vizinhas",
        marker=dict(size=13, color="#2a78d6", line=dict(width=2, color=SUPERFICIE)),
        hovertemplate="vizinha %{x}: %{y:.1f} m/s<extra></extra>",
    ))
    figura.add_trace(go.Scatter(
        x=[len(vizinhos) + 1.4], y=[extremo_real], mode="markers+text",
        name="Erro real no dia de vendaval",
        marker=dict(size=17, color=ESTADO["critico"], symbol="diamond",
                    line=dict(width=2, color=SUPERFICIE)),
        text=["fora de alcance"], textposition="middle right",
        textfont=dict(color=TINTA_FRACA, size=11),
        hovertemplate="o extremo que ninguém por perto mediu<extra></extra>",
    ))
    # A seta sobe do teto até o extremo: o vão é o que o método não alcança.
    figura.add_annotation(
        x=len(vizinhos) + 1.4, y=extremo_real - 0.4,
        ax=len(vizinhos) + 1.4, ay=teto + 0.15,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor=ESTADO["critico"],
    )
    figura = _base(figura, altura=390, titulo_y="Erro do ERA5 (m/s)",
                   esquerda=82, baixo=54, topo=86)
    figura.add_annotation(
        xref="paper", yref="paper", x=0, y=1.26, showarrow=False, xanchor="left",
        text="Esquema do mecanismo — não são dados medidos",
        font=dict(size=11, color=TINTA_FRACA),
    )
    figura.update_layout(
        xaxis=dict(showticklabels=False, title="", range=[0.3, len(vizinhos) + 2.9]),
        yaxis=dict(range=[0, extremo_real + 2.2]),
    )
    figura.update_yaxes(automargin=True, title_standoff=16)
    return figura


# ── Seção 2 ───────────────────────────────────────────────────────────────────

def grade_do_experimento(celulas: list[dict]) -> go.Figure:
    """A matriz do experimento como grade, com o estado de cada execução.

    Substitui uma tabela mais um aviso: dá para ver de relance quais
    combinações existem, quais estão íntegras e qual está comprometida.
    """
    figura = go.Figure()
    if not celulas:
        return _base(figura)
    df = pd.DataFrame(celulas)
    cor_por_estado = {
        "ok": ESTADO["ok"], "divergente": ESTADO["critico"],
        "parcial": ESTADO["atencao"], "ausente": "#e6e4df",
    }
    for estado, rotulo in (
        ("ok", "Execução íntegra"),
        ("parcial", "Cobertura parcial"),
        ("divergente", "Não corresponde ao desenho"),
        ("ausente", "Não executada"),
    ):
        recorte = df[df["estado"] == estado]
        if recorte.empty:
            continue
        figura.add_trace(go.Scatter(
            x=recorte["coluna"], y=recorte["linha"], mode="markers+text",
            name=rotulo,
            marker=dict(size=46, symbol="square", color=cor_por_estado[estado],
                        line=dict(width=3, color=SUPERFICIE)),
            text=recorte["sigla"], textposition="middle center",
            # Tinta escura sobre fundo claro: branco sobre âmbar não alcança
            # contraste legível.
            textfont=dict(
                color=TINTA if estado in ("parcial", "ausente") else SUPERFICIE,
                size=11),
            hovertext=recorte["detalhe"], hoverinfo="text",
        ))
    ordem_colunas = ["Sem extremos inventados", "Com extremos inventados"]
    ordem_linhas = list(dict.fromkeys(df["linha"]))
    figura = _base(figura, altura=390, esquerda=230, baixo=60, topo=66)
    figura.update_layout(
        xaxis=dict(title="", tickfont=dict(size=12), type="category",
                   categoryorder="array", categoryarray=ordem_colunas,
                   range=[-0.6, len(ordem_colunas) - 0.4]),
        yaxis=dict(title="", tickfont=dict(size=12), type="category",
                   categoryorder="array", categoryarray=ordem_linhas,
                   range=[len(ordem_linhas) - 0.4, -0.6]),
    )
    figura.update_xaxes(showgrid=False, automargin=True)
    figura.update_yaxes(showgrid=False, automargin=True)
    return figura


# ── Seção 4 ───────────────────────────────────────────────────────────────────

def linha_do_tempo(periodos: dict) -> go.Figure:
    """Treino, validação e teste numa régua só — deixa ver de relance que não
    há sobreposição."""
    faixas = [
        ("Treino", periodos.get("treino"), "#2a78d6", "o modelo aprendeu aqui"),
        ("Validação", periodos.get("validacao"), "#eda100", "ajuste das escolhas"),
        ("Teste", periodos.get("teste"), ESTADO["ok"], "nunca visto — é daqui que saem os resultados"),
    ]
    figura = go.Figure()
    for i, (nome, intervalo, cor, nota) in enumerate(faixas):
        if not intervalo:
            continue
        inicio, fim = pd.Timestamp(intervalo[0]), pd.Timestamp(intervalo[1])
        figura.add_trace(go.Scatter(
            x=[inicio, fim], y=[i, i], mode="lines",
            line=dict(color=cor, width=18), name=nome,
            hovertemplate=f"<b>{nome}</b><br>{inicio:%Y}–{fim:%Y}<br>{nota}<extra></extra>",
        ))
        figura.add_annotation(
            x=inicio + (fim - inicio) / 2, y=i, text=f"{nome}  {inicio:%Y}–{fim:%Y}",
            showarrow=False, font=dict(color=SUPERFICIE, size=12), yshift=0,
        )
    figura = _base(figura, altura=190)
    figura.update_layout(
        showlegend=False,
        yaxis=dict(showticklabels=False, range=[-0.8, len(faixas) - 0.2], title=""),
        xaxis=dict(title=""),
    )
    figura.update_yaxes(showgrid=False)
    return figura
