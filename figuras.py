"""
Figuras da narrativa.

Existe porque a primeira versão das seções explicava por escrito o que um
gráfico mostra melhor, 81 blocos de texto contra 3 figuras, medido. Aqui
ficam as figuras que substituem parágrafo.

Cor segue o trabalho que ela faz, não o gosto:
- **divergente** (duas cores + cinza no meio) quando o sinal importa, como
  num viés que pode ser para mais ou para menos;
- **categórica** em ordem fixa quando a cor é identidade;
- **de estado** (reservada) quando a cor diz se algo está certo, suspeito ou
  quebrado, e nesse caso nunca vai sozinha: sempre com rótulo escrito.

Os palettes usados foram validados por script (banda de luminosidade, croma,
separação sob daltonismo, piso de visão normal e contraste contra o fundo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from theme import diff_colorscale

# Tinta para texto, nunca a cor da série.
TINTA = "#f2f0eb"
TINTA_FRACA = "#b4b1a9"
SUPERFICIE = "#141413"
LINHA = "#3a3936"
GRADE = "#2a2927"

# Paleta de estado (reservada; nunca reaproveitada como "série 4").
ESTADO = {
    "ok": "#0ca30c", "atencao": "#fab219", "grave": "#ec835a", "critico": "#d91f11", "neutro": "#9a9894",
}

GRAFICO = {"responsive": True}

# Rampas sequenciais de uma cor só, clara -> escura. Usadas quando o dado não
# muda de sinal e a cor precisa dizer "quanto", não "para que lado". Piso
# levantado (nunca abaixo da luminância que ainda contrasta com o basemap
# `carto-darkmatter`, quase preto) e VENTO trocado de azul pra verde-azulado:
# o azul não se distinguia bem do próprio tom do basemap escuro.
SEQUENCIAL_FALTA = [
    [0.00, "#d95926"], [0.25, "#e37a49"], [0.50, "#f08a4b"],
    [0.75, "#f9ad7c"], [1.00, "#ffc9a0"],
]
SEQUENCIAL_VENTO = [
    [0.00, "#0e9488"], [0.25, "#14b8a6"], [0.50, "#5eead4"],
    [0.75, "#99f6e4"], [1.00, "#ecfeff"],
]
SEQUENCIAL_AZUL = [
    [0.00, "#dce9fb"], [0.20, "#86b6ef"], [0.40, "#3987e5"],
    [0.60, "#256abf"], [0.80, "#184f95"], [1.00, "#10365f"],
]


def _base(figura: go.Figure, altura: int = 380, titulo_y: str = "",
          esquerda: int = 70, baixo: int = 50, topo: int = 60) -> go.Figure:
    """Margens generosas por padrão: renderizar e olhar mostrou rótulo de eixo
    cortado e título de eixo por cima dos valores em todas as figuras."""
    # `template` explícito: sem ele a figura herda o tema do Streamlit, e num
    # tema escuro o texto dos eixos sai cinza claro sobre este fundo claro,
    # ilegível. Cor de fonte é fixada em cada elemento pelo mesmo motivo.
    figura.update_layout(
        template="plotly_dark",
        height=altura,
        margin=dict(l=esquerda, r=30, t=topo, b=baixo),
        plot_bgcolor=SUPERFICIE,
        paper_bgcolor=SUPERFICIE,
        font=dict(color=TINTA, size=15),
        title=dict(text="", font=dict(color=TINTA)),
        yaxis_title=titulo_y,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=TINTA, size=14)),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
    )
    figura.update_xaxes(showgrid=False, linecolor=LINHA,
                        tickfont=dict(color=TINTA, size=14),
                        title_font=dict(color=TINTA, size=15))
    figura.update_yaxes(gridcolor=GRADE, zerolinecolor=LINHA,
                        tickfont=dict(color=TINTA, size=14),
                        title_font=dict(color=TINTA, size=15))
    return figura


# ── Seção 1 ───────────────────────────────────────────────────────────────────

def mapa_do_erro(sdf: pd.DataFrame, titulo: str) -> go.Figure:
    """Onde o ERA5 erra mais, estação por estação.

    O valor é um viés com sinal, então a escala é divergente com cinza no
    zero, assim "erra para menos" e "erra para mais" não viram a mesma cor.
    """
    figura = go.Figure()
    if sdf.empty:
        return _base(figura)

    minimo, maximo = float(sdf["value"].min()), float(sdf["value"].max())
    mesmo_sinal = minimo * maximo > 0
    if mesmo_sinal:
        # Todos os valores caem do mesmo lado do zero: isso é magnitude, não
        # polaridade. Uma escala divergente simétrica jogaria fora metade do
        # intervalo de cor e comprimiria o dado todo num tom só, aqui vai um
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
            style="carto-darkmatter",
            center=dict(lat=float(sdf["latitude"].mean()), lon=float(sdf["longitude"].mean())),
            zoom=4.1,
        ),
        height=460, margin=dict(l=0, r=0, t=42, b=0),
        title=dict(text=titulo, font=dict(size=14, color=TINTA)),
        paper_bgcolor=SUPERFICIE, showlegend=False,
        font=dict(color=TINTA),
        hoverlabel=dict(font=dict(color=TINTA), bgcolor="#1f1f1e"),
    )
    return figura


def barras_da_interpolacao(dados: pd.DataFrame) -> go.Figure:
    """Viés de cada método de interpolação no pico anual.

    Duas categorias com significado de estado, a linhagem que estava em
    produção e as alternativas testadas, então a cor vem da paleta de
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
    figura.add_vline(x=0, line_width=1, line_color=LINHA)
    figura = _base(figura, altura=430, esquerda=200, baixo=70)
    figura.update_layout(
        xaxis_title="Viés no pico anual (m/s), negativo = subestima",
        yaxis_title="", bargap=0.35,
        yaxis=dict(categoryorder="array", categoryarray=list(dados["Método"])),
    )
    figura.update_xaxes(title_standoff=20, automargin=True)
    figura.update_yaxes(automargin=True)
    return figura


# ── Seção 2 ───────────────────────────────────────────────────────────────────

def tabela_do_experimento(celulas: list[dict]) -> str | None:
    """A matriz do experimento como tabela HTML colorida, texto real do navegador.

    Era um `go.Scatter` com quadrados e texto desenhados no canvas — ficava
    pequeno e borrado dependendo da densidade de pixels da tela de quem olha.
    Tabela HTML não tem esse problema: o texto é texto de verdade, do tamanho
    que o navegador (ou o zoom de quem lê) decidir. HTML montado à mão, não
    `DataFrame.style` — o `Styler` do pandas importa `matplotlib` por baixo, e
    o `matplotlib` desta máquina está quebrado (ABI do NumPy).
    """
    if not celulas:
        return None
    ordem_linhas = list(dict.fromkeys(c["linha"] for c in celulas))
    ordem_colunas = ["Sem extremos inventados", "Com extremos inventados"]
    por_posicao = {(c["linha"], c["coluna"]): c for c in celulas}

    cor_fundo = {"ok": ESTADO["ok"], "divergente": ESTADO["critico"],
                 "parcial": ESTADO["atencao"], "ausente": "#35342f"}
    cor_texto = {"ok": SUPERFICIE, "divergente": TINTA,
                 "parcial": SUPERFICIE, "ausente": TINTA}

    cabecalho = "".join(
        f"<th style='padding:14px 22px; color:{TINTA}; font-weight:600; "
        f"font-size:1.2rem; text-align:center;'>{coluna}</th>"
        for coluna in ordem_colunas
    )
    linhas_html = [f"<tr><th></th>{cabecalho}</tr>"]
    for linha in ordem_linhas:
        celulas_html = []
        for coluna in ordem_colunas:
            cel = por_posicao.get((linha, coluna))
            estado = cel["estado"] if cel else "ausente"
            texto = cel["sigla"] if cel else ""
            celulas_html.append(
                f"<td style='background:{cor_fundo[estado]}; color:{cor_texto[estado]}; "
                "text-align:center; font-weight:600; font-size:1.35rem; padding:26px; "
                f"border-radius:6px;'>{texto}</td>"
            )
        linhas_html.append(
            f"<tr><td style='color:{TINTA}; padding:12px 22px; white-space:nowrap; "
            f"font-size:1.15rem;'>{linha}</td>" + "".join(celulas_html) + "</tr>"
        )
    return (
        "<table style='border-collapse:separate; border-spacing:10px; width:100%;'>"
        + "".join(linhas_html) + "</table>"
    )


# ── Seção 3 ───────────────────────────────────────────────────────────────────

def barras_por_metrica(quadro: pd.DataFrame, metrica: str, nome_metrica: str,
                        cores: dict) -> go.Figure:
    """Um traço por abordagem, uma barra por configuração, valor escrito nela.

    Antes vivia solto em `narrativa.py` e nunca passava por `_base()` — sem
    `template="plotly_dark"` a figura herdava o tema do Streamlit e os eixos
    saíam ilegíveis, o mesmo bug que `_base()` existe para evitar em todo o
    resto do painel. Também não tinha valor escrito na barra, só no hover.
    """
    figura = go.Figure()
    if quadro.empty:
        return _base(figura)
    for pipeline, grupo in quadro.groupby("pipeline"):
        grupo = grupo.sort_values(metrica, key=abs)
        figura.add_bar(
            name=grupo["Abordagem"].iloc[0], x=grupo["Configuração"], y=grupo[metrica],
            marker_color=cores.get(pipeline),
            text=[f"{v:.2f}" for v in grupo[metrica]],
            textposition="outside", textfont=dict(color=TINTA, size=14),
            hovertemplate="%{x}<br>%{y:.3f}<extra>%{fullData.name}</extra>",
        )
    figura = _base(figura, altura=530, esquerda=60, baixo=110, titulo_y=nome_metrica)
    figura.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.08,
        uniformtext=dict(mode="show"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    figura.update_xaxes(tickangle=-20, automargin=True)
    figura.update_yaxes(automargin=True)
    return figura


# ── Seção 4 ───────────────────────────────────────────────────────────────────

def linha_do_tempo(periodos: dict) -> go.Figure:
    """Treino, validação e teste numa régua só, deixa ver de relance que não
    há sobreposição."""
    faixas = [
        ("Treino", periodos.get("treino"), "#3987e5", "o modelo aprendeu aqui"),
        ("Validação", periodos.get("validacao"), "#c98500", "ajuste das escolhas"),
        ("Teste", periodos.get("teste"), ESTADO["ok"], "nunca visto, é daqui que saem os resultados"),
    ]
    figura = go.Figure()
    for i, (nome, intervalo, cor, nota) in enumerate(faixas):
        if not intervalo:
            continue
        inicio, fim = pd.Timestamp(intervalo[0]), pd.Timestamp(intervalo[1])
        figura.add_trace(go.Scatter(
            x=[inicio, fim], y=[i, i], mode="lines",
            line=dict(color=cor, width=22), name=nome,
            hovertemplate=f"<b>{nome}</b><br>{inicio:%Y}-{fim:%Y}<br>{nota}<extra></extra>",
        ))
        figura.add_annotation(
            x=inicio + (fim - inicio) / 2, y=i, text=f"{nome}  {inicio:%Y}-{fim:%Y}",
            showarrow=False, font=dict(color=SUPERFICIE, size=14), yshift=0,
        )
    figura = _base(figura, altura=230)
    figura.update_layout(
        showlegend=False,
        yaxis=dict(showticklabels=False, range=[-0.8, len(faixas) - 0.2], title=""),
        xaxis=dict(title=""),
    )
    figura.update_yaxes(showgrid=False)
    return figura


def barras_por_trimestre(dados: pd.DataFrame, cores: dict, nomes: dict) -> go.Figure:
    """Viés no extremo por trimestre, uma barra por abordagem, valor na barra.

    Antes vivia solto em `narrativa.py`, sem passar por `_base()` — mesmo bug
    de tema (e mesma falta de rótulo na barra) do gráfico da seção 3.
    """
    figura = go.Figure()
    if dados.empty:
        return _base(figura)
    ordem_estacoes = ["DJF", "MAM", "JJA", "SON"]
    for pipeline, grupo in dados.groupby("pipeline"):
        grupo = grupo.set_index("season").reindex(ordem_estacoes).reset_index()
        figura.add_bar(
            name=nomes.get(pipeline, pipeline), x=grupo["season"], y=grupo["Bias_P90"],
            marker_color=cores.get(pipeline),
            text=[f"{v:.2f}" if pd.notna(v) else "" for v in grupo["Bias_P90"]],
            textposition="outside", textfont=dict(color=TINTA, size=14),
            hovertemplate="%{x}<br>%{y:.2f} m/s<extra>%{fullData.name}</extra>",
        )
    figura = _base(figura, altura=440, esquerda=70, baixo=60,
                   titulo_y="Viés no extremo (m/s)")
    figura.update_layout(
        barmode="group", bargap=0.3, bargroupgap=0.08,
        uniformtext=dict(mode="show"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    figura.update_xaxes(automargin=True)
    figura.update_yaxes(automargin=True)
    return figura


def barras_por_area(recorte: pd.DataFrame, cor: str) -> go.Figure:
    """Acerto médio do mês e o pior mês, uma barra + um marcador por área.

    Antes vivia solto em `narrativa.py`, sem `_base()` e sem valor escrito —
    mesmo bug das seções 3 e 4, corrigido aqui do mesmo jeito.
    """
    figura = go.Figure()
    if recorte.empty:
        return _base(figura)
    areas = [f"Área {a}" for a in recorte["area"]]
    figura.add_bar(
        name="Mês médio", x=areas, y=recorte["R2_mes_medio"], marker_color=cor,
        text=[f"{v:.2f}" for v in recorte["R2_mes_medio"]],
        textposition="outside", textfont=dict(color=TINTA, size=14),
        hovertemplate="%{x}<br>mês médio: %{y:.3f}<extra></extra>",
    )
    figura.add_trace(go.Scatter(
        name="Pior mês", x=areas, y=recorte["R2_mes_pior"], mode="markers+text",
        marker=dict(size=14, symbol="diamond", color=ESTADO["critico"],
                    line=dict(width=1, color=SUPERFICIE)),
        text=[f"{v:.2f}" for v in recorte["R2_mes_pior"]],
        textposition="bottom center", textfont=dict(color=TINTA, size=13),
        hovertemplate="%{x}<br>pior mês: %{y:.3f}<extra></extra>",
    ))
    figura = _base(figura, altura=440, esquerda=70, baixo=60,
                   titulo_y="R² na janela mensal")
    figura.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        uniformtext=dict(mode="show"),
    )
    figura.update_xaxes(automargin=True)
    figura.update_yaxes(automargin=True)
    return figura


def metodos_lado_a_lado(dados95: pd.DataFrame, dados99: pd.DataFrame) -> go.Figure:
    """Cada método de interpolação nos dois níveis de extremo, no mesmo eixo."""
    figura = go.Figure()
    if dados99.empty:
        return _base(figura)
    ordem = list(dados99.sort_values("bias")["Método"])
    for dados, rotulo, cor in ((dados95, "Dia de vento forte (P95)", "#5eead4"),
                               (dados99, "Dia de vendaval (P99)", "#0e9488")):
        if dados.empty:
            continue
        d = dados.set_index("Método").reindex(ordem).reset_index()
        figura.add_bar(
            name=rotulo, y=d["Método"], x=d["bias"], orientation="h",
            marker_color=cor,
            text=[f"<b>{v:+.1f}</b>" for v in d["bias"]],
            textposition="outside", textfont=dict(color=TINTA, size=15),
            hovertemplate="%{y}<br>" + rotulo + ": %{x:.2f} m/s<extra></extra>",
        )
    figura.add_vline(x=0, line_width=1, line_color=LINHA)
    figura = _base(figura, altura=720, esquerda=210, baixo=80)
    figura.update_layout(
        barmode="group", bargap=0.4, bargroupgap=0.15,
        uniformtext=dict(mode="show"),
        xaxis_title="Viés no extremo (m/s), negativo = subestima",
        yaxis=dict(categoryorder="array", categoryarray=ordem), yaxis_title="",
    )
    figura.update_xaxes(title_standoff=20, automargin=True)
    figura.update_yaxes(automargin=True)
    return figura


def valores_por_nivel(niveis: pd.DataFrame) -> go.Figure:
    """O que o instrumento mediu e o que o ERA5 estimou, lado a lado.

    O vao entre as duas barras e a diferenca. Mostrar os dois valores em vez
    de so a subtracao evita pedir que o leitor confie num numero que ele nao
    consegue conferir na tela.
    """
    figura = go.Figure()
    if niveis.empty:
        return _base(figura)
    figura.add_bar(
        name="Medido pelo INMET", x=niveis["Nível"], y=niveis["inmet_medio"],
        marker_color="#ecfeff",
        text=[f"{v:.1f}" for v in niveis["inmet_medio"]],
        textposition="outside", textfont=dict(color=TINTA, size=15),
        hovertemplate="%{x}<br>INMET: %{y:.1f} m/s<extra></extra>",
    )
    figura.add_bar(
        name="Estimado pelo ERA5", x=niveis["Nível"], y=niveis["era5_medio"],
        marker_color="#0e9488",
        text=[f"{v:.1f}" for v in niveis["era5_medio"]],
        textposition="outside", textfont=dict(color=TINTA, size=15),
        hovertemplate="%{x}<br>ERA5: %{y:.1f} m/s<extra></extra>",
    )
    topo_do_eixo = float(niveis["inmet_medio"].max()) * 1.42
    for _, linha in niveis.iterrows():
        figura.add_annotation(
            x=linha["Nível"], y=topo_do_eixo * 0.94,
            text=f"faltam {abs(linha['vies_medio']):.1f} m/s",
            showarrow=False, font=dict(color=ESTADO["critico"], size=16),
            bgcolor=SUPERFICIE, borderpad=3,
        )
    figura = _base(figura, altura=480, titulo_y="Rajada de vento (m/s)",
                   esquerda=84, baixo=58, topo=68)
    figura.update_layout(
        barmode="group", bargap=0.34,
        yaxis=dict(range=[0, topo_do_eixo]),
    )
    return figura


def valores_extremos_corrigidos(niveis: pd.DataFrame) -> go.Figure:
    """O que o INMET mediu, o Vendaval IA e a V5 estimam, em P90/P95/P99.

    Mesma leitura de `valores_por_nivel` (Seção 1), trocando o ERA5 bruto
    pelos dois métodos já corrigidos — por isso o rótulo da diferença carrega
    sinal (cada um pode superestimar ou faltar, diferente do ERA5 bruto, que
    só falta).
    """
    figura = go.Figure()
    if niveis.empty:
        return _base(figura)
    tem_v5 = "v5_medio" in niveis.columns and niveis["v5_medio"].notna().any()
    figura.add_bar(
        name="Medido pelo INMET", x=niveis["Nível"], y=niveis["inmet_medio"],
        marker_color="#ecfeff",
        text=[f"{v:.1f}" for v in niveis["inmet_medio"]],
        textposition="outside", textfont=dict(color=TINTA, size=15),
        hovertemplate="%{x}<br>INMET: %{y:.1f} m/s<extra></extra>",
    )
    figura.add_bar(
        name="Corrigido pelo Vendaval IA", x=niveis["Nível"], y=niveis["corrigido_medio"],
        marker_color=ESTADO["ok"],
        text=[f"{v:.1f}" for v in niveis["corrigido_medio"]],
        textposition="outside", textfont=dict(color=TINTA, size=15),
        hovertemplate="%{x}<br>Corrigido: %{y:.1f} m/s<extra></extra>",
    )
    maximos = [niveis["inmet_medio"].max(), niveis["corrigido_medio"].max()]
    if tem_v5:
        figura.add_bar(
            name="V5 (Kriging Ordinário)", x=niveis["Nível"], y=niveis["v5_medio"],
            marker_color=ESTADO["neutro"],
            text=[f"{v:.1f}" for v in niveis["v5_medio"]],
            textposition="outside", textfont=dict(color=TINTA, size=15),
            hovertemplate="%{x}<br>V5: %{y:.1f} m/s<extra></extra>",
        )
        maximos.append(niveis["v5_medio"].max())
    topo_do_eixo = float(max(maximos)) * 1.42
    for _, linha in niveis.iterrows():
        sinal_ia = "+" if linha["vies_medio"] >= 0 else "−"
        texto = f"IA {sinal_ia}{abs(linha['vies_medio']):.1f}"
        if tem_v5 and pd.notna(linha.get("vies_v5_medio")):
            sinal_v5 = "+" if linha["vies_v5_medio"] >= 0 else "−"
            texto += f" · V5 {sinal_v5}{abs(linha['vies_v5_medio']):.1f}"
        figura.add_annotation(
            x=linha["Nível"], y=topo_do_eixo * 0.94,
            text=texto + " m/s",
            showarrow=False, font=dict(color=TINTA, size=15),
            bgcolor=SUPERFICIE, borderpad=3,
        )
    figura = _base(figura, altura=480, titulo_y="Rajada de vento (m/s)",
                   esquerda=84, baixo=58, topo=68)
    figura.update_layout(
        barmode="group", bargap=0.3,
        yaxis=dict(range=[0, topo_do_eixo]),
    )
    return figura


def tres_mapas(df: pd.DataFrame, nivel: str, rotulo: str) -> go.Figure:
    """Medido, estimado e a subtracao dos dois, na mesma tela.

    Os dois primeiros dividem a escala de cor, senao pareceriam iguais. O
    terceiro tem escala propria, divergente e centrada no zero, porque ali o
    sinal e o que importa.
    """
    from plotly.subplots import make_subplots

    figura = make_subplots(
        rows=1, cols=3, specs=[[{"type": "map"}] * 3],
        subplot_titles=("Medido pelo INMET", "Estimado pelo ERA5",
                        "Quanto falta no ERA5"),
        horizontal_spacing=0.07,
    )
    if df.empty:
        return figura

    col_inmet, col_era5 = f"inmet_{nivel}", f"era5_{nivel}"
    diferenca = df[col_era5] - df[col_inmet]
    piso = float(min(df[col_inmet].min(), df[col_era5].min()))
    teto = float(max(df[col_inmet].max(), df[col_era5].max()))
    # Quase toda estacao fica do mesmo lado do zero, entao a pergunta aqui e
    # "quanto falta", nao "para que lado". Uma escala divergente inverteria a
    # leitura: o vermelho, que parece alarme, cairia justamente sobre as
    # estacoes onde o ERA5 acerta. Vai uma rampa de uma cor so sobre o quanto
    # falta, com o teto no percentil 98 para nao achatar o grosso do dado.
    falta = -diferenca
    teto_falta = float(np.percentile(falta.clip(lower=0), 98)) or 1.0
    # Centro pela caixa que contem as estacoes, nao pela media delas: a media
    # puxa o enquadramento para o sudeste, onde a rede e mais densa.
    centro = dict(lat=float((df["latitude"].min() + df["latitude"].max()) / 2),
                  lon=float((df["longitude"].min() + df["longitude"].max()) / 2))

    paineis = [
        (df[col_inmet], SEQUENCIAL_VENTO, piso, teto, "INMET", False),
        (df[col_era5], SEQUENCIAL_VENTO, piso, teto, "ERA5", True),
        (falta, SEQUENCIAL_FALTA, 0.0, teto_falta, "Falta no ERA5", True),
    ]
    for i, (valores, escala, cmin, cmax, nome, mostra_escala) in enumerate(paineis, start=1):
        figura.add_trace(go.Scattermap(
            lat=df["latitude"], lon=df["longitude"], mode="markers",
            marker=dict(
                size=6, color=valores, colorscale=escala, cmin=cmin, cmax=cmax,
                opacity=0.92, showscale=mostra_escala,
                colorbar=dict(
                    title=dict(text="m/s" if i == 2 else "falta<br>m/s", side="right",
                                font=dict(color=TINTA, size=13)),
                    tickfont=dict(color=TINTA, size=12),
                    thickness=14, len=0.78, x=0.655 if i == 2 else 1.0,
                    outlinewidth=0,
                ),
            ),
            customdata=np.stack([df["codigo_estacao"], valores], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>" + nome
                          + ": %{customdata[1]:.1f} m/s<extra></extra>",
            name=nome,
        ), row=1, col=i)

    for eixo in ("map", "map2", "map3"):
        figura.update_layout(**{eixo: dict(style="carto-darkmatter", center=centro, zoom=3.3)})
    figura.update_layout(
        height=500, margin=dict(l=0, r=70, t=64, b=0),
        title=dict(text=f"Rajada no nível {rotulo}", font=dict(size=16, color=TINTA)),
        paper_bgcolor=SUPERFICIE, showlegend=False,
        font=dict(color=TINTA, size=14),
        hoverlabel=dict(font=dict(color=TINTA), bgcolor="#1f1f1e"),
    )
    for anotacao in figura.layout.annotations:
        anotacao.font = dict(size=15, color=TINTA)
    return figura


def mapas_extremos(grid: pd.DataFrame, estacoes: pd.DataFrame, nivel: str) -> go.Figure:
    """INMET, Vendaval IA corrigido e V5, lado a lado, num nível (P90/P95/P99/Máximo).

    INMET e V5 são só estação (334/234 pontos, sem malha contínua); o
    Vendaval IA tem grid publicado, por isso o painel do meio soma a malha
    com as estações por cima, e os outros dois são só os pontos. Escala de
    cor comum aos três, senão a mesma cor significaria coisas diferentes em
    cada painel.
    """
    from plotly.subplots import make_subplots

    figura = make_subplots(
        rows=1, cols=3, specs=[[{"type": "map"}] * 3],
        subplot_titles=("Medido pelo INMET", "Corrigido pelo Vendaval IA",
                        "V5 (Kriging Ordinário)"),
        horizontal_spacing=0.07,
    )
    if grid.empty or estacoes.empty:
        return figura

    valores = [estacoes["inmet"], grid["rajada_max_corrigida"]]
    if "v5" in estacoes.columns:
        valores.append(estacoes["v5"].dropna())
    piso = float(min(v.min() for v in valores))
    teto = float(max(v.max() for v in valores))
    centro = dict(lat=float(estacoes["latitude"].mean()), lon=float(estacoes["longitude"].mean()))

    figura.add_trace(go.Scattermap(
        lat=estacoes["latitude"], lon=estacoes["longitude"], mode="markers",
        marker=dict(size=9, color=estacoes["inmet"], colorscale=SEQUENCIAL_VENTO,
                    cmin=piso, cmax=teto, opacity=0.95, showscale=False),
        customdata=estacoes["estacao"], name="INMET",
        hovertemplate="<b>%{customdata}</b><br>%{marker.color:.1f} m/s<extra></extra>",
    ), row=1, col=1)

    figura.add_trace(go.Scattermap(
        lat=grid["latitude"], lon=grid["longitude"], mode="markers",
        marker=dict(size=6, color=grid["rajada_max_corrigida"], colorscale=SEQUENCIAL_VENTO,
                    cmin=piso, cmax=teto, opacity=0.55, showscale=False),
        name="Grid corrigido",
        hovertemplate="%{lat:.2f}, %{lon:.2f}<br>%{marker.color:.1f} m/s<extra></extra>",
    ), row=1, col=2)
    figura.add_trace(go.Scattermap(
        lat=estacoes["latitude"], lon=estacoes["longitude"], mode="markers",
        marker=dict(size=5, color="black"), customdata=estacoes["estacao"],
        name="Estações", hovertemplate="<b>%{customdata}</b><extra></extra>",
    ), row=1, col=2)

    if "v5" in estacoes.columns:
        v5_validas = estacoes.dropna(subset=["v5"])
        figura.add_trace(go.Scattermap(
            lat=v5_validas["latitude"], lon=v5_validas["longitude"], mode="markers",
            marker=dict(
                size=9, color=v5_validas["v5"], colorscale=SEQUENCIAL_VENTO,
                cmin=piso, cmax=teto, opacity=0.95, showscale=True,
                colorbar=dict(title=dict(text="m/s", side="right", font=dict(color=TINTA, size=13)),
                              tickfont=dict(color=TINTA, size=12), thickness=14, len=0.78),
            ),
            customdata=v5_validas["estacao"], name="V5",
            hovertemplate="<b>%{customdata}</b><br>%{marker.color:.1f} m/s<extra></extra>",
        ), row=1, col=3)

    for eixo in ("map", "map2", "map3"):
        figura.update_layout(**{eixo: dict(style="carto-darkmatter", center=centro, zoom=2.6)})
    figura.update_layout(
        height=500, margin=dict(l=0, r=70, t=64, b=0),
        title=dict(text=f"Rajada no nível {nivel}", font=dict(size=16, color=TINTA)),
        paper_bgcolor=SUPERFICIE, showlegend=False,
        font=dict(color=TINTA, size=14),
        hoverlabel=dict(font=dict(color=TINTA), bgcolor="#1f1f1e"),
    )
    for anotacao in figura.layout.annotations:
        anotacao.font = dict(size=15, color=TINTA)
    return figura
