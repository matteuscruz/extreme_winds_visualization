"""Tokens de design visual — sem Streamlit, sem I/O de disco (mesma filosofia
do `viz.py` do dashboard de referência, `interpolation comparisson/streamlit_app/`).

Duas dimensões de cor coexistem neste dashboard, e são INTENCIONALMENTE
diferentes — não é redundância a "limpar":
- `ARM_COLORS` (em app.py) já cobre a dimensão "configuração de ablation"
  (original/synthetic/newfeatures/all/basin/all_basin), reaproveitada em
  bars/heatmap/density/boxplot/scatter.
- `PIPELINE_COLORS` (aqui) cobre a dimensão "pipeline de modelo"
  (lazy/mlp/lstm), que hoje não tinha nenhuma cor fixa em lugar nenhum — só
  aparecia como texto (`PIPELINE_LABELS`). O dashboard de referência só tinha
  UMA dimensão de cor ("fonte"); aqui precisamos das duas porque o produto
  compara pipeline × configuração simultaneamente.
"""
from __future__ import annotations

# Deliberadamente distintas das 6 cores já usadas em ARM_COLORS (app.py) para
# não colidir visualmente quando os dois tokens aparecem na mesma tela (ex.:
# contorno do marcador = pipeline, preenchimento = configuração).
# Validado por medição, não por gosto: o trio anterior (#6b7280, #0f766e,
# #7c3aed) reprovava em três checagens, o teal e o cinza-azulado ficavam a
# ΔE 2,8 sob protanopia e a ΔE 9,3 em VISÃO NORMAL, abaixo do piso de 15,
# e os dois não alcançavam o piso de croma (liam como cinza).
# Este trio passa em todas: banda de luminosidade, croma, separação sob
# daltonismo, piso de visão normal e contraste ≥ 3:1 contra a superfície.
PIPELINE_COLORS = {
    "lazy": "#d95926",   # laranja
    "mlp": "#9085e9",    # violeta
    "lstm": "#008300",   # verde
}

# Colorscale divergente pros mapas de DIFERENÇA (residual = candidato -
# referência). Porte literal de
# `interpolation comparisson/streamlit_app/viz.py::diff_colorscale` —
# problema do RdBu linear padrão: numa escala simétrica [-dmax, +dmax],
# valores pequenos caem perto do meio da escala, que é quase branco, e ficam
# ilegíveis mesmo sendo um sinal real. Fix: mesmos 11 stops do ColorBrewer
# RdBu (cores inalteradas), reposicionados com um expoente p>1 que PUXA os
# tons mais saturados pra perto do centro. Formato de saída
# (`[[posição, "rgb(...)"], ...]`) é aceito de forma idêntica por
# `go.Heatmap`, `go.Scatter` e `go.Scattermap` (todos compartilham o mesmo
# validador de colorscale do Plotly).
_RDBU_STOPS = [
    (0.0, "rgb(5,48,97)"), (0.1, "rgb(33,102,172)"), (0.2, "rgb(67,147,195)"),
    (0.3, "rgb(146,197,222)"), (0.4, "rgb(209,229,240)"), (0.5, "rgb(247,247,247)"),
    (0.6, "rgb(253,219,199)"), (0.7, "rgb(244,165,130)"), (0.8, "rgb(214,96,77)"),
    (0.9, "rgb(178,24,43)"), (1.0, "rgb(103,0,31)"),
]  # posição 0 = azul (sub-estima) · posição 1 = vermelho (sobre-estima)


def diff_colorscale(p: float = 2.0) -> list[list]:
    """Colorscale divergente azul-branco-vermelho com sensibilidade reforçada
    perto do zero. `p` controla o quanto: 1.0 = RdBu linear padrão; 1.4-2.0
    para uma correção perceptível mas sutil. Não usado ainda por nenhum
    gráfico (o mapa de diferença de `build_interp_map` está fora do escopo
    deste redesign) — mantido aqui para reaproveitamento futuro."""
    stops = []
    for pos, color in _RDBU_STOPS:
        offset = pos - 0.5
        if offset == 0:
            new_pos = 0.5
        else:
            sign = 1 if offset > 0 else -1
            new_pos = 0.5 + sign * (abs(offset) ** p) * (0.5 ** (1 - p))
        stops.append([round(new_pos, 4), color])
    return stops
