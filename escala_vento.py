"""Escala de cor dos mapas de velocidade do vento.

Mesma paleta do painel do Vendaval Ciclone: amarelo claro para o vento fraco, vermelho escuro para o forte,
em faixas discretas, e cinza opcional para o que fica abaixo do piso. Cada
faixa tem uma cor só, sem gradiente, e a legenda marca exatamente as
fronteiras das faixas, para dar para ler o valor no mapa.

Sem Streamlit e sem leitura de disco.
"""

from __future__ import annotations

import numpy as np

COR_ABAIXO_DO_PISO = "#b3b3b3"
CORES = [
    "#ffff99", "#ffe64d", "#ffcc00", "#ffb300", "#ff9900",
    "#ff7300", "#ff4d00", "#e62600", "#cc0000",
]


def escala(limites: list[float], cinza_abaixo: bool = False) -> dict:
    """Escala discreta a partir das fronteiras das faixas.

    `limites` tem uma fronteira a mais que o número de faixas, em ordem
    crescente. Com `cinza_abaixo`, a primeira faixa sai cinza (o piso) e as
    outras recebem as cores quentes, do amarelo ao vermelho.

    Devolve o que o Plotly pede (`colorscale`, `cmin`, `cmax`) e a barra de
    cor (`colorbar`) com as marcações nas fronteiras reais.
    """
    limites = [float(v) for v in limites]
    if len(limites) < 2 or any(b <= a for a, b in zip(limites, limites[1:])):
        raise ValueError("`limites` precisa ter ao menos duas fronteiras crescentes.")

    faixas = len(limites) - 1
    quentes = faixas - (1 if cinza_abaixo else 0)
    if quentes < 1:
        raise ValueError("Com piso cinza é preciso ao menos uma faixa quente.")
    indices = np.rint(np.linspace(0, len(CORES) - 1, quentes)).astype(int)
    cores = ([COR_ABAIXO_DO_PISO] if cinza_abaixo else []) + [CORES[i] for i in indices]

    vmin, vmax = limites[0], limites[-1]
    posicoes = [(v - vmin) / (vmax - vmin) for v in limites]
    colorscale = []
    for cor, inicio, fim in zip(cores, posicoes[:-1], posicoes[1:]):
        colorscale += [[inicio, cor], [fim, cor]]
    colorscale[-1][0] = 1.0

    return {
        "colorscale": colorscale,
        "cmin": vmin,
        "cmax": vmax,
        "colorbar": {
            "tickmode": "array",
            "tickvals": limites,
            "ticktext": [f"{v:g}" for v in limites],
        },
        "cores": cores,
    }
