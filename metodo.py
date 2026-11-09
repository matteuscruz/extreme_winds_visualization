"""Seção 3, o experimento: o que foi feito, na ordem em que foi feito.

O texto descreve o método e traz uma prévia do resultado do estudo do cluster
3. Situação de downloads e dos clusters não aparece aqui, porque envelhece.
Fontes: a documentação do pipeline, os dicionários dos grupos de variáveis e a
auditoria do projeto. Os números do cluster 3 vêm de
`dataset/estudo_c3_importancia.csv` e `dataset/estudo_c3_configuracoes.csv`,
derivados dos resultados do estudo de variáveis.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import clusters as cl

TINTA = "#f2f0eb"
TINTA_FRACA = "#b4b1a9"
FUNDO = "#17181a"
GRAFICO = {"responsive": True, "displaylogo": False}

IMPORTANCIA = "dataset/estudo_c3_importancia.csv"
CONFIGURACOES = "dataset/estudo_c3_configuracoes.csv"
ERRO_POR_CLUSTER = "dataset/erro_por_cluster.csv"

# Os quatro grupos do estudo: (id, título, o que é, colunas no estudo).
GRUPOS = [
    ("grupo1", "Grupo 1 · Vento e camada limite", "O vento perto do chão, a camada de ar que toca a superfície e o tempo.", 45),
    ("grupo2", "Grupo 2 · Convecção e tempestades", "O ambiente que forma tempestades e correntes de ar descendentes.", 29),
    ("grupo3", "Grupo 3 · Situação sinótica", "O quadro de grande escala: pressão, ventos e giro em altitude.", 30),
    ("grupo4", "Grupo 4 · Relevo e superfície", "A forma do terreno, fixa no tempo, na célula de cada estação.", 63),
]

# Cada variável do estudo e o que a sigla significa. "No ponto" é a célula do
# ERA5 da estação, "no entorno" são as estatísticas em 75 e 250 km, e as
# defasagens são os valores das horas anteriores.
VARIAVEIS = {
    # Grupo 1
    "gust10fg": "Rajada do ERA5 no ponto",
    "gust": "Rajada do ERA5 no entorno",
    "w10": "Velocidade do vento a 10 m",
    "u10": "Vento a 10 m, componente leste-oeste",
    "v10": "Vento a 10 m, componente norte-sul",
    "w100": "Velocidade do vento a 100 m",
    "u100": "Vento a 100 m, componente leste-oeste",
    "v100": "Vento a 100 m, componente norte-sul",
    "bwd_10_100": "Cisalhamento do vento entre 10 e 100 m",
    "zust": "Velocidade de fricção, a turbulência perto do chão",
    "blh": "Altura da camada limite, onde o ar se mistura",
    "mbld": "Dissipação de energia na camada limite",
    "msshf": "Fluxo de calor sensível, o calor que sobe do solo",
    "t2m": "Temperatura a 2 m",
    "d2m": "Ponto de orvalho a 2 m, a umidade do ar",
    "dewpoint_depression": "Diferença entre temperatura e ponto de orvalho: ar seco ou úmido",
    "sp": "Pressão na superfície",
    "hora_solar_sin": "Hora solar do dia, em seno (ciclo do dia)",
    "hora_solar_cos": "Hora solar do dia, em cosseno (ciclo do dia)",
    "doy_sin": "Dia do ano, em seno (ciclo das estações)",
    "doy_cos": "Dia do ano, em cosseno (ciclo das estações)",
    "latitude": "Latitude da estação",
    "longitude": "Longitude da estação",
    # Grupo 2
    "cape": "CAPE, a energia disponível para uma tempestade se formar",
    "mcpr": "Taxa de chuva convectiva, a das tempestades",
    "mtpr": "Taxa de chuva total",
    "dcape_j_kg": "DCAPE, a energia das correntes de ar descendentes",
    "dls_0_6km": "Cisalhamento do vento de 0 a 6 km de altura",
    "lls_0_1km": "Cisalhamento do vento de 0 a 1 km de altura",
    "mean_wind_0_6km": "Vento médio de 0 a 6 km de altura",
    "lapse_2_4km": "Queda da temperatura com a altura, de 2 a 4 km",
    "lapse_700_500": "Queda da temperatura com a altura, de 700 a 500 hPa",
    # Grupo 3
    "mslp_hpa": "Pressão ao nível do mar no ponto",
    "mslp": "Pressão ao nível do mar no entorno",
    "grad_mslp_hpa_100km": "Gradiente da pressão no ponto: quanto ela muda a cada 100 km",
    "grad_mslp": "Gradiente da pressão no entorno",
    "mslp_tend_3h": "Tendência da pressão: variação em 3 horas",
    "w850": "Velocidade do vento a 850 hPa (cerca de 1,5 km)",
    "w500": "Velocidade do vento a 500 hPa (cerca de 5,5 km)",
    "z500_m": "Altura da superfície de 500 hPa",
    "vort850": "Vorticidade a 850 hPa, o giro do ar",
    "vort500": "Vorticidade a 500 hPa, o giro do ar",
    "omega700": "Movimento vertical do ar a 700 hPa",
    "t850": "Temperatura a 850 hPa",
    # Grupo 4
    "z_surface": "Altura do terreno",
    "sdor": "Variação do relevo dentro da célula",
    "sdfor": "Variação do relevo, filtrada para escalas de 3 a 22 km",
    "slor": "Declividade do terreno",
    "isor": "Forma do relevo: serra alongada ou morro espalhado",
    "anor": "Orientação do relevo, para onde a serra aponta",
    "land_sea_mask": "Fração de terra da célula",
}

SITUACAO = {"manter": "Mantida", "inconclusiva": "Inconclusiva", "descartar": "Descartada"}

# O que foi feito sobre as variáveis do ERA5.
ENGENHARIA = [
    ("Velocidade e cisalhamento",
     "O vento chega em duas componentes, leste-oeste e norte-sul. Delas saem a velocidade e o "
     "cisalhamento, a diferença do vento entre dois níveis de altura."),
    ("Entorno, em 75 e 250 km",
     "Para cada estação, o ERA5 em volta foi resumido por mediana, máximo e desvio. O raio de 75 km "
     "é o entorno próximo, e o de 250 km o regional. A distância é medida na esfera, e a célula da "
     "própria estação entra."),
    ("Contraste entre escalas",
     "A diferença entre o entorno próximo e o regional. Dois valores iguais querem dizer coisas "
     "diferentes se o entorno largo está calmo ou agitado."),
    ("Defasagens de 1 a 3 horas",
     "O valor 1 hora antes, o máximo e o desvio das 3 horas anteriores, e a variação em relação a "
     "3 horas antes. A rajada é um evento de tendência: importa o que o ar vinha fazendo."),
    ("Ciclos do tempo",
     "A hora solar e o dia do ano entram em seno e cosseno, para 23 h ficar perto de 0 h e 31 de "
     "dezembro perto de 1 de janeiro."),
    ("Perfil vertical (grupo 2)",
     "Níveis abaixo do solo são descartados hora a hora. Vento e temperatura são interpolados por "
     "altura acima do solo, e a DCAPE é calculada a partir do perfil."),
]

# Nome claro de cada configuração comparada.
NOMES_CONFIG = {
    "era5": "Referência: ERA5 sem correção",
    "base": "Só vento e camada limite (grupo 1)",
    "full": "Todos os grupos",
    "sel__val12": "12 variáveis selecionadas",
    "drop_grp__grupo1": "Todos menos o grupo 1",
    "drop_grp__grupo2": "Todos menos o grupo 2",
    "drop_grp__grupo3": "Todos menos o grupo 3",
    "drop_grp__grupo4": "Todos menos o grupo 4",
    "add_grp__grupo2": "Grupo 1 mais o grupo 2",
    "add_grp__grupo3": "Grupo 1 mais o grupo 3",
    "add_grp__grupo4": "Grupo 1 mais o grupo 4",
    "ctrl__noise": "Controle: grupo 1 mais ruído",
    "ctrl__perm_static": "Controle: grupo 1 mais relevo embaralhado",
}
NOMES_ISOLADOS = {
    "so_grupo2": "Só o grupo 2",
    "so_grupo3": "Só o grupo 3",
    "so_grupo4": "Só o grupo 4",
    "so_ruido": "Só ruído",
}

CONFIGURACOES_EXPLICADAS = [
    ("Referência: ERA5 sem correção", "A rajada do ERA5 como ela vem. É o que os modelos precisam superar."),
    ("Só vento e camada limite", "Apenas o grupo 1."),
    ("Todos os grupos", "Os quatro grupos juntos, 167 colunas."),
    ("Grupo 1 mais um grupo", "O grupo 1 com o grupo 2, 3 ou 4 somado, um de cada vez."),
    ("Todos menos um grupo", "Os quatro grupos menos um, para ver o que se perde sem ele."),
    ("Um grupo sozinho", "Cada um dos grupos 2, 3 e 4 sem o grupo 1."),
    ("Controles", "Colunas de ruído e variáveis de relevo embaralhadas. Servem de piso: um ganho que não passa deles não é real."),
    ("12 variáveis selecionadas", "Só as variáveis que a importância por permutação manteve."),
]

CORES_DIVISAO = {"Treino e validação": "#3d7ea6", "Teste": "#d9a441", "Ano nunca visto": "#9aa3ad"}


# ── Figuras ───────────────────────────────────────────────────────────────────

def figura_da_divisao() -> go.Figure:
    """Cada mês de 2000 a 2025 e para que ele serve."""
    anos = list(range(2000, 2026))
    meses = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    nomes = list(CORES_DIVISAO)
    z, texto = [], []
    for m in range(1, 13):
        linha, linha_t = [], []
        for ano in anos:
            k = 2 if ano == 2025 else (1 if m in (1, 4, 7, 10) else 0)
            linha.append(k)
            linha_t.append(f"{meses[m - 1]} de {ano}<br>{nomes[k]}")
        z.append(linha)
        texto.append(linha_t)
    cores = list(CORES_DIVISAO.values())
    figura = go.Figure(go.Heatmap(
        x=anos, y=meses, z=z, text=texto, hovertemplate="%{text}<extra></extra>",
        colorscale=[[0, cores[0]], [0.33, cores[0]], [0.34, cores[1]], [0.66, cores[1]],
                    [0.67, cores[2]], [1, cores[2]]],
        zmin=0, zmax=2, showscale=False, xgap=2, ygap=2,
    ))
    for nome, cor in CORES_DIVISAO.items():
        figura.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nome,
                                    marker=dict(size=12, color=cor, symbol="square")))
    figura.update_layout(
        height=360, margin=dict(l=40, r=10, t=10, b=30), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=TINTA)),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        xaxis=dict(showgrid=False, dtick=2, tickfont=dict(color=TINTA_FRACA, size=12)),
        yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(color=TINTA_FRACA, size=12)),
    )
    return figura


def figura_das_configuracoes(cfg: pd.DataFrame) -> go.Figure:
    """Erro de cada configuração no teste, do cluster 3. Menor é melhor."""
    rmse = {}
    for r in cfg.itertuples():
        if r.origem in ("referencia", "arvores") and r.arm in NOMES_CONFIG:
            rmse[NOMES_CONFIG[r.arm]] = r.rmse_teste
        elif r.origem == "isolado" and r.arm in NOMES_ISOLADOS:
            rmse[NOMES_ISOLADOS[r.arm]] = r.rmse_teste
    # Ordem lógica, não por valor: entre 2,29 e 2,31 não há ranking de verdade.
    ordem = [n for n in (
        [NOMES_CONFIG[k] for k in ("era5", "base", "full", "sel__val12",
                                   "add_grp__grupo2", "add_grp__grupo3", "add_grp__grupo4",
                                   "drop_grp__grupo1", "drop_grp__grupo2", "drop_grp__grupo3",
                                   "drop_grp__grupo4", "ctrl__noise", "ctrl__perm_static")]
        + [NOMES_ISOLADOS[k] for k in ("so_grupo2", "so_grupo3", "so_grupo4", "so_ruido")]
    ) if n in rmse]
    era5 = rmse[NOMES_CONFIG["era5"]]
    cores = []
    for nome in ordem:
        if nome == NOMES_CONFIG["era5"]:
            cores.append("#9aa3ad")
        elif nome.startswith("Só ") and nome != NOMES_CONFIG["base"]:
            cores.append("#6d7279")
        elif rmse[nome] < era5:
            cores.append("#3d7ea6")
        else:
            cores.append("#b8932f")
    figura = go.Figure(go.Bar(
        x=[rmse[n] for n in ordem], y=ordem, orientation="h", marker_color=cores,
        text=[f"{rmse[n]:.2f}" for n in ordem], textposition="outside",
        textfont=dict(color=TINTA), cliponaxis=False,
        hovertemplate="%{y}<br>Erro: %{x:.2f} m/s<extra></extra>"))
    figura.add_vline(x=era5, line=dict(color="#9aa3ad", dash="dash", width=1))
    figura.update_layout(
        height=620, margin=dict(l=10, r=40, t=10, b=40), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        xaxis=dict(title="Erro no teste (RMSE, m/s), menor é melhor", gridcolor="#26292d",
                   range=[0, max(rmse.values()) * 1.1], tickfont=dict(color=TINTA_FRACA)),
        yaxis=dict(tickfont=dict(color=TINTA), showgrid=False, automargin=True, autorange="reversed"),
    )
    return figura


NOMES_MODELO = {
    "era5": "ERA5 sem correção",
    "ml_local": "ML local",
    "ml_global_trimestral": "ML global por trimestre",
    "ml_global_anual": "ML global anual",
    "lstm_local": "LSTM local",
    "lstm_global_trimestral": "LSTM global por trimestre",
    "lstm_global_anual": "LSTM global anual",
}


def _virgula(v: float, casas: int = 2) -> str:
    """Número com vírgula decimal, como no resto do texto."""
    return f"{v:.{casas}f}".replace(".", ",")


def _milhar(v: float) -> str:
    """1600 vira 1.600."""
    return f"{int(round(v)):,}".replace(",", ".")


def figura_do_cluster(erro: pd.DataFrame, cluster: int) -> go.Figure:
    """Erro de cada modelo no teste de um cluster, contra o ERA5."""
    sub = erro[erro["cluster"] == cluster].set_index("modelo")["rmse_teste"]
    era5 = float(sub["era5"])
    nomes = [NOMES_MODELO[m] for m in NOMES_MODELO if m in sub.index]
    valores = [float(sub[m]) for m in NOMES_MODELO if m in sub.index]
    cores = ["#9aa3ad" if n == NOMES_MODELO["era5"] else ("#3d7ea6" if v < era5 else "#b8932f")
             for n, v in zip(nomes, valores)]
    figura = go.Figure(go.Bar(
        x=valores, y=nomes, orientation="h", marker_color=cores,
        text=[f"{v:.2f}" for v in valores], textposition="outside",
        textfont=dict(color=TINTA), cliponaxis=False,
        hovertemplate="%{y}<br>Erro: %{x:.2f} m/s<extra></extra>"))
    figura.add_vline(x=era5, line=dict(color="#9aa3ad", dash="dash", width=1))
    figura.update_layout(
        height=380, margin=dict(l=10, r=40, t=10, b=40), plot_bgcolor=FUNDO,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=14),
        hoverlabel=dict(font=dict(color=TINTA, size=14), bgcolor="#1f1f1e"),
        xaxis=dict(title="Erro no teste (RMSE, m/s), menor é melhor", gridcolor="#26292d",
                   range=[0, max(valores) * 1.15], tickfont=dict(color=TINTA_FRACA)),
        yaxis=dict(tickfont=dict(color=TINTA), showgrid=False, automargin=True, autorange="reversed"),
    )
    return figura


# ── Tabelas ───────────────────────────────────────────────────────────────────

def tabela_do_grupo(imp: pd.DataFrame, grupo: str) -> pd.DataFrame:
    sub = imp[imp["grupo"] == grupo].sort_values("d_rmse_medio", ascending=False)
    situacao = []
    for r in sub.itertuples():
        txt = SITUACAO[r.decisao]
        if r.variavel == "anor":
            txt = "Mantida, retirada depois"
        situacao.append(txt)
    return pd.DataFrame({
        "Variável": sub["variavel"].values,
        "O que é": [VARIAVEIS.get(v, "") for v in sub["variavel"]],
        "Poder preditivo": sub["d_rmse_medio"].clip(lower=0).values,
        "Situação": situacao,
    })


# ── Seção ─────────────────────────────────────────────────────────────────────

def render():
    imp = pd.read_csv(IMPORTANCIA)
    cfg = pd.read_csv(CONFIGURACOES)
    g = cl.carregar()
    est = cl.carregar_estacoes()
    cel = cl.carregar_celulas()

    st.header("3 · Experimento")
    st.markdown(
        "O experimento seguiu sete passos. As escolhas foram feitas num cluster só, o 3, "
        "por escalabilidade, e depois levadas aos outros."
    )

    st.markdown("#### 1. O que entra e o que sai")
    st.markdown(
        "O modelo recebe o ERA5 hora a hora e estima a **rajada máxima do dia** que a "
        "estação do INMET mediria. Cada exemplo é uma estação em um dia, tomado na hora "
        "em que o ERA5 teve o pico de rajada. A referência a superar é a própria rajada "
        "do ERA5 nessa hora."
    )

    st.markdown("#### 2. Quatro grupos de variáveis")
    st.markdown(
        "No cluster 3, onde o estudo foi feito, foram usadas 24 variáveis do ERA5, de superfície e "
        "de níveis de pressão, organizadas em quatro grupos pela física do vento forte."
    )
    c3 = est[est["modelada"] & (est["cluster"] == 3)]
    st.markdown(
        f"O cluster 3 tem {len(c3)} estações do INMET, mas só {int(c3['com_serie'].sum())} têm "
        "rajada medida até 2024. Foram elas que treinaram e testaram o estudo. As demais só "
        "começaram a reportar em 2025."
    )
    st.plotly_chart(cl.mapa_do_cluster(g, cel, est, 3), width="stretch", config=GRAFICO,
                    key="mapa_cluster3")
    abas = st.tabs([t.split(" · ")[0] for _, t, _, _ in GRUPOS])
    for aba, (gid, titulo, descricao, colunas) in zip(abas, GRUPOS):
        with aba:
            st.markdown(f"**{titulo.split(' · ')[1]}.** {descricao} {colunas} colunas no estudo.")
            st.dataframe(
                tabela_do_grupo(imp, gid), hide_index=True, width="stretch",
                column_config={
                    "Variável": st.column_config.TextColumn(width="small"),
                    "O que é": st.column_config.TextColumn(width="large"),
                    "Poder preditivo": st.column_config.ProgressColumn(
                        min_value=0, max_value=float(imp["d_rmse_medio"].max()), format="%.3f",
                        help="Quanto o erro piora, em m/s, quando a variável é embaralhada. "
                             "Quanto maior, mais o modelo depende dela."),
                    "Situação": st.column_config.TextColumn(width="small"),
                })
    st.caption(
        "Cada variável reúne o valor no ponto, as estatísticas do entorno e as defasagens. O poder "
        "preditivo é medido no cluster 3, na validação, com um modelo LightGBM."
    )

    st.markdown("##### Como as variáveis foram trabalhadas")
    st.markdown(
        "Das variáveis do ERA5 saíram outras, calculadas. No total, 167 colunas nos quatro grupos."
    )
    st.table({"Operação": [e[0] for e in ENGENHARIA], "O que faz": [e[1] for e in ENGENHARIA]})
    st.plotly_chart(cl.mapa_do_entorno(g, est, 3), width="stretch", config=GRAFICO,
                    key="mapa_entorno")
    st.caption("O entorno de uma estação do cluster 3: os pontos do ERA5 a menos de 75 km e a menos de 250 km.")

    st.markdown("#### 3. Configurações comparadas")
    st.markdown("Cada configuração decide quais colunas o modelo enxerga.")
    st.table({"Configuração": [c[0] for c in CONFIGURACOES_EXPLICADAS],
              "O que o modelo recebe": [c[1] for c in CONFIGURACOES_EXPLICADAS]})

    st.markdown("#### 4. Quais modelos")
    st.markdown(
        "Primeiro foram testados **39 algoritmos de regressão** (biblioteca LazyPredict), entre "
        "lineares, árvores, florestas, boosting e redes, na configuração só com vento e camada "
        "limite. Os 5 melhores de cada trimestre ficaram fixos e foram usados em todas as "
        "outras configurações, cada um treinado com 5 sementes aleatórias para o resultado "
        "não depender da sorte."
    )
    st.plotly_chart(figura_das_configuracoes(cfg), width="stretch", config=GRAFICO, key="configuracoes")
    st.markdown(
        "Só o **grupo 1** supera o ERA5, e juntar os outros três não melhora. Sozinhos, os grupos "
        "2, 3 e 4 ficam piores que o ERA5, embora melhores que ruído. As 12 variáveis selecionadas "
        "mantêm o erro com muito menos colunas. Este é o quadro das árvores; com a rede LSTM ele "
        "muda um pouco, e isso aparece nos resultados."
    )

    st.markdown("#### 5. Quais variáveis ficam")
    st.markdown(
        "Para escolher variáveis, cada uma foi **embaralhada** e mediu-se quanto o erro "
        "piorava, a importância por permutação. Uma coluna de ruído dá o piso: ficam as "
        "variáveis cuja piora média passa dele e que pioram o erro em ao menos 3 dos 4 "
        "trimestres. As colunas de uma mesma variável foram embaralhadas juntas."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Colunas testadas", 167)
    c2.metric("Variáveis", int(len(imp)))
    c3.metric("Mantidas", int((imp["decisao"] == "manter").sum()), help="Viram 58 colunas.")
    c4.metric("Colunas nos modelos finais", 49,
              help="Nove colunas da orientação do relevo foram retiradas depois de uma revisão do método.")

    st.markdown("#### 6. Treino e teste")
    st.markdown(
        "Os modelos finais usam essas 49 colunas, em **três escalas**: **local** (um cluster, um "
        "trimestre), **global por trimestre** (todos os clusters juntos) e **global anual** "
        "(todos os clusters, o ano todo). Cada escala tem uma versão de aprendizado de máquina "
        "clássico e uma rede LSTM, que lê as 24 horas até o pico. Um ajuste final de quantis "
        "estica de volta os picos que o treino tende a suavizar."
    )
    st.plotly_chart(figura_da_divisao(), width="stretch", config=GRAFICO, key="divisao")
    st.markdown(
        "O teste são meses inteiros, janeiro, abril, julho e outubro, que o modelo não viu, "
        "para dias vizinhos não ficarem um no treino e outro no teste. A validação, usada para "
        "parar o treino e escolher entre modelos, são meses sorteados entre os demais. 2025 "
        "fica fora de tudo. As estações são as mesmas no treino e no teste: o teste mede "
        "meses novos, não estações novas."
    )

    st.markdown("#### 7. Validação no cluster 12")
    erro = pd.read_csv(ERRO_POR_CLUSTER)
    dist = cl.distancia_entre_clusters(g, 3, 12)
    st.markdown(
        "O cluster 12, no leste de Minas Gerais, no Espírito Santo e no Rio de Janeiro, ficou "
        f"reservado para **validar**. Ele está a cerca de {_milhar(dist)} km do cluster 3, onde as "
        "variáveis foram escolhidas, e tem outro regime de vento. Recebeu as mesmas variáveis e "
        "o mesmo treino. Se elas ajudam lá, o modelo não está bom só por causa da região onde "
        "foi escolhido."
    )
    st.plotly_chart(cl.mapa_do_cluster(g, cel, est, 12), width="stretch", config=GRAFICO,
                    key="mapa_cluster12")
    e12 = erro[erro["cluster"] == 12].set_index("modelo")["rmse_teste"]
    era5 = float(e12["era5"])
    modelos = [m for m in NOMES_MODELO if m != "era5" and m in e12.index]
    melhores = [m for m in modelos if e12[m] < era5]
    top = min(modelos, key=lambda m: e12[m])
    st.plotly_chart(figura_do_cluster(erro, 12), width="stretch", config=GRAFICO, key="erro_cluster12")
    st.markdown(
        f"{len(melhores)} dos {len(modelos)} modelos ficam abaixo do ERA5 no cluster 12. O melhor, "
        f"a {NOMES_MODELO[top]}, vai de {_virgula(era5)} para {_virgula(float(e12[top]))} m/s."
    )
