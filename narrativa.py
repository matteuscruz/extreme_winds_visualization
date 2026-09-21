"""
Narrativa — as seções que contam a história do projeto.

Cada seção abre com a pergunta que responde, em linguagem de quem não é da
área, e fecha com o achado num destaque. Nenhum número aqui é digitado: todos
vêm de `apuracao.py`, calculado dos artefatos na hora.

Onde falta informação para afirmar algo, a seção descreve a lacuna em vez de
preencher com uma interpretação plausível.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import apuracao as ap
import figuras as fg
from theme import PIPELINE_COLORS

GRAFICO = {"responsive": True}

# Cada seção: chave de navegação, título e a pergunta que ela responde.
SECOES = [
    ("problema", "1 · O problema",
     "Por que a correção que já existe não resolve o vendaval?"),
    ("experimento", "2 · O experimento",
     "O que foi testado — e o que significa cada configuração?"),
    ("resultado", "3 · O resultado",
     "Qual combinação venceu, e em quê?"),
    ("prova", "4 · A prova",
     "A correção continua valendo em anos que o modelo nunca viu?"),
    ("estabilidade", "5 · A estabilidade",
     "O acerto se mantém mês a mês, ou só na média do ano?"),
    ("entrega", "6 · A entrega",
     "O que sai disso na prática, e com que ressalva?"),
]


def _m(valor, casas=2, sufixo=""):
    """Formata número para a tela, ou um travessão quando não foi medido."""
    if valor is None or (isinstance(valor, float) and valor != valor):
        return "—"
    return f"{valor:.{casas}f}{sufixo}"


# ── Hub de entrada ────────────────────────────────────────────────────────────

def hub(ir_para):
    st.title("Vendaval: a IA captura o extremo que a interpolação não captura?")
    st.markdown(
        "O ERA5 é a base de clima mais usada do mundo e **subestima a rajada "
        "forte**. Já existe uma correção para isso, por interpolação — mas "
        "ela não consegue produzir um extremo que as estações vizinhas não "
        "registraram. Este projeto testa se modelos treinados nas variáveis "
        "atmosféricas do ERA5 passam desse limite, no Sul do Brasil."
    )

    st.info(
        "**O que este painel cobre:** o experimento de modelagem e o mapa "
        "corrigido que ele produz. **O que não cobre:** o estudo de "
        "interpolação em si, que é trabalho separado e aparece aqui só para "
        "enunciar o problema — e a coleta e o preparo do ERA5 e do INMET, "
        "que acontecem antes.",
        icon="🧭",
    )

    p = ap.panorama()
    contagem = ap.contagem_de_estacoes()
    colunas = st.columns(4)
    colunas[0].metric("Estações do INMET", p["estacoes"] or "—")
    colunas[1].metric("Áreas analisadas", p["areas"] or "—")
    colunas[2].metric(
        "Erro do ERA5 no extremo", _m(p["viés_era5"], 1, " m/s"),
        help="Viés médio no percentil 90 da rajada. Negativo = subestima.",
    )
    colunas[3].metric(
        "Depois da correção", _m(p["viés_melhor"], 1, " m/s"),
        delta=_m(-p["reducao_pct"], 0, "% de erro") if p["reducao_pct"] else None,
        delta_color="inverse",
        help="Melhor resultado entre os experimentos que cobriram todas as áreas.",
    )
    if contagem["faixa_usada"]:
        menor, maior = contagem["faixa_usada"]
        st.caption(
            f"A rede tem {p['estacoes']} estações catalogadas; cada experimento "
            f"usou entre {menor} e {maior}, porque nem toda estação tem série "
            "utilizável no período."
        )

    st.divider()
    st.caption("**Por onde começar**")
    for linha in range(0, len(SECOES), 3):
        for coluna, (chave, titulo, pergunta) in zip(
            st.columns(3), SECOES[linha:linha + 3]
        ):
            with coluna.container(border=True):
                st.markdown(f"**{titulo}**")
                st.caption(pergunta)
                st.button(
                    "Abrir", key=f"ir_{chave}",
                    on_click=ir_para, args=(chave,),
                    width="stretch",
                )


# ── 1 · O problema ────────────────────────────────────────────────────────────

def problema():
    st.header("1 · O problema")
    st.subheader("Por que a correção que já existe não resolve o vendaval?")

    base = ap.baseline_era5()
    if base:
        colunas = st.columns(3)
        colunas[0].metric("Viés na rajada extrema", _m(base["Bias_P90"], 1, " m/s"),
                          help="Percentil 90: os 10% de dias de vento mais forte.")
        colunas[1].metric("Viés em todos os dias", _m(base["Bias"], 1, " m/s"))
        colunas[2].metric("R²", _m(base["R2"], 2),
                          help="Negativo: prever sempre a média erraria menos.")

    erro = ap.erro_era5_por_estacao("P90")
    if not erro.empty:
        st.plotly_chart(
            fg.mapa_do_erro(erro, "Quanto o ERA5 subestima a rajada extrema, por estação"),
            width="stretch", config=GRAFICO, key="fig_erro_era5",
        )
        st.caption(
            f"{len(erro)} estações. Todas negativas — o ERA5 subestima em "
            f"toda a rede, de {abs(erro['value'].max()):.0f} a "
            f"{abs(erro['value'].min()):.0f} m/s conforme o lugar."
        )

    st.divider()
    st.markdown(
        "#### Isso já vinha sendo corrigido\n"
        "Mede-se o erro em cada estação, espalha-se esse campo pelo mapa e "
        "soma-se de volta ao ERA5. Para o dia comum, funciona."
    )
    st.plotly_chart(
        fg.esquema_do_teto(), width="stretch", config=GRAFICO, key="fig_teto",
    )
    st.error(
        "**O erro espalhado é uma média ponderada dos vizinhos — e média "
        "ponderada nunca sai da faixa entre eles.** Se nenhuma estação por "
        "perto mediu o vendaval, o mapa não tem como inventá-lo. Não é "
        "defeito de implementação: trocar o jeito de espalhar não resolve.",
        icon="🚨",
    )

    teto = ap.teto_da_interpolacao("p99")
    dados = ap.baseline_interpolacao("p99")
    if teto and not dados.empty:
        st.plotly_chart(
            fg.barras_da_interpolacao(dados),
            width="stretch", config=GRAFICO, key="fig_interpolacao",
        )
        st.caption(
            f"Validação que esconde cada estação e tenta prevê-la pelas "
            f"outras, sobre {teto['estacoes']} estações. Os "
            f"{teto['n_metodos_producao']} métodos em produção erram o pico "
            f"por {abs(teto['melhor_vies_producao']):.1f} a "
            f"{abs(teto['pior_vies']):.1f} m/s, sempre para menos. "
            "Números do estudo companheiro de interpolação, com régua "
            "diferente da deste painel — servem para enunciar o problema, "
            "não como placar contra as seções 3 e 5."
        )

    st.divider()
    st.markdown(
        "#### A pergunta deste projeto\n"
        "Um modelo que aprende das **variáveis atmosféricas do próprio "
        "ponto** — instabilidade, cisalhamento, umidade — escapa desse "
        "limite? Ele não depende do que o vizinho mediu."
    )
    st.code("rajada corrigida  =  fator previsto  ×  rajada do ERA5", language="text")
    st.caption(
        "O modelo prevê o fator de erro, não o vento. Prever vento do zero "
        "exigiria reaprender meteorologia; prever o quanto uma reanálise "
        "erra aproveita tudo o que o ERA5 já acerta."
    )


# ── 2 · O experimento ─────────────────────────────────────────────────────────

def experimento():
    st.header("2 · O experimento")
    st.subheader("O que foi testado — e o que significa cada configuração?")

    from engine import PERIODOS

    quantos = ap.modelos_rastreados()
    colunas = st.columns(3)
    colunas[0].metric("Abordagens de modelagem", len(ap.NOME_PIPELINE))
    colunas[1].metric("Configurações de entrada", len(ap.DESENHO_ABLACAO))
    colunas[2].metric("Modelos avaliados no screening", quantos or "—")

    st.markdown(
        "As configurações não são seis ideias soltas: são **duas perguntas "
        "cruzadas** — dar mais variáveis ao modelo ajuda, e inventar "
        "exemplos de vendaval que faltavam ajuda?"
    )
    st.plotly_chart(
        fg.grade_do_experimento(ap.celulas_do_experimento()),
        width="stretch", config=GRAFICO, key="fig_grade",
    )

    divergencias = ap.conferencia_do_desenho()
    if divergencias:
        afetadas = sorted({d["Configuração"] for d in divergencias})
        st.error(
            f"**A linha de base não é a linha de base.** Em "
            f"{len(divergencias)} execuções, “{afetadas[0]}” rodou com mais "
            "variáveis do que o desenho prevê. Como todo “quanto melhorou” "
            "é medido contra ela, **esses ganhos apontam para a referência "
            "errada** — e o efeito das variáveis novas não pode ser isolado "
            "com estes artefatos.",
            icon="🚨",
        )
        with st.expander("Ver a divergência execução por execução"):
            st.dataframe(
                pd.DataFrame(divergencias)[
                    ["Abordagem", "Configuração", "Declarado", "Gravado", "Variáveis usadas"]],
                hide_index=True, width="stretch",
            )
            gemeos = ap.bracos_gemeos()
            if gemeos:
                st.caption(
                    "Consequência: "
                    + "; ".join(f"{a}: “{b}” e “{c}”" for a, b, c in gemeos)
                    + " acabaram sendo a mesma execução."
                )
    else:
        st.success("Cada execução corresponde ao braço que diz ser.", icon="✅")

    parciais = ap.cobertura_parcial()
    if not parciais.empty:
        st.warning(
            "**Cobertura parcial não é execução interrompida.** Há variáveis "
            "que existem só em parte do território, e descartar as estações "
            "sem medição real é mais honesto do que imputar o vazio. Ainda "
            "assim, média sobre menos áreas não se compara de igual para "
            "igual — por isso essas execuções ficam fora dos gráficos da "
            "seção 3.",
            icon="⚠️",
        )

    with st.expander("De onde vêm os “extremos inventados”"):
        st.markdown(
            "Vendaval é raro, e o modelo aprende mal o que vê pouco — o "
            "projeto registra acerto alto no treino e muito mais baixo na "
            "validação, sinal de que decorou o dia comum.\n\n"
            "A resposta foi treinar uma **rede geradora** para produzir "
            "rajadas extremas artificiais, parecidas com as reais de cada "
            "área, e somá-las ao treino. A coluna da direita da grade usa "
            "esses dados; a da esquerda, não."
        )

    st.caption(
        f"Treino {PERIODOS['treino'][0][:4]}–{PERIODOS['treino'][1][:4]} · "
        f"validação {PERIODOS['validacao'][0][:4]} · "
        f"teste {PERIODOS['teste'][0][:4]}–{PERIODOS['teste'][1][:4]}, "
        "lido dos metadados das execuções."
    )


# ── 3 · O resultado ───────────────────────────────────────────────────────────

def _barras_por_metrica(quadro: pd.DataFrame, metrica: str) -> go.Figure:
    """Um traço por abordagem, uma barra por configuração."""
    figura = go.Figure()
    for pipeline, grupo in quadro.groupby("pipeline"):
        grupo = grupo.sort_values(metrica, key=abs)
        figura.add_bar(
            name=ap.NOME_PIPELINE.get(pipeline, pipeline),
            x=grupo["Configuração"], y=grupo[metrica],
            marker_color=PIPELINE_COLORS.get(pipeline),
            hovertemplate="%{x}<br>%{y:.3f}<extra>%{fullData.name}</extra>",
        )
    figura.update_layout(
        barmode="group", height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title=ap.NOME_METRICA.get(metrica, metrica),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figura


def resultado():
    st.header("3 · O resultado")
    st.subheader("Qual combinação venceu, e em quê?")

    quadro = ap.quadro_experimentos()
    if quadro.empty:
        st.warning("Nenhum experimento sincronizado.", icon="⚠️")
        return

    completos = quadro[quadro["completo"]] if quadro["completo"].any() else quadro
    base = ap.baseline_era5()

    melhor_extremo = ap.campeao("Bias_P90")
    melhor_geral = ap.campeao("R2")

    if melhor_extremo is not None and base:
        reducao = (abs(base["Bias_P90"]) - abs(melhor_extremo["Bias_P90"])) / abs(base["Bias_P90"]) * 100
        st.success(
            f"**No que mais importa — acertar a rajada extrema — quem vence é "
            f"{melhor_extremo['Abordagem']} na configuração "
            f"“{melhor_extremo['Configuração']}”**: o viés cai de "
            f"{abs(base['Bias_P90']):.1f} m/s para "
            f"{abs(melhor_extremo['Bias_P90']):.1f} m/s, "
            f"uma redução de {reducao:.0f}%.",
            icon="✅",
        )

    # O achado que o ranking por uma métrica só esconde.
    if (melhor_extremo is not None and melhor_geral is not None
            and melhor_extremo["pipeline"] != melhor_geral["pipeline"]):
        st.warning(
            f"**Não existe um vencedor único.** Quem melhor acerta o extremo "
            f"({melhor_extremo['Abordagem']}) tem R² de {melhor_extremo['R2']:.2f} "
            f"no dia a dia; quem melhor acerta o dia a dia "
            f"({melhor_geral['Abordagem']}, R² {melhor_geral['R2']:.2f}) erra o "
            f"extremo por {abs(melhor_geral['Bias_P90']):.1f} m/s. "
            "A escolha depende de para que o dado vai ser usado.",
            icon="⚖️",
        )

    if ap.conferencia_do_desenho():
        st.caption(
            "⚠️ Os números abaixo são o desempenho absoluto de cada "
            "experimento, que continua válido. O que **não** se deve ler "
            "daqui é “quanto cada configuração melhorou sobre a base” — a "
            "base publicada não é a base (ver seção 2)."
        )

    metrica = st.radio(
        "Comparar por", list(ap.NOME_METRICA),
        format_func=lambda m: ap.NOME_METRICA[m],
        horizontal=True, key="narrativa_metrica",
    )
    st.plotly_chart(
        _barras_por_metrica(completos, metrica),
        width="stretch", config=GRAFICO, key="narrativa_barras",
    )
    if not quadro["completo"].all():
        st.caption(
            "O gráfico mostra apenas os experimentos que cobriram todas as "
            "áreas. Os demais aparecem na seção 2."
        )

    with st.expander("Ver o quadro completo, número por número"):
        colunas = ["Abordagem", "Configuração", "areas", "estacoes"] + list(ap.NOME_METRICA)
        st.dataframe(
            quadro[colunas].rename(columns={
                "areas": "Áreas", "estacoes": "Estações", **ap.NOME_METRICA,
            }),
            hide_index=True, width="stretch",
        )
        st.caption(
            "Viés negativo = subestima a rajada. R² negativo = pior que prever "
            "sempre a média. REQM = raiz do erro quadrático médio, em m/s."
        )


# ── 4 · A prova ───────────────────────────────────────────────────────────────

def prova():
    st.header("4 · A prova")
    st.subheader("A correção continua valendo em anos que o modelo nunca viu?")

    from engine import PERIODOS

    st.plotly_chart(
        fg.linha_do_tempo(PERIODOS), width="stretch",
        config=GRAFICO, key="fig_linha_do_tempo",
    )

    if PERIODOS["divergentes"]:
        st.warning(
            "Nem todos os experimentos declaram o mesmo recorte de tempo.",
            icon="⚠️",
        )
        for rotulo, fatia, nomes in PERIODOS["divergentes"]:
            st.caption(f"· {rotulo}: {fatia[0]} a {fatia[1]} em {', '.join(nomes)}")
    else:
        st.success(
            f"Os três recortes não se sobrepõem, e os "
            f"{ap.panorama()['experimentos']} experimentos declaram o mesmo — "
            "todo número deste painel vem do período de teste.",
            icon="✅",
        )

    st.divider()
    por_trimestre = _resultados_por_trimestre()
    if por_trimestre.empty:
        st.info(
            "Nenhum experimento grava métrica por trimestre.", icon="ℹ️",
        )
        return

    figura = go.Figure()
    for pipeline, grupo in por_trimestre.groupby("pipeline"):
        grupo = grupo.set_index("season").reindex(["DJF", "MAM", "JJA", "SON"]).reset_index()
        figura.add_bar(
            name=ap.NOME_PIPELINE.get(pipeline, pipeline),
            x=grupo["season"], y=grupo["Bias_P90"],
            marker_color=PIPELINE_COLORS.get(pipeline),
            hovertemplate="%{x}<br>%{y:.2f} m/s<extra>%{fullData.name}</extra>",
        )
    figura.update_layout(
        barmode="group", height=340, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Viés no extremo (m/s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(figura, width="stretch", config=GRAFICO, key="narrativa_trimestres")
    st.caption(
        "DJF verão · MAM outono · JJA inverno · SON primavera (hemisfério "
        "sul). Só as abordagens que gravam resultado por trimestre aparecem."
    )


@st.cache_data(show_spinner=False)
def _resultados_por_trimestre() -> pd.DataFrame:
    """Viés no extremo por trimestre, para as abordagens que o gravam."""
    import numpy as np
    linhas = []
    for pipeline, arm in ap.combos_existentes():
        bruto = ap._resultados(pipeline, arm)
        if bruto.empty:
            continue
        recorte = bruto[bruto["split"] == "test"] if "test" in set(bruto["split"]) else bruto
        trimestres = recorte[recorte["season"].isin(["DJF", "MAM", "JJA", "SON"])]
        if trimestres.empty:
            continue
        for season, grupo in trimestres.groupby("season"):
            peso = grupo["n_samples"].astype(float)
            linhas.append({
                "pipeline": pipeline, "arm": arm, "season": season,
                "Bias_P90": float(np.average(grupo["Bias_P90"], weights=peso)),
            })
    if not linhas:
        return pd.DataFrame()
    bruto = pd.DataFrame(linhas)
    return (
        bruto.groupby(["pipeline", "season"], as_index=False)["Bias_P90"].mean()
    )


# ── 5 · A estabilidade ────────────────────────────────────────────────────────

def estabilidade():
    st.header("5 · A estabilidade")
    st.subheader("O acerto se mantém mês a mês, ou só na média do ano?")

    dados = ap.estabilidade_deploy()
    if dados.empty:
        st.info(
            "Nenhum experimento sincronizado grava o acerto recalculado por "
            "janela mensal — esta comparação não pode ser feita com os "
            "artefatos atuais.", icon="ℹ️",
        )
        return

    st.markdown(
        "Na operação o modelo roda sobre **um mês de cada vez**. O número "
        "que importa não é a média do ano, é o pior mês."
    )

    opcoes = sorted(dados["arm"].unique(), key=lambda a: list(ap.NOME_ARM).index(a))
    escolhido = st.selectbox(
        "Configuração", opcoes,
        format_func=lambda a: ap.NOME_ARM.get(a, a),
        key="estabilidade_arm",
    )
    recorte = dados[dados["arm"] == escolhido].sort_values("area")

    queda = (recorte["R2_mes_medio"] - recorte["R2_mes_pior"])
    pior_area = recorte.loc[recorte["R2_mes_pior"].idxmin()]
    mais_instavel = recorte.loc[queda.idxmax()]

    colunas = st.columns(3)
    colunas[0].metric("Acerto médio no mês", _m(recorte["R2_mes_medio"].mean(), 2),
                      help="R² médio das janelas mensais, nas áreas desta configuração.")
    colunas[1].metric("Pior mês registrado", _m(recorte["R2_mes_pior"].min(), 2),
                      help=f"Área {pior_area['area']}.")
    colunas[2].metric("Maior oscilação", _m(queda.max(), 2),
                      delta=f"área {mais_instavel['area']}", delta_color="off",
                      help="Distância entre o mês médio e o pior mês da área.")

    figura = go.Figure()
    figura.add_bar(
        name="Mês médio", x=[f"Área {a}" for a in recorte["area"]],
        y=recorte["R2_mes_medio"], marker_color=PIPELINE_COLORS["lazy"],
        hovertemplate="%{x}<br>mês médio: %{y:.3f}<extra></extra>",
    )
    figura.add_trace(go.Scatter(
        name="Pior mês", x=[f"Área {a}" for a in recorte["area"]],
        y=recorte["R2_mes_pior"], mode="markers",
        marker={"size": 10, "symbol": "diamond", "color": "#b2182b"},
        hovertemplate="%{x}<br>pior mês: %{y:.3f}<extra></extra>",
    ))
    figura.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="R² na janela mensal",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(figura, width="stretch", config=GRAFICO, key="narrativa_estabilidade")

    fragil = ap.area_mais_fragil()
    if fragil:
        st.error(
            f"**A área {fragil['area']} é o ponto fraco em "
            f"{fragil['em_quantas']} das {fragil['de_um_total']} "
            f"configurações** — não é azar de uma execução. Lá o pior mês "
            f"cai para {fragil['pior_mes']:.2f}, contra "
            f"{fragil['mes_medio']:.2f} de média.",
            icon="🚨",
        )

    resumo = ap.estabilidade_por_configuracao()
    # A faixa só é lida entre configurações que cobriram o mesmo território —
    # comparar uma média de 4 áreas com uma de 14 não mede configuração,
    # mede quais áreas entraram na conta.
    cobertura_cheia = resumo["areas"].max() if not resumo.empty else 0
    comparaveis = resumo[resumo["areas"] == cobertura_cheia]
    if len(comparaveis) > 1:
        faixa = comparaveis["mes_medio"].max() - comparaveis["mes_medio"].min()
        st.info(
            "**Trocar de configuração quase não mexe na estabilidade:** "
            f"entre as {len(comparaveis)} de cobertura completa, o acerto "
            f"mensal varia {faixa:.3f}. A geografia pesa muito mais do que "
            "a escolha de variáveis.",
            icon="🧭",
        )
        with st.expander("Comparar as configurações entre si"):
            st.dataframe(
                resumo[["Configuração", "areas", "mes_medio", "pior_mes", "queda_media"]]
                .rename(columns={
                    "areas": "Áreas", "mes_medio": "R² mês (média)",
                    "pior_mes": "Pior mês", "queda_media": "Queda média"}),
                hide_index=True, width="stretch",
            )

    with st.expander("Ver área por área"):
        st.dataframe(
            recorte[["area", "Modelo", "R2_anual", "R2_mes_medio",
                     "R2_mes_desvio", "R2_mes_pior"]].rename(columns={
                "area": "Área", "R2_anual": "R² no ano",
                "R2_mes_medio": "R² mês (média)",
                "R2_mes_desvio": "R² mês (desvio)",
                "R2_mes_pior": "R² pior mês"}),
            hide_index=True, width="stretch",
        )
        st.caption(
            "Só os modelos clássicos gravam este recorte — é a única "
            "abordagem que recalcula o acerto por janela mensal."
        )


# ── 6 · A entrega ─────────────────────────────────────────────────────────────

def entrega(ir_para):
    st.header("6 · A entrega")
    st.subheader("O que sai disso na prática, e com que ressalva?")

    from engine import (
        CORRECTED_GRID_VERSIONS, CORRECTED_GRID_DEFAULT_VERSION,
        corrected_grid_date_bounds, corrected_grid_snapshot,
        build_grid_map, stations_geo_df,
    )

    disponiveis = {
        nome: caminho for nome, caminho in CORRECTED_GRID_VERSIONS.items()
        if caminho.exists() and any(caminho.glob("*.nc"))
    }
    if not disponiveis:
        st.info("Nenhuma versão do mapa corrigido está publicada.", icon="ℹ️")
        return

    anos = sorted({
        int(a.stem.rsplit("_", 1)[-1])
        for caminho in disponiveis.values()
        for a in caminho.glob("grid_corrected_*.nc")
        if a.stem.rsplit("_", 1)[-1].isdigit()
    })
    colunas = st.columns(2)
    colunas[0].metric("Versões publicadas", ", ".join(sorted(disponiveis)))
    colunas[1].metric("Anos cobertos", f"{min(anos)}–{max(anos)}" if anos else "—")

    st.markdown(
        "Um **mapa de rajada corrigido, dia a dia**, cobrindo todo o "
        "domínio — não só onde existe estação."
    )

    versao = (CORRECTED_GRID_DEFAULT_VERSION if CORRECTED_GRID_DEFAULT_VERSION
              in disponiveis else sorted(disponiveis)[0])
    limites = corrected_grid_date_bounds(versao)
    if limites:
        dia = st.select_slider(
            "Dia mostrado", options=list(
                pd.date_range(limites[0], limites[1], freq="30D").strftime("%Y-%m-%d")),
            key="entrega_dia",
        )
        recorte = corrected_grid_snapshot(dia, versao)
        if not recorte.empty:
            esquerda, direita = st.columns(2)
            with esquerda:
                st.plotly_chart(
                    build_grid_map(recorte, "ws_original", stations_geo_df, None,
                                   "ERA5 original", height=380, show_colorbar=False),
                    width="stretch", config=GRAFICO, key="fig_grid_antes",
                )
            with direita:
                st.plotly_chart(
                    build_grid_map(recorte, "rajada_max_corrigida", stations_geo_df,
                                   None, f"Corrigido ({versao})", height=380),
                    width="stretch", config=GRAFICO, key="fig_grid_depois",
                )
            st.caption(f"{dia} · mesma escala de cor nos dois mapas.")

    st.button("Explorar o mapa dia a dia", key="ir_grid",
              on_click=ir_para, args=("explorador",))

    st.divider()
    st.markdown("#### A ressalva que o mapa carrega")
    esquerda, direita = st.columns(2)
    with esquerda.container(border=True):
        st.markdown("**Prever na estação e espalhar**")
        st.caption(
            "Mantém o desempenho medido, mas a cobertura fica presa à "
            "densidade de estações do dia. É o método das versões aqui."
        )
    with direita.container(border=True):
        st.markdown("**Prever direto em cada célula**")
        st.caption(
            "Todo ponto recebe valor todo dia, mas algumas variáveis do "
            "treino dependem de haver estação no ponto e faltam na célula. "
            "O projeto mediu essa perda, e ela não é desprezível."
        )
    st.info(
        "O acerto das seções 3 e 5 é o **da estação**. No mapa ele é menor: "
        "ali o resultado é extrapolação do que foi medido, não medição.",
        icon="🧭",
    )
    st.caption(
        "Não há métrica publicada comparando as versões do mapa entre si — "
        "o painel não afirma qual é melhor porque esse número não foi medido."
    )
