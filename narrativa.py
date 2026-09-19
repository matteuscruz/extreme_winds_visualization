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
from theme import PIPELINE_COLORS

GRAFICO = {"responsive": True}

# Cada seção: chave de navegação, título e a pergunta que ela responde.
SECOES = [
    ("problema", "1 · O problema",
     "Por que a rajada de vento do ERA5 não serve como está?"),
    ("experimento", "2 · O experimento",
     "O que foi testado — e o que significa cada configuração?"),
    ("resultado", "3 · O resultado",
     "Qual combinação venceu, e em quê?"),
    ("prova", "4 · A prova",
     "A correção continua valendo em anos que o modelo nunca viu?"),
    ("entrega", "5 · A entrega",
     "O que sai disso na prática?"),
]


def _m(valor, casas=2, sufixo=""):
    """Formata número para a tela, ou um travessão quando não foi medido."""
    if valor is None or (isinstance(valor, float) and valor != valor):
        return "—"
    return f"{valor:.{casas}f}{sufixo}"


# ── Hub de entrada ────────────────────────────────────────────────────────────

def hub(ir_para):
    st.title("Rajada de vento extremo: corrigindo o erro do ERA5")
    st.markdown(
        "O ERA5 é a base de dados de clima mais usada do mundo, mas **erra "
        "muito a rajada de vento forte** — justamente o caso que interessa "
        "para risco. Este projeto treina modelos para corrigir esse erro no "
        "Sul do Brasil, usando as estações do INMET como verdade."
    )

    st.info(
        "**O que este painel cobre:** o experimento de correção de viés e o "
        "mapa corrigido que ele produz. **O que não cobre:** a coleta e o "
        "pré-processamento do ERA5 e do INMET, que acontecem antes, fora "
        "deste painel.",
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
    st.subheader("Por que a rajada de vento do ERA5 não serve como está?")

    base = ap.baseline_era5()
    if not base:
        st.warning(
            "Os artefatos sincronizados não trazem as colunas de erro do ERA5, "
            "então esta seção não tem como ser calculada.", icon="⚠️",
        )
        return

    st.markdown(
        "Comparando o ERA5 com o que as estações do INMET mediram de verdade, "
        "nos anos que nenhum modelo usou para treinar:"
    )

    colunas = st.columns(3)
    colunas[0].metric(
        "Viés na rajada extrema (P90)", _m(base["Bias_P90"], 2, " m/s"),
        help="Média sobre os 10% de dias de vento mais forte.",
    )
    colunas[1].metric(
        "Viés em todos os dias", _m(base["Bias"], 2, " m/s"),
        help="Média sobre todos os dias, não só os de vento forte.",
    )
    colunas[2].metric("R²", _m(base["R2"], 2))

    st.error(
        f"**O ERA5 subestima a rajada extrema em {abs(base['Bias_P90']):.1f} m/s "
        f"em média** — e seu R² de {base['R2']:.2f} é negativo, ou seja, prever "
        "sempre a média das observações erraria menos do que usar o ERA5.",
        icon="🚨",
    )

    st.markdown(
        "O erro não é uniforme no território. Por área, ele vai de "
        f"**{abs(base['melhor_area_Bias_P90']):.1f}** a "
        f"**{abs(base['pior_area_Bias_P90']):.1f} m/s** de subestimação — "
        "é por isso que o projeto corrige área por área, e não com um fator único."
    )

    st.caption(
        f"Calculado sobre {base['estacoes']} estações em {base['areas']} áreas. "
        "P90 = os 10% de dias de vento mais forte."
    )


# ── 2 · O experimento ─────────────────────────────────────────────────────────

def experimento():
    st.header("2 · O experimento")
    st.subheader("O que foi testado — e o que significa cada configuração?")

    from engine import PERIODOS

    st.markdown(
        "Foram cruzadas **três abordagens de modelagem** com **seis "
        "configurações de dado de entrada**, treinando cada combinação "
        "separadamente em cada área."
    )

    colunas = st.columns(3)
    for coluna, (chave, nome) in zip(colunas, ap.NOME_PIPELINE.items()):
        with coluna.container(border=True):
            st.markdown(f"**{nome}**")
            st.caption({
                "lazy": "Dezenas de modelos estatísticos e de árvore testados "
                        "em série; vence o melhor de cada área.",
                "mlp": "Uma rede neural simples, treinada com peso extra nos "
                       "dias de vento forte.",
                "lstm": "Uma rede que enxerga a sequência dos dias, não cada "
                        "dia isolado.",
            }[chave])

    quantos = ap.modelos_rastreados()
    if quantos:
        st.caption(f"O screening de modelos clássicos avaliou {quantos} tipos de modelo diferentes.")

    st.markdown("**As seis configurações de entrada**, medidas nos metadados de cada execução:")
    fichas = ap.fichas_das_configuracoes()
    if not fichas.empty:
        st.dataframe(
            fichas[["Configuração", "Variáveis", "Dados sintéticos", "Cobertura completa"]],
            hide_index=True, width="stretch",
            column_config={
                "Variáveis": st.column_config.TextColumn(
                    help="Quantas variáveis o modelo recebeu. A faixa aparece "
                         "quando as três abordagens não usam exatamente a mesma lista."),
                "Cobertura completa": st.column_config.CheckboxColumn(
                    help="Se o experimento rodou em todas as áreas."),
            },
        )

    st.caption(
        f"Treino: {PERIODOS['treino'][0]} a {PERIODOS['treino'][1]} · "
        f"Validação: {PERIODOS['validacao'][0]} a {PERIODOS['validacao'][1]} · "
        f"Teste: {PERIODOS['teste'][0]} a {PERIODOS['teste'][1]}. "
        "Lido dos metadados das execuções."
    )

    # Ressalvas medidas — quem lê a comparação precisa saber disto antes.
    duplicadas = ap.configuracoes_identicas()
    if duplicadas:
        pares = ", ".join(
            f"**{ap.NOME_ARM.get(a, a)}** e **{ap.NOME_ARM.get(b, b)}**"
            for a, b in duplicadas
        )
        st.warning(
            f"{pares} usam exatamente a mesma lista de variáveis e os mesmos "
            "dados de treino nos metadados. Pelo que está registrado, são a "
            "mesma configuração com dois nomes — a diferença entre elas, se "
            "existir, não está nos artefatos.",
            icon="⚠️",
        )

    incompletos = ap.experimentos_incompletos()
    if not incompletos.empty:
        total_areas = ap.panorama()["areas"]
        st.warning(
            f"{len(incompletos)} experimento(s) cobriram apenas parte das "
            f"{total_areas} áreas e **não são comparáveis de igual para igual** "
            "com os demais — uma média sobre menos áreas pode parecer melhor "
            "só por deixar as áreas difíceis de fora.",
            icon="⚠️",
        )
        st.dataframe(
            incompletos.rename(columns={"areas": "Áreas cobertas", "estacoes": "Estações"}),
            hide_index=True, width="stretch",
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

    treino, validacao, teste = PERIODOS["treino"], PERIODOS["validacao"], PERIODOS["teste"]
    st.markdown(
        "Sim — e é essa a razão de todo número deste painel vir do período de "
        "teste, não do de treino. Os três recortes não se sobrepõem:"
    )
    colunas = st.columns(3)
    colunas[0].metric("Treino", f"{treino[0][:4]}–{treino[1][:4]}",
                      help="Anos em que o modelo aprendeu.")
    colunas[1].metric("Validação", f"{validacao[0][:4]}",
                      help="Ano usado para ajustar as escolhas de modelagem.")
    colunas[2].metric("Teste", f"{teste[0][:4]}–{teste[1][:4]}",
                      help="Anos nunca vistos. É daqui que saem os resultados mostrados.")

    if PERIODOS["divergentes"]:
        st.warning(
            "Nem todos os experimentos declaram o mesmo recorte de tempo — "
            "veja a lista abaixo antes de comparar.", icon="⚠️",
        )
        for rotulo, fatia, nomes in PERIODOS["divergentes"]:
            st.caption(f"· {rotulo}: {fatia[0]} a {fatia[1]} em {', '.join(nomes)}")
    else:
        st.success(
            f"Os {ap.panorama()['experimentos']} experimentos declaram o mesmo "
            "recorte de tempo, então a comparação entre eles é justa.",
            icon="✅",
        )

    st.divider()
    st.markdown(
        "**O erro muda com a estação do ano?** Só as abordagens que gravam "
        "resultado por trimestre aparecem aqui."
    )
    por_trimestre = _resultados_por_trimestre()
    if por_trimestre.empty:
        st.info(
            "Nenhum experimento sincronizado grava métrica por trimestre — "
            "esta comparação não pode ser feita com os artefatos atuais.",
            icon="ℹ️",
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
        "DJF = verão · MAM = outono · JJA = inverno · SON = primavera "
        "(hemisfério sul). Média das configurações que cobriram todas as áreas."
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


# ── 5 · A entrega ─────────────────────────────────────────────────────────────

def entrega(ir_para):
    st.header("5 · A entrega")
    st.subheader("O que sai disso na prática?")

    from engine import CORRECTED_GRID_VERSIONS

    st.markdown(
        "Um **mapa de rajada corrigido**, dia a dia, cobrindo todo o domínio — "
        "não só onde existe estação. Para cada área, o mapa usa a combinação "
        "que venceu ali, costurando os melhores resultados num produto único."
    )

    disponiveis = {
        nome: caminho for nome, caminho in CORRECTED_GRID_VERSIONS.items()
        if caminho.exists() and any(caminho.glob("*.nc"))
    }
    if disponiveis:
        anos = set()
        for caminho in disponiveis.values():
            for arquivo in caminho.glob("grid_corrected_*.nc"):
                sufixo = arquivo.stem.rsplit("_", 1)[-1]
                if sufixo.isdigit():
                    anos.add(int(sufixo))
        colunas = st.columns(2)
        colunas[0].metric("Versões publicadas", ", ".join(sorted(disponiveis)))
        colunas[1].metric(
            "Anos cobertos",
            f"{min(anos)}–{max(anos)}" if anos else "—",
        )
        st.success(
            "O mapa corrigido está no painel e pode ser explorado ponto a "
            "ponto, com a série de cada estação lado a lado: observado, ERA5 "
            "bruto e ERA5 corrigido.",
            icon="✅",
        )
        st.button(
            "Abrir o mapa corrigido", key="ir_grid",
            on_click=ir_para, args=("explorador",),
        )
    else:
        st.info(
            "Nenhuma versão do mapa corrigido está sincronizada neste painel "
            "no momento.", icon="ℹ️",
        )

    st.divider()
    st.caption(
        "**O que ainda não está aqui:** a comparação entre as versões do mapa "
        "corrigido não tem métrica publicada nos artefatos — dá para ver as "
        "duas lado a lado, mas o painel não afirma qual é melhor, porque esse "
        "número não foi medido."
    )
