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
    teto = ap.teto_da_interpolacao("p99")

    st.markdown(
        "O ERA5 é a base de clima mais usada do mundo, e **subestima a "
        "rajada forte** — justamente o caso que interessa para risco. "
        "Comparado com o que as estações do INMET mediram, em anos que "
        "nenhum modelo usou para treinar:"
    )

    if base:
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
        st.caption(
            f"Calculado sobre {base['estacoes']} estações em {base['areas']} "
            "áreas. Viés negativo = subestima."
        )
    else:
        st.warning(
            "Os artefatos sincronizados não trazem as colunas de erro do "
            "ERA5, então esta parte não pode ser calculada.", icon="⚠️",
        )

    st.divider()
    st.markdown("#### Isso já vinha sendo corrigido — e funciona, até certo ponto")
    st.markdown(
        "Existe um trabalho anterior, de interpolação, que corrige o ERA5 "
        "assim: mede-se o erro em cada estação, esse campo de erros é "
        "espalhado pelo mapa, e o resultado é somado de volta ao ERA5. "
        "Para o dia comum, resolve."
    )
    st.code(
        "erro na estação  →  espalha o erro pelo mapa  →  soma de volta ao ERA5",
        language="text",
    )

    st.markdown("#### Mas ele não consegue criar um extremo")
    st.markdown(
        "O erro espalhado num ponto é uma **média ponderada** dos erros das "
        "estações vizinhas. Média ponderada fica sempre entre o menor e o "
        "maior valor que entrou nela — nunca acima. Se nenhuma estação por "
        "perto registrou o vendaval, o mapa não tem como inventá-lo."
    )
    st.info(
        "Isso não é defeito de implementação, e trocar o jeito de espalhar "
        "não resolve: é propriedade do mecanismo, e vale para qualquer peso "
        "que se escolha.",
        icon="🧭",
    )

    if teto:
        st.error(
            f"**Medido:** numa validação que esconde cada estação e tenta "
            f"prevê-la a partir das outras, sobre {teto['estacoes']} "
            f"estações, os {teto['n_metodos_producao']} métodos dessa "
            "linhagem erram o pico anual por "
            f"{abs(teto['melhor_vies_producao']):.1f} a "
            f"{abs(teto['pior_vies']):.1f} m/s — todos para menos.",
            icon="🚨",
        )
        dados = ap.baseline_interpolacao("p99")
        with st.expander("Ver os métodos de interpolação avaliados"):
            st.dataframe(
                dados[["Método", "bias", "rmse", "corr", "em_producao"]].rename(
                    columns={"bias": "Viés", "rmse": "REQM",
                             "corr": "Correlação", "em_producao": "Em produção"}),
                hide_index=True, width="stretch",
                column_config={
                    "Em produção": st.column_config.CheckboxColumn(
                        help="Métodos da linhagem que já corrigia o ERA5 quando este projeto começou."),
                },
            )
            if teto["melhor_alternativa"]:
                st.caption(
                    "O mesmo estudo achou alternativas melhores dentro da "
                    f"própria interpolação — a melhor delas, "
                    f"“{teto['melhor_alternativa']}”, reduz o erro típico de "
                    f"{teto['reqm_pior_producao']:.1f} para "
                    f"{teto['reqm_melhor_alternativa']:.1f} m/s. Mesmo assim, "
                    "todas continuam presas ao mesmo limite: nenhuma "
                    "consegue passar do maior erro vizinho."
                )
        st.caption(
            "Números do estudo companheiro de interpolação, não deste "
            "projeto. A régua é outra — percentil, conjunto de estações e "
            "desenho de validação diferem — então servem para enunciar o "
            "problema, nunca como placar contra os resultados das seções 3 e 5."
        )

    st.divider()
    st.markdown("#### A pergunta que este projeto faz")
    st.markdown(
        "Um modelo que aprende a partir das **variáveis atmosféricas do "
        "ERA5** — e não dos vizinhos — escapa desse limite? Ele enxerga "
        "instabilidade, cisalhamento e umidade no próprio ponto, então pode "
        "em princípio apontar um extremo que nenhuma estação próxima viu."
    )
    st.markdown(
        "E há um detalhe que costuma surpreender: **o modelo não tenta "
        "prever o vento.** Ele prevê o quanto o ERA5 errou — um fator que "
        "depois multiplica o valor do ERA5."
    )
    st.code("rajada corrigida  =  fator previsto  ×  rajada do ERA5", language="text")
    st.markdown(
        "Prever o vento do zero exigiria reaprender toda a meteorologia; "
        "prever o quanto uma reanálise erra é um problema muito menor, e "
        "aproveita tudo o que o ERA5 já acerta. **As próximas seções medem "
        "se isso funcionou.**"
    )


# ── 2 · O experimento ─────────────────────────────────────────────────────────

def experimento():
    st.header("2 · O experimento")
    st.subheader("O que foi testado — e o que significa cada configuração?")

    from engine import PERIODOS

    st.markdown(
        "Foram cruzadas **três abordagens de modelagem** com **seis "
        "configurações de entrada**, treinando cada combinação separadamente "
        "em cada área. Ao todo, 18 experimentos."
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

    st.divider()
    st.markdown("#### As seis configurações são duas perguntas cruzadas")
    st.markdown(
        "Não são seis ideias soltas. São duas perguntas testadas ao mesmo "
        "tempo: **dar mais variáveis ao modelo ajuda?** e **inventar "
        "exemplos de vendaval que faltavam ajuda?**"
    )

    grade = []
    por_grupos: dict[str, dict] = {}
    for arm, ficha in ap.DESENHO_ABLACAO.items():
        por_grupos.setdefault(ficha["grupos"], {})[ficha["sinteticos"]] = arm
    for grupos, colunas_por_sintetico in por_grupos.items():
        rotulos = [
            ap.NOME_GRUPO.get(g.strip(), g.strip())
            for g in grupos.split(",")
        ]
        grade.append({
            "O que o modelo recebe": " + ".join(rotulos),
            "Sem extremos inventados": ap.NOME_ARM.get(
                colunas_por_sintetico.get(False), "—"),
            "Com extremos inventados": ap.NOME_ARM.get(
                colunas_por_sintetico.get(True), "—"),
        })
    st.dataframe(pd.DataFrame(grade), hide_index=True, width="stretch")
    if any("—" in linha.values() for linha in grade):
        st.caption(
            "As lacunas na tabela são reais: nem toda combinação foi "
            "executada, então algumas comparações de par não existem."
        )

    with st.expander("De onde vêm os “extremos inventados”"):
        st.markdown(
            "Vendaval é raro por definição, e o modelo aprende mal aquilo que "
            "vê pouco — o próprio projeto registra acerto alto no treino e "
            "muito mais baixo na validação, sinal de que decorou o dia comum "
            "em vez de aprender o dia extremo.\n\n"
            "A resposta foi treinar uma **rede geradora** (uma GAN) só para "
            "produzir rajadas extremas artificiais, estatisticamente "
            "parecidas com as reais de cada área, e acrescentá-las ao treino. "
            "As configurações da coluna da direita são as que usam esses "
            "dados; as da esquerda, não. Comparar as duas colunas é o teste "
            "de se a ideia funcionou."
        )

    st.caption(
        f"Treino: {PERIODOS['treino'][0]} a {PERIODOS['treino'][1]} · "
        f"Validação: {PERIODOS['validacao'][0]} a {PERIODOS['validacao'][1]} · "
        f"Teste: {PERIODOS['teste'][0]} a {PERIODOS['teste'][1]}. "
        "Lido dos metadados das execuções."
    )

    st.divider()
    st.markdown("#### O que conferir antes de comparar")

    divergencias = ap.conferencia_do_desenho()
    if divergencias:
        afetadas = sorted({d["Configuração"] for d in divergencias})
        st.error(
            "**A linha de base do experimento não corresponde ao que deveria "
            f"ser.** Em {len(divergencias)} das execuções publicadas, a "
            f"configuração “{afetadas[0]}” — que deveria usar só o conjunto "
            "mínimo de variáveis — foi executada com variáveis a mais. "
            "Como toda comparação “quanto melhorou?” é medida contra ela, "
            "**esses ganhos estão medidos contra a referência errada** e o "
            "efeito das variáveis novas não pode ser isolado com os "
            "artefatos atuais.",
            icon="🚨",
        )
        st.dataframe(
            pd.DataFrame(divergencias)[
                ["Abordagem", "Configuração", "Declarado", "Gravado", "Variáveis usadas"]
            ],
            hide_index=True, width="stretch",
            column_config={
                "Declarado": st.column_config.TextColumn(
                    help="Grupos de variáveis que o desenho do experimento prevê para este braço."),
                "Gravado": st.column_config.TextColumn(
                    help="Grupos efetivamente registrados nos metadados da execução."),
            },
        )
        gemeos = ap.bracos_gemeos()
        if gemeos:
            pares = "; ".join(f"{a}: “{b}” e “{c}”" for a, b, c in gemeos)
            st.caption(
                f"Consequência direta — {pares} acabaram usando exatamente a "
                "mesma lista de variáveis e os mesmos dados de treino. "
                "Comparar uma com a outra não responde pergunta nenhuma."
            )
    else:
        st.success(
            "Cada execução publicada corresponde ao braço que diz ser: os "
            "grupos de variáveis registrados batem com o desenho do "
            "experimento.",
            icon="✅",
        )

    parciais = ap.cobertura_parcial()
    if not parciais.empty:
        total_areas = ap.panorama()["areas"]
        st.warning(
            f"{len(parciais)} execução(ões) cobriram só parte das "
            f"{total_areas} áreas. **Isso não é execução interrompida** — há "
            "variáveis que existem só em parte do território, e descartar as "
            "estações sem medição real é mais honesto do que preencher o "
            "vazio por imputação. Mesmo assim, uma média sobre menos áreas "
            "não é comparável de igual para igual com uma média sobre todas.",
            icon="⚠️",
        )
        st.dataframe(
            parciais.rename(columns={"areas": "Áreas cobertas", "estacoes": "Estações"}),
            hide_index=True, width="stretch",
        )
        st.caption(
            "Os metadados da execução não registram se o descarte por "
            "cobertura estava ligado, então a explicação acima é a leitura "
            "mais provável — não uma afirmação."
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
        "Um acerto medido uma vez sobre o ano inteiro esconde o mês ruim. "
        "Na operação real o modelo roda sobre **um mês de cada vez** — então "
        "o número que importa não é a média anual, é o pior mês."
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

    st.warning(
        f"**A estabilidade não é uniforme no território.** Na área "
        f"{mais_instavel['area']} o acerto cai {queda.max():.2f} entre o mês "
        f"médio e o pior mês; na mais estável, a queda é de "
        f"{queda.min():.2f}. Onde a oscilação é grande, o número anual é uma "
        "promessa que o modelo não cumpre todo mês.",
        icon="⚖️",
    )

    fragil = ap.area_mais_fragil()
    if fragil:
        st.error(
            f"**A área {fragil['area']} é o ponto fraco em "
            f"{fragil['em_quantas']} das {fragil['de_um_total']} "
            "configurações testadas** — não é azar de uma execução. "
            f"Lá o pior mês cai para {fragil['pior_mes']:.2f}, contra "
            f"{fragil['mes_medio']:.2f} de média mensal. Qualquer uso do "
            "resultado nessa região merece cautela extra.",
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
            "**Trocar de configuração quase não mexe na estabilidade.** "
            f"Entre a melhor e a pior das {len(comparaveis)} configurações "
            "que cobriram todas as áreas, o acerto mensal médio varia "
            f"{faixa:.3f}. O que separa as áreas "
            "uma da outra é muito maior do que o que separa as "
            "configurações — a geografia pesa mais do que a escolha de "
            "variáveis.",
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

    from engine import CORRECTED_GRID_VERSIONS

    st.markdown(
        "Um **mapa de rajada corrigido**, dia a dia, cobrindo todo o "
        "domínio — não só onde existe estação. Para cada área, o mapa usa a "
        "combinação que venceu ali."
    )

    disponiveis = {
        nome: caminho for nome, caminho in CORRECTED_GRID_VERSIONS.items()
        if caminho.exists() and any(caminho.glob("*.nc"))
    }
    if not disponiveis:
        st.info(
            "Nenhuma versão do mapa corrigido está publicada neste painel.",
            icon="ℹ️",
        )
        return

    anos = set()
    for caminho in disponiveis.values():
        for arquivo in caminho.glob("grid_corrected_*.nc"):
            sufixo = arquivo.stem.rsplit("_", 1)[-1]
            if sufixo.isdigit():
                anos.add(int(sufixo))
    colunas = st.columns(2)
    colunas[0].metric("Versões publicadas", ", ".join(sorted(disponiveis)))
    colunas[1].metric("Anos cobertos", f"{min(anos)}–{max(anos)}" if anos else "—")

    st.button(
        "Abrir o mapa corrigido", key="ir_grid",
        on_click=ir_para, args=("explorador",),
    )

    st.divider()
    st.markdown("#### A ressalva que o mapa carrega")
    st.markdown(
        "O modelo aprendeu **onde há estação medindo**. Levar isso para um "
        "mapa contínuo exige uma escolha, e as duas saídas possíveis têm "
        "custo — nenhuma é de graça."
    )

    esquerda, direita = st.columns(2)
    with esquerda.container(border=True):
        st.markdown("**Prever na estação e espalhar**")
        st.caption(
            "O modelo roda onde tem estação, e o valor é interpolado para o "
            "resto do mapa. Mantém o desempenho medido, mas a cobertura fica "
            "presa à densidade de estações daquele dia — em anos antigos, "
            "com pouquíssimas estações ativas, boa parte do domínio fica sem "
            "valor."
        )
        st.caption("É o método das versões publicadas aqui.")
    with direita.container(border=True):
        st.markdown("**Prever direto em cada célula**")
        st.caption(
            "O modelo é aplicado célula a célula, então todo ponto recebe "
            "valor todo dia. O custo é que algumas variáveis do treino "
            "dependem de haver uma estação no ponto — lags da própria "
            "rajada observada e a mediana histórica da estação — e não "
            "existem numa célula vazia. O projeto mediu a perda de acerto "
            "que isso causa, e ela não é desprezível."
        )
        st.caption("É o método de uma versão mais recente, ainda não publicada aqui.")

    st.info(
        "**Por que isso está na tela:** quem usar o mapa precisa saber que "
        "ele é uma extrapolação do que foi medido na estação, não uma "
        "medição. O acerto mostrado nas seções 3 e 5 é o da estação — no "
        "mapa, ele é menor.",
        icon="🧭",
    )

    st.divider()
    st.caption(
        "**O que ainda não está aqui:** não há métrica publicada comparando "
        "as versões do mapa entre si. Dá para ver as duas lado a lado no "
        "Explorador, mas o painel não afirma qual é melhor, porque esse "
        "número não foi medido."
    )
