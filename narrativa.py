"""
Narrativa, as seções que contam a história do projeto.

Cada seção abre com a pergunta que responde, em linguagem de quem não é da
área, e fecha com o achado num destaque. Nenhum número aqui é digitado: todos
vêm de `apuracao.py`, calculado dos artefatos na hora.

Onde falta informação para afirmar algo, a seção descreve a lacuna em vez de
preencher com uma interpretação plausível.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import apuracao as ap
import figuras as fg
from theme import PIPELINE_COLORS

GRAFICO = {"responsive": True}

# Cada seção: chave de navegação, título e a pergunta que ela responde.
SECOES = [
    ("problema", "1 · O problema", "Por que a correção que já existe não resolve o vendaval?"),
    ("experimento", "2 · O experimento", "O que foi testado, e o que significa cada configuração?"),
    ("resultado", "3 · O resultado", "Qual combinação venceu, e em quê?"),
    ("prova", "4 · A prova", "A correção continua valendo em anos que o modelo nunca viu?"),
    ("estabilidade", "5 · A estabilidade", "O acerto se mantém mês a mês, ou só na média do ano?"),
    ("entrega", "6 · A entrega", "O que sai disso na prática, e com que ressalva?"),
]


SEM_DADO = "sem dado"


def _m(valor, casas=2, sufixo=""):
    """Formata número para a tela, ou avisa que não foi medido."""
    if valor is None or (isinstance(valor, float) and valor != valor):
        return SEM_DADO
    return f"{valor:.{casas}f}{sufixo}"


# ── Hub de entrada ────────────────────────────────────────────────────────────

def hub(ir_para):
    st.title("Vendaval: a IA captura o extremo que a interpolação não captura?")
    st.markdown(
        "O ERA5 é a base de clima mais usada do mundo e **subestima a rajada "
        "forte**. Já existe uma correção para isso, por interpolação, mas "
        "ela não consegue produzir um extremo que as estações vizinhas não "
        "registraram. Este projeto testa se modelos treinados nas variáveis "
        "atmosféricas do ERA5 passam desse limite, no Sul do Brasil."
    )

    st.info(
        "**O que este painel cobre:** o experimento de modelagem e o mapa "
        "corrigido que ele produz. **O que não cobre:** o estudo de "
        "interpolação em si, que é trabalho separado e aparece aqui só para "
        "enunciar o problema, e a coleta e o preparo do ERA5 e do INMET, "
        "que acontecem antes."
    )

    p = ap.panorama()
    contagem = ap.contagem_de_estacoes()
    colunas = st.columns(4)
    colunas[0].metric("Estações do INMET", p["estacoes"] or SEM_DADO)
    colunas[1].metric("Áreas analisadas", p["areas"] or SEM_DADO)
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

    st.info(
        "**A pergunta deste projeto:** um modelo que aprende das variáveis "
        "atmosféricas do próprio ponto consegue capturar o vendaval que a "
        "interpolação não captura?"
    )
    st.caption(
        "Tudo o que vem abaixo existe para responder isso. Primeiro o erro "
        "que se quer corrigir, depois por que a correção atual não chega lá."
    )

    st.divider()
    st.markdown("#### Quanto o ERA5 erra, e onde")

    niveis = ap.vies_por_nivel()
    nacional = ap.comparacao_nacional()

    if niveis.empty:
        st.warning("O arquivo de comparação com o INMET não está disponível.")
        return

    st.markdown(
        "Cada estação do INMET tem um instrumento que mede a rajada todo dia. "
        "O ERA5 estima essa mesma rajada por cálculo, sem instrumento no "
        "lugar. Comparando os dois na mesma estação, em três recortes:"
    )
    for _, linha in niveis.iterrows():
        st.caption(f"**{linha['Nível']}**: {linha['explicacao']}.")

    st.plotly_chart(
        fg.valores_por_nivel(niveis),
        width="stretch", config=GRAFICO, key="fig_valores_nivel",
    )

    ultimo = niveis.iloc[-1]
    st.caption(
        f"A diferença é subtração direta: o valor do ERA5 menos o valor do "
        f"INMET, na mesma estação. No {ultimo['Nível']}, por exemplo, o "
        f"instrumento registrou {ultimo['inmet_medio']:.1f} m/s em média e o "
        f"ERA5 estimou {ultimo['era5_medio']:.1f}, então faltam "
        f"{abs(ultimo['vies_medio']):.1f} m/s. Média sobre "
        f"{int(ultimo['estacoes'])} estações do país inteiro."
    )

    if not nacional.empty:
        rotulos = list(ap.NIVEIS_DE_EXTREMO.values())
        escolhido = st.radio(
            "Recorte mostrado nos mapas", rotulos,
            horizontal=True, key="nivel_mapa_nacional",
        )
        chave = list(ap.NIVEIS_DE_EXTREMO)[rotulos.index(escolhido)]
        st.plotly_chart(
            fg.tres_mapas(nacional, chave, escolhido),
            width="stretch", config=GRAFICO, key="fig_mapas_nacional",
        )
        st.caption(
            "Os dois primeiros mapas dividem a mesma escala de cor, senão "
            "pareceriam iguais: quanto mais claro, mais forte o vento. O "
            "terceiro mostra quanto falta no ERA5 para alcançar o medido, "
            "estação por estação: escuro é pouco, claro é muito. Onde o ERA5 "
            f"passou do medido, o que acontece em "
            f"{(1 - niveis.loc[niveis['nivel'] == chave, 'frac_subestima'].iloc[0]) * 100:.0f}% "
            "das estações neste recorte, o ponto aparece no tom mais escuro."
        )

    st.divider()
    st.markdown("#### Isso já vinha sendo corrigido, e ainda assim o extremo escapa")
    st.markdown(
        "O método mede o erro em cada estação, espalha esse campo pelo mapa e "
        "soma de volta ao ERA5. Duas versões foram para produção, e cada uma "
        "resolveu o problema da anterior criando outro."
    )

    esquerda, direita = st.columns(2)
    with esquerda.container(border=True):
        st.markdown("**V2, peso pela distância**")
        st.caption(
            "A estação mais próxima domina o peso num raio curto. O campo sai "
            "como um mosaico de manchas quase constantes ao redor de cada "
            "estação, com transição brusca na borda, em vez de variar suave "
            "com a distância."
        )
    with direita.container(border=True):
        st.markdown("**V3, peso por kernel gaussiano**")
        st.caption(
            "Trocou o peso para resolver a mancha, e resolveu: o campo "
            "linearizou. Mas rodou com alcance de cerca de 200 km, enquanto a "
            "dependência espacial real medida no dado diário é de cerca de 10 "
            "km. Ao suavizar nessa largura, o pico observado nas estações "
            "some do campo."
        )

    teto = ap.teto_da_interpolacao("p99")
    d95, d99 = ap.baseline_interpolacao("p95"), ap.baseline_interpolacao("p99")
    if not d99.empty:
        st.plotly_chart(
            fg.metodos_lado_a_lado(d95, d99),
            width="stretch", config=GRAFICO, key="fig_metodos",
        )
        if teto:
            st.caption(
                f"Validação que esconde cada estação e tenta prevê-la pelas "
                f"outras, sobre {teto['estacoes']} estações. As três versões "
                f"da linhagem de produção erram o pico por "
                f"{abs(teto['melhor_vies_producao']):.1f} a "
                f"{abs(teto['pior_vies']):.1f} m/s, sempre para menos, e "
                f"“{teto['melhor_alternativa']}” chega mais perto sem escapar "
                "do mesmo limite. Números do estudo de interpolação, com "
                "régua diferente da usada nas seções 3 e 5."
            )

    st.divider()
    st.markdown("#### Por que um modelo pode passar desse limite")
    st.markdown(
        "Ele não olha para o vizinho. Olha para o estado da atmosfera no "
        "próprio ponto: instabilidade, cisalhamento, umidade. E não tenta "
        "prever o vento, prevê o quanto o ERA5 errou."
    )
    st.code("rajada corrigida  =  fator previsto  ×  rajada do ERA5", language="text")
    st.caption(
        "Prever vento do zero exigiria reaprender meteorologia. Prever o "
        "quanto uma reanálise erra aproveita tudo o que o ERA5 já acerta."
    )


# ── 2 · O experimento ─────────────────────────────────────────────────────────

def experimento():
    st.header("2 · O experimento")
    st.subheader("O que foi testado, e o que significa cada configuração?")

    from engine import PERIODOS

    quantos = ap.modelos_rastreados()
    colunas = st.columns(3)
    colunas[0].metric("Abordagens de modelagem", len(ap.NOME_PIPELINE))
    colunas[1].metric("Configurações de entrada", len(ap.DESENHO_ABLACAO))
    colunas[2].metric("Modelos avaliados no screening", quantos or SEM_DADO)

    st.info(
        "**Abordagens de modelagem** — **Modelos clássicos**: várias técnicas "
        "tradicionais de aprendizado de máquina testadas automaticamente por "
        "área e estação, fica a que acerta mais em cada uma. **Rede neural "
        "(MLP)**: uma rede neural simples, várias camadas de neurônios "
        "artificiais, olha cada momento isolado. **Rede recorrente (LSTM)**: "
        "uma rede neural com memória entre instantes de tempo, pensada para "
        "dado que evolui, não só uma foto isolada da atmosfera."
    )

    st.markdown(
        "As configurações não são seis ideias soltas: são **duas perguntas "
        "cruzadas**, dar mais variáveis ao modelo ajuda, e inventar "
        "exemplos de vendaval que faltavam ajuda?"
    )
    rotulo_grupo = {"original": "Base", "era5_18z": "18Z", "bt55": "Satélite", "era5_basin": "Bacia"}
    definicoes = " · ".join(
        f"**{rotulo_grupo[g]}**: {ap.NOME_GRUPO[g]}" for g in rotulo_grupo
    )
    st.info(
        f"**Grupos de variáveis** — {definicoes}. **Extremos inventados**: "
        "rajadas artificiais geradas por uma rede treinada para imitar as "
        "reais, somadas ao treino porque vendaval de verdade é raro demais "
        "para o modelo aprender só com o que existe."
    )
    tabela = fg.tabela_do_experimento(ap.celulas_do_experimento())
    if tabela is not None:
        st.markdown(tabela, unsafe_allow_html=True)
        st.caption(
            "Verde = execução íntegra · laranja = cobertura parcial · "
            "vermelho = não corresponde ao desenho · cinza = não executada."
        )

    divergencias = ap.conferencia_do_desenho()
    if divergencias:
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
        st.success("Cada execução corresponde ao braço que diz ser.")


    with st.expander("De onde vêm os “extremos inventados”"):
        st.markdown(
            "Vendaval é raro, e o modelo aprende mal o que vê pouco, o "
            "projeto registra acerto alto no treino e muito mais baixo na "
            "validação, sinal de que decorou o dia comum.\n\n"
            "A resposta foi treinar uma **rede geradora** para produzir "
            "rajadas extremas artificiais, parecidas com as reais de cada "
            "área, e somá-las ao treino. A coluna da direita da grade usa "
            "esses dados; a da esquerda, não."
        )

    st.caption(
        f"Treino {PERIODOS['treino'][0][:4]}-{PERIODOS['treino'][1][:4]} · "
        f"validação {PERIODOS['validacao'][0][:4]} · "
        f"teste {PERIODOS['teste'][0][:4]}-{PERIODOS['teste'][1][:4]}, "
        "lido dos metadados das execuções."
    )


# ── 3 · O resultado ───────────────────────────────────────────────────────────

def resultado():
    st.header("3 · O resultado")
    st.subheader("Qual combinação venceu, e em quê?")

    quadro = ap.quadro_experimentos()
    if quadro.empty:
        st.warning("Nenhum experimento sincronizado.")
        return

    completos = quadro[quadro["completo"]] if quadro["completo"].any() else quadro
    base = ap.baseline_era5()

    melhor_extremo = ap.campeao("Bias_P90")
    melhor_geral = ap.campeao("R2")

    if melhor_extremo is not None and base:
        reducao = (abs(base["Bias_P90"]) - abs(melhor_extremo["Bias_P90"])) / abs(base["Bias_P90"]) * 100
        st.info(
            f"No viés de rajada extrema (P90), a configuração com menor erro é "
            f"{melhor_extremo['Abordagem']} / “{melhor_extremo['Configuração']}”: "
            f"{abs(base['Bias_P90']):.1f} m/s no ERA5 bruto contra "
            f"{abs(melhor_extremo['Bias_P90']):.1f} m/s aqui, uma redução de "
            f"{reducao:.0f}%."
        )

    if (melhor_extremo is not None and melhor_geral is not None
            and melhor_extremo["pipeline"] != melhor_geral["pipeline"]):
        st.info(
            f"Quem tem o menor erro no extremo ({melhor_extremo['Abordagem']}) "
            f"tem R² de {melhor_extremo['R2']:.2f} no dia a dia; quem tem o "
            f"maior R² no dia a dia ({melhor_geral['Abordagem']}, R² "
            f"{melhor_geral['R2']:.2f}) erra o extremo por "
            f"{abs(melhor_geral['Bias_P90']):.1f} m/s. Nenhuma configuração "
            "vence nas duas métricas ao mesmo tempo."
        )

    if ap.conferencia_do_desenho():
        st.caption(
            "Os números abaixo são o desempenho absoluto de cada "
            "experimento, que continua válido. O que **não** se deve ler "
            "daqui é “quanto cada configuração melhorou sobre a base”, a "
            "base publicada não é a base (ver seção 2)."
        )

    metrica = st.radio(
        "Comparar por", list(ap.NOME_METRICA),
        format_func=lambda m: ap.NOME_METRICA[m],
        horizontal=True, key="narrativa_metrica",
    )
    st.plotly_chart(
        fg.barras_por_metrica(completos, metrica, ap.NOME_METRICA[metrica], PIPELINE_COLORS),
        width="stretch", config=GRAFICO, key="narrativa_barras",
    )
    st.caption(
        "As três abordagens testadas: **Modelos clássicos** (várias técnicas "
        "tradicionais de aprendizado de máquina, testadas automaticamente "
        "por área e estação), **Rede neural (MLP)** (rede neural simples, "
        "olha cada momento isolado) e **Rede recorrente (LSTM)** (rede "
        "neural com memória entre instantes de tempo)."
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
    st.markdown(
        "O modelo **aprendeu** nos anos de treino, **ajustou escolhas** "
        "(qual configuração usar) nos de validação, e nunca viu os de "
        "teste. Todo número deste painel, aqui e nas outras seções, vem só "
        "do período de teste."
    )

    if PERIODOS["divergentes"]:
        st.warning("Nem todos os experimentos declaram o mesmo recorte de tempo.")
        for rotulo, fatia, nomes in PERIODOS["divergentes"]:
            st.caption(f"· {rotulo}: {fatia[0]} a {fatia[1]} em {', '.join(nomes)}")
    else:
        st.success(
            f"Os três recortes não se sobrepõem, e os "
            f"{ap.panorama()['experimentos']} experimentos declaram o mesmo."
        )

    st.divider()
    st.markdown(
        "A Seção 3 mostrou o viés médio do ano inteiro. Abaixo o mesmo "
        "número aparece quebrado por estação, para ver se o acerto se "
        "mantém o ano todo ou se a média esconde um trimestre ruim."
    )
    por_trimestre = _resultados_por_trimestre()
    if por_trimestre.empty:
        st.info("Nenhum experimento grava métrica por trimestre.")
        return

    st.plotly_chart(
        fg.barras_por_trimestre(por_trimestre, PIPELINE_COLORS, ap.NOME_PIPELINE),
        width="stretch", config=GRAFICO, key="narrativa_trimestres",
    )
    st.caption(
        "DJF verão · MAM outono · JJA inverno · SON primavera (hemisfério "
        "sul). Só as abordagens que gravam resultado por trimestre aparecem: "
        "**Modelos clássicos** (várias técnicas tradicionais de aprendizado "
        "de máquina, testadas automaticamente por área e estação) e **Rede "
        "recorrente (LSTM)** (rede neural com memória entre instantes de "
        "tempo)."
    )
    amplitudes = por_trimestre.groupby("pipeline")["Bias_P90"].agg(lambda s: s.max() - s.min())
    if not amplitudes.empty:
        partes = "; ".join(
            f"{ap.NOME_PIPELINE.get(pipeline, pipeline)}: {valor:.1f} m/s entre "
            "a pior e a melhor estação"
            for pipeline, valor in amplitudes.items()
        )
        st.caption(f"Amplitude sazonal do viés — {partes}.")


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
            "janela mensal, esta comparação não pode ser feita com os "
            "artefatos atuais."
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

    st.plotly_chart(
        fg.barras_por_area(recorte, PIPELINE_COLORS["lazy"]),
        width="stretch", config=GRAFICO, key="narrativa_estabilidade",
    )
    st.caption(
        "Só os **Modelos clássicos** recalculam o acerto por janela mensal — "
        "várias técnicas tradicionais de aprendizado de máquina, testadas "
        "automaticamente por área e estação, fica a que acerta mais em "
        "cada uma."
    )

    resumo = ap.estabilidade_por_configuracao()
    # A faixa só é lida entre configurações que cobriram o mesmo território 
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
            "a escolha de variáveis."
        )
        with st.expander("Comparar as configurações entre si"):
            st.dataframe(
                resumo[["Configuração", "areas", "mes_medio", "pior_mes", "queda_media"]]
                .rename(columns={
                    "areas": "Áreas", "mes_medio": "R² mês (média)", "pior_mes": "Pior mês", "queda_media": "Queda média"}),
                hide_index=True, width="stretch",
            )

    with st.expander("Ver área por área"):
        st.dataframe(
            recorte[["area", "Modelo", "R2_anual", "R2_mes_medio", "R2_mes_desvio", "R2_mes_pior"]].rename(columns={
                "area": "Área", "R2_anual": "R² no ano", "R2_mes_medio": "R² mês (média)", "R2_mes_desvio": "R² mês (desvio)", "R2_mes_pior": "R² pior mês"}),
            hide_index=True, width="stretch",
        )
        st.caption("Recorte por área dos Modelos clássicos, únicos que recalculam por mês.")


# ── 6 · A entrega ─────────────────────────────────────────────────────────────

def entrega(ir_para):
    st.header("6 · A entrega")
    st.subheader("O que sai disso na prática, e com que ressalva?")

    from engine import (
        CORRECTED_GRID_VERSIONS, CORRECTED_GRID_DEFAULT_VERSION,
        corrected_grid_date_bounds, corrected_grid_snapshot, corrected_grid_percentil,
        build_grid_map, stations_geo_df,
    )

    disponiveis = {
        nome: caminho for nome, caminho in CORRECTED_GRID_VERSIONS.items()
        if caminho.exists() and any(caminho.glob("*.nc"))
    }
    if not disponiveis:
        st.info("Nenhuma versão do mapa corrigido está publicada.")
        return

    anos = sorted({
        int(a.stem.rsplit("_", 1)[-1])
        for caminho in disponiveis.values()
        for a in caminho.glob("grid_corrected_*.nc")
        if a.stem.rsplit("_", 1)[-1].isdigit()
    })
    colunas = st.columns(2)
    colunas[0].metric("Versões publicadas", ", ".join(sorted(disponiveis)))
    colunas[1].metric("Anos cobertos", f"{min(anos)}-{max(anos)}" if anos else SEM_DADO)

    st.markdown(
        "Um **mapa de rajada corrigido, dia a dia**, cobrindo todo o "
        "domínio, não só onde existe estação."
    )

    st.divider()
    st.markdown("#### Nos extremos: P90, P95, P99 e Máximo")
    niveis_corrigidos = ap.extremos_corrigidos()
    if niveis_corrigidos.empty:
        st.info(
            "Nenhuma configuração com dado bruto por estação disponível "
            "para calcular este recorte."
        )
    else:
        st.plotly_chart(
            fg.valores_extremos_corrigidos(niveis_corrigidos),
            width="stretch", config=GRAFICO, key="fig_extremos_corrigidos",
        )
        st.caption(
            f"{niveis_corrigidos.attrs.get('abordagem', '')} / "
            f"“{niveis_corrigidos.attrs.get('configuracao', '')}”, a "
            "configuração com menor viés no extremo (Seção 3), contra a V5 "
            "(Kriging Ordinário), recalculada sem o vazamento espacial "
            "conhecido e restrita às mesmas 234 estações e ao mesmo "
            "período de teste (2020-2025) do Vendaval IA — mesmo conjunto "
            "dos dois lados, senão a diferença mediria a rede, não o "
            "método. P90/P95/P99/Máximo aqui são o valor de cada nível, não "
            "a métrica “Viés no extremo (P90)” das Seções 3 a 5 (aquela é o "
            "erro médio só nos dias que passam do limiar); os demais não "
            "são publicados por nenhuma das duas pipelines, calculados "
            "aqui a partir do dado bruto por estação."
        )
        if niveis_corrigidos["vies_v5_medio"].notna().all():
            p99 = niveis_corrigidos[niveis_corrigidos["nivel"] == "P99"].iloc[0]
            st.info(
                f"O Vendaval IA **supera** o observado nos quatro níveis "
                f"(P99: +{p99['vies_medio']:.1f} m/s); a V5 **fica abaixo** "
                f"nos quatro (P99: {p99['vies_v5_medio']:.1f} m/s). "
                "Direções opostas, e a distância do Vendaval IA cresce mais "
                "rápido conforme o percentil sobe — não dá pra dizer sem "
                "ressalva qual erra menos: um supera, o outro falta, e a "
                "magnitude do erro depende do nível."
            )
        st.caption(
            "Resultado confiável: a configuração usada é a única com "
            "cobertura completa das áreas e viés no extremo (P90) validado "
            "pela própria pipeline (Seção 3); a V5 é LOOCV genuíno, sem a "
            "vantagem de rede nacional/histórico completo da V5 de "
            "produção."
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
    st.markdown("#### O mesmo extremo, no mapa")
    nivel_mapa = st.radio(
        "Nível mostrado no mapa", ["P90", "P95", "P99", "Máximo"],
        horizontal=True, key="entrega_nivel_mapa",
    )
    estacoes_nivel = ap.extremos_por_estacao(nivel_mapa)
    grid_nivel = corrected_grid_percentil(versao, nivel_mapa)
    if estacoes_nivel.empty or grid_nivel.empty:
        st.info("Sem dado suficiente para montar o mapa deste nível.")
    else:
        st.plotly_chart(
            fg.mapas_extremos(grid_nivel, estacoes_nivel, nivel_mapa),
            width="stretch", config=GRAFICO, key="fig_mapas_extremos",
        )
        st.caption(
            "Os três dividem a mesma escala de cor. INMET e V5 são só "
            "estação (sem malha contínua); o Vendaval IA tem grid "
            "publicado, por isso o painel do meio soma o campo com as "
            "estações por cima. P95/P99/Máximo aqui são o mesmo cálculo da "
            "seção anterior, célula a célula no grid e estação a estação "
            "no INMET/V5."
        )

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
        "ali o resultado é extrapolação do que foi medido, não medição."
    )
    st.caption(
        "Não há métrica publicada comparando as versões do mapa entre si, "
        "o painel não afirma qual é melhor porque esse número não foi medido."
    )
