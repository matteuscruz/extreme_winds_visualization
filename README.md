![Cabeçalho](figs/extreme_winds.png)

# Rajada de vento extremo — correção de viés

Painel de resultados da correção de viés de rajada de vento extremo no Sul do
Brasil. O ERA5 subestima sistematicamente a rajada forte; este projeto treina
modelos para corrigir esse erro, usando as estações do INMET como verdade.

O painel é uma **vitrine de resultado**: ele carrega artefatos já calculados e
não treina nada.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Como a tela é organizada

A navegação é dividida entre **narrativa** e **consulta**, porque os dois
públicos precisam de coisas diferentes.

A narrativa é para quem chega sem contexto. São cinco seções, cada uma
abrindo com a pergunta que responde:

| Seção | Pergunta |
|---|---|
| 1 · O problema | Por que a rajada de vento do ERA5 não serve como está? |
| 2 · O experimento | O que foi testado — e o que significa cada configuração? |
| 3 · O resultado | Qual combinação venceu, e em quê? |
| 4 · A prova | A correção continua valendo em anos que o modelo nunca viu? |
| 5 · A entrega | O que sai disso na prática? |

O **Explorador** é para quem é do projeto: as quatro telas de consulta
originais, preservadas inteiras — comparação geral, erro no espaço e no tempo,
diagnóstico dos modelos e mapa corrigido. Quem quiser cruzar qualquer
combinação de abordagem, configuração, área e trimestre continua conseguindo.

## Como o código é dividido

```
app.py           # casca: configuração da página, navegação e barra lateral
narrativa.py     # as cinco seções da história
explorador.py    # as quatro telas de consulta técnica
apuracao.py      # os números da narrativa, calculados dos artefatos
engine.py        # motor: carga de dado, métricas e figuras
theme.py         # tokens de cor
```

**Nenhum número mostrado na tela é digitado no código.** Todos são calculados
dos artefatos no momento em que a página carrega, para que o painel não passe
a mentir quando a pipeline for reexecutada. Onde um número não existe no
artefato, a tela descreve a lacuna em vez de preencher com um valor plausível.

## De onde vêm os dados

```
artifacts/
├── ablation/<abordagem>/<configuração>/   # 3 abordagens × 6 configurações
│   ├── results.parquet                    # métricas por área, trimestre e split
│   ├── predictions.parquet
│   ├── predictions_by_station.csv
│   ├── run_meta.json                      # variáveis usadas e recorte de tempo
│   ├── mlp_cluster_results.csv            # só MLP — inclui o baseline do ERA5
│   ├── lazy_cluster_results.csv           # só modelos clássicos — todos os candidatos
│   └── histories.json                     # só LSTM — curva de treino
├── corrected_grid/<versão>/               # o produto final: mapa corrigido, um .nc por ano
dataset/
├── raw/INMET_Stratified.nc                # observação, fonte do painel "Observado"
└── shp/                                   # polígonos das áreas
```

Os artefatos são gerados no repositório de pesquisa e copiados para cá. O
recorte de treino, validação e teste é lido de `run_meta.json`, não fixado no
código.

## Ressalvas que o painel declara sozinho

O painel checa e avisa na tela, em vez de deixar o leitor tropeçar:

- **Experimentos com cobertura parcial** aparecem marcados e ficam fora dos
  gráficos comparativos. Uma média calculada sobre menos áreas pode parecer
  melhor só por deixar as áreas difíceis de fora.
- **Configurações com a mesma lista de variáveis e os mesmos dados de treino**
  são sinalizadas como duplicatas: pelo que está nos metadados, são a mesma
  coisa com dois nomes.
- **Divergência de recorte de tempo** entre experimentos, se houver, é
  listada antes de qualquer comparação.
- **Contagem de estações** é mostrada separada por fonte, porque as fontes não
  concordam entre si — a rede catalogada é maior do que a que cada experimento
  conseguiu usar.
