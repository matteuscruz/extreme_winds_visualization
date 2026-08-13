![Header](figs/extreme_winds.png)

# IRC Vendaval — Dashboard de Resultados

Visualização interativa da correção de viés de rajadas de vento extremo
no Sul do Brasil. Os modelos (LazyPredict, MLP, LSTM) foram treinados com ERA5 para corrigir
sistematicamente o viés em relação às observações INMET.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estrutura

```
├── app.py                        # Dashboard Streamlit
├── requirements.txt
├── artifacts/
│   ├── mlp_clusters/             # Resultados do MLP com extreme weighting
│   │   ├── mlp_cluster_results.csv
│   │   ├── feature_importance.csv
│   │   ├── predictions_by_station.csv
│   │   └── stations_metadata.csv
│   └── lazy_clusters/            # Screening LazyPredict (43 modelos)
│       └── lazy_cluster_results.csv
└── dataset/
    └── shp/                      # Polígonos dos 6 clusters espaciais
```

## Abas do dashboard

**Global Comparison Panel**
- Visão agregada (All clusters) e por cluster das melhores combinações de pipeline × configuração (gráficos de barras e mapas espaciais)
- Detalhamento de métricas por trimestre (DJF, MAM, JJA, SON)
- Tabela de deltas de regressão (R², RMSE, Bias) vs ERA5 bruto
- Ranking top 5 de modelos do screening de machine learning

**Spatial & Temporal Error Inspector**
- Mapas espaciais interativos (IDW, Nearest) comparando lado a lado: Observações INMET, ERA5 original e os resultados preditivos (Lazy, MLP, LSTM)
- Inspeção por métricas de erro temporais (P90, Max, Mean, etc)

**Model Diagnostics & Explainability**
- Gráficos de dispersão (scatter plot) e regressão OLS por cluster (Predito vs INMET Observado)
- Histórico de treinamento (Loss Curves) para arquiteturas deep learning (LSTM)
- Importância de variáveis (Feature Permutation Importance) para os modelos estruturados (MLP)

**AI database**
- Explorador do banco de dados unificado final (Corrected Grid Explorer)
- Séries temporais ponto-a-ponto comparando a rajada diária observada, ERA5 bruto e corrigido
