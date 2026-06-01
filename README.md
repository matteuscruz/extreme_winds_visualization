# IRC Vendaval — Dashboard de Resultados

Visualização interativa da correção de viés de rajadas de vento extremo
no Sul do Brasil. Os modelos foram treinados com ERA5 para corrigir
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

**Explorador MLP**
- Mapa do Sul do Brasil com polígonos de cluster e estações clicáveis
- Série temporal observado × predito × ERA5 por estação
- Distribuição (violin) e scatter com regressão OLS por cluster
- Métricas de qualidade (R², RMSE, Bias@P90) comparando MLP vs ERA5 bruto
- Importância de features por permutação

**Screening LazyPredict**
- Ranking dos 43 modelos avaliados por cluster com MLPRegressor destacado
- Tabela completa com R², RMSE e tempo de treino por modelo
