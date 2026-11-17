# Grade v1.2: previsões por modelo

Dois arquivos, um por região, com a rajada máxima diária prevista na grade ERA5 (2000-2025):

| Arquivo | Região | Grade |
|---|---|---|
| `pred_c1_c2_c3_c4.nc` | RS/SC, áreas 1 a 4 | 32 × 37, 0,25° |
| `pred_c12.nc` | área 12 | 22 × 22, 0,25° |

Variáveis: `era5`, `ml_local`, `ml_global_trimestral`, `ml_global_anual`, `lstm_local`, `lstm_global_anual` e `cluster_id` (-1 fora da malha).

## Compressão

Os arquivos originais (107 MB e 27,5 MB) foram regravados com as previsões em inteiros de 16 bits, resolução de 0,01 m/s, e compressão zlib nível 9. Tamanho final 46,4 MB e 12,1 MB. O erro máximo medido contra o original é 0,005 m/s e as posições sem dado são idênticas. O `xarray` converte de volta para metros por segundo ao abrir.

## Dentro e fora da amostra

Dias de 2000-2024 nos meses de treino e validação foram vistos pelos modelos nas células com estação. Fora da amostra ficam só janeiro, abril, julho e outubro de 2000-2024 e todo o ano de 2025.
