"""
Explorador — o painel de consulta técnica.

São as quatro telas originais do dashboard, preservadas: quem é do projeto
continua conseguindo cruzar qualquer combinação de pipeline, configuração,
cluster e trimestre. A diferença é que elas deixaram de ser a porta de
entrada: agora vivem atrás da narrativa, para quem já sabe o que procura.

O escopo global (pipelines, configurações, cluster, trimestre, estação) é lido
de `st.session_state`, onde a barra lateral de `app.py` o escreve — antes ele
vinha de variáveis de módulo compartilhadas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from engine import *  # noqa: F403 — motor completo: constantes, loaders e figuras
# `import *` não traz nomes iniciados por underscore; estes são auxiliares do
# motor que estas telas usam e precisam vir nomeados um a um.
from apuracao import NOME_ARM, NOME_METRICA  # noqa: F401 — rótulos em português
from engine import (  # noqa: F401
    _ablation_aggregate,
    _ablation_select_split,
    _best_combo_per_cluster,
    _geom_for_cluster_choice,
    _obs_pred_metrics,
    _resolve_all_season,
    _sdf_from_cluster_winners,
)


# NENHUMA e TODAS_AS_AREAS vêm do motor (`import *` acima) — não são
# redefinidos aqui de propósito: duas cópias do mesmo sentinela foi
# exatamente o que quebrou esta tela antes.


def _ajuda_das_versoes() -> str:
    """Descreve cada versão do mapa corrigido a partir dos arquivos que ela
    de fato tem. Antes este texto era digitado à mão e já divergia do
    conteúdo publicado."""
    partes = []
    for nome, caminho in CORRECTED_GRID_VERSIONS.items():  # noqa: F405
        anos = sorted(
            int(a.stem.rsplit("_", 1)[-1])
            for a in caminho.glob("grid_corrected_*.nc")
            if a.stem.rsplit("_", 1)[-1].isdigit()
        )
        if anos:
            partes.append(f"{nome}: {min(anos)}–{max(anos)} ({len(anos)} anos)")
    return " · ".join(partes) if partes else "Sem versão publicada."


def _escopo():
    """Lê da sessão o recorte escolhido na barra lateral."""
    pipes = st.session_state.get("global_pipelines", [])
    arms = st.session_state.get("global_arms", [])
    combos = [
        c for c in ablation_combos  # noqa: F405
        if c["pipeline"] in pipes and c["arm"] in arms
    ]
    estacao = st.session_state.get("global_station")
    return (
        combos,
        st.session_state.get("global_cluster"),
        st.session_state.get("global_season"),
        None if estacao in (None, NENHUMA) else estacao,
    )


@st.fragment
def _render_tab_global():
    selected_combos, global_cluster, global_season, global_station = _escopo()
    if not ablation_combos:
        st.info(
            "Nenhum resultado de experimento está publicado neste painel. "
            "Os artefatos são gerados no repositório de pesquisa e copiados "
            "para cá; enquanto isso não acontece, esta tela fica vazia.",
            icon="ℹ️",
        )
    else:
        all_results = load_all_ablation(ablation_combos)
        derived_quarterly = derive_quarterly_results(ablation_combos, all_results)
        if not derived_quarterly.empty:
            all_results = pd.concat([all_results, derived_quarterly], ignore_index=True)
        panel_clusters = (
            sorted(all_results["cluster_id"].unique(), key=str)
            if not all_results.empty else []
        )

        col_metric, col_cluster = st.columns([1, 1])
        with col_metric:
            metric = st.selectbox("Métrica", ABLATION_METRICS,
                                 format_func=lambda m: NOME_METRICA.get(m, m),
                                 key="ablation_metric")
        with col_cluster:
            panel_cluster_choice = st.selectbox(
                "Área",
                [TODAS_AS_AREAS] + [f"Área {c}" for c in panel_clusters],
                key="ablation_cluster",
                help=(
                    "O padrão mistura todas as áreas numa média ponderada, o "
                    "que pode esconder uma área que vai muito bem ou muito "
                    "mal sozinha — escolha uma para ver o número dela."
                ),
            )

        # Cards/barras/heatmap/tabela abaixo usam a visão agregada 'ALL'
        # (todos os trimestres combinados) — o detalhamento por trimestre
        # isolado fica no bloco "Per-quarter snapshot" logo abaixo, sem
        # precisar de um seletor pra trocar de trimestre.
        panel_season = "ALL"
        by_season = _resolve_all_season(all_results)
        if by_season.empty:
            st.info(
                "Nenhuma combinação tem dado para este recorte — as "
                "abordagens não cobrem os mesmos trimestres.",
                icon="ℹ️",
            )

        by_cluster = by_season
        if panel_cluster_choice != TODAS_AS_AREAS:
            chosen_cluster = panel_cluster_choice.removeprefix("Área ")
            by_cluster = by_season[by_season["cluster_id"].astype(str) == chosen_cluster]

        selected_rows = _ablation_select_split(by_cluster)
        summary_df = _ablation_aggregate(selected_rows, ["pipeline", "arm"])

        # ── Best per cluster (barras + mapa espacial) — lado a lado pra
        # economizar espaço vertical. O 1º mostra, por cluster, a MELHOR
        # combinação pipeline×configuração entre TODAS as testadas — não é
        # filtrado por "Cluster" acima (esse seletor foca 1 cluster; este
        # gráfico já compara todos). O 2º é a versão espacial do 1º: em vez
        # de barra, mostra o campo interpolado do erro do modelo (|Predito −
        # INMET observado|, só onde há observação real), "costurando" o
        # resultado de cada cluster com o combo que venceu ali.
        col_best_cluster, col_best_map = st.columns(2)
        with col_best_cluster:
            _best_cluster_fig = build_best_per_cluster_bars(all_results, metric, ablation_combos)
            if not _best_cluster_fig.data:
                st.caption("Sem dado para este recorte.")
            else:
                st.plotly_chart(
                    _best_cluster_fig, width="stretch", key="best_per_cluster_chart",
                )
        with col_best_map:
            _all_winners = _best_combo_per_cluster(all_results, metric)
            _best_map_sdf = _sdf_from_cluster_winners(
                ablation_combos, _all_winners, "Error (|Pred − Obs|)", "Mean",
            )
            if _best_map_sdf.empty or len(_best_map_sdf) < 2:
                st.caption("Sem dado para este recorte.")
            else:
                st.plotly_chart(
                    build_interp_map(_best_map_sdf, "IDW (original)", False, height=380),
                    width="stretch", key="best_per_cluster_map",
                )
        st.divider()

        # ── Best per cluster × trimestre — mapa espacial ─────────────────
        # Mesma ideia do mapa acima, mas 1 mapa por trimestre (DJF/MAM/JJA/
        # SON), cada um "costurado" com o vencedor daquele cluster NAQUELE
        # trimestre especificamente (não o vencedor do ano inteiro).
        _quarters_for_maps = [
            s for s in ("DJF", "MAM", "JJA", "SON") if s in all_results["season"].unique()
        ]
        if not _quarters_for_maps:
            st.caption("Nenhuma abordagem sincronizou resultado por trimestre — só o agregado do ano.")
        else:
            # Calcula os 4 sdf's ANTES de desenhar, pra poder compartilhar a
            # mesma escala de cor (cmin/cmax) e mostrar só 1 colorbar — mesmo
            # padrão já usado no comparativo de 5 painéis da Seção 2.
            _quarter_sdfs = []
            for quarter in _quarters_for_maps:
                q_winners = _best_combo_per_cluster(all_results, metric, season=quarter)
                q_sdf = _sdf_from_cluster_winners(
                    ablation_combos, q_winners, "Error (|Pred − Obs|)", "Mean", season=quarter,
                )
                _quarter_sdfs.append((quarter, q_sdf))

            _quarter_vals = pd.concat(
                [s["value"] for _, s in _quarter_sdfs if s is not None and not s.empty],
                ignore_index=True,
            )
            _q_cmin = float(_quarter_vals.min()) if not _quarter_vals.empty else None
            _q_cmax = float(_quarter_vals.max()) if not _quarter_vals.empty else None
            _q_last_valid = max(
                (i for i, (_, s) in enumerate(_quarter_sdfs) if s is not None and len(s) >= 2),
                default=-1,
            )

            quarter_map_cols = st.columns(len(_quarters_for_maps))
            for i, (q_col, (quarter, q_sdf)) in enumerate(zip(quarter_map_cols, _quarter_sdfs)):
                with q_col:
                    st.caption(
                        f"{quarter} ({len(q_sdf)} estações)",
                        help=(
                            "A contagem é de estações com ao menos uma "
                            "previsão válida neste trimestre — não o tamanho "
                            "da rede. Nem toda estação reporta o ano inteiro "
                            "(falha de sensor, desativação no meio do ano), "
                            "então o número muda entre DJF/MAM/JJA/SON mesmo "
                            "com a rede sendo sempre a mesma. DJF ainda "
                            "atravessa a virada do ano (dez+jan+fev), ao "
                            "contrário dos outros três, o que aumenta a "
                            "diferença."
                        ),
                    )
                    if q_sdf.empty or len(q_sdf) < 2:
                        st.caption("Sem dado.")
                    else:
                        st.plotly_chart(
                            build_interp_map(
                                q_sdf, "IDW (original)", False, height=340,
                                cmin_override=_q_cmin, cmax_override=_q_cmax,
                                show_colorbar=(i == _q_last_valid),
                            ),
                            width="stretch", key=f"best_per_cluster_map_{quarter}",
                        )
        st.divider()

        # ── Cards de métrica — melhor configuração por pipeline vs. "original" ──
        direction = ABLATION_METRIC_DIRECTIONS.get(metric, "higher")
        delta_df = build_ablation_delta_table(summary_df, metric)
        if not delta_df.empty:
            card_cols = st.columns(len(ABLATION_PIPELINES))
            for card_col, pipeline in zip(card_cols, ABLATION_PIPELINES):
                pipeline_label = PIPELINE_LABELS.get(pipeline, pipeline)
                sub = delta_df[delta_df["Pipeline"] == pipeline_label]
                if sub.empty:
                    card_col.metric(pipeline_label, "—")
                    continue
                if direction == "higher":
                    best_row = sub.loc[sub[metric].idxmax()]
                    delta_val, delta_color, delta_label = best_row[f"Δ{metric}"], "normal", f"Δ{metric}"
                elif direction == "lower":
                    best_row = sub.loc[sub[metric].idxmin()]
                    delta_val, delta_color, delta_label = best_row[f"Δ{metric}"], "inverse", f"Δ{metric}"
                else:  # "zero" (Bias/Bias_P90) — mais perto de zero é melhor, sinal
                    # bruto do delta não distingue melhora de piora; compara |valor|.
                    best_row = sub.loc[sub[metric].abs().idxmin()]
                    base_rows = sub[sub["Arm"] == "original"]
                    delta_val = (
                        abs(best_row[metric]) - abs(base_rows.iloc[0][metric])
                        if not base_rows.empty else float("nan")
                    )
                    delta_color, delta_label = "inverse", f"Δ|{metric}|"
                card_col.metric(
                    f"{pipeline_label} — best: {best_row['Arm']}",
                    f"{best_row[metric]:.3f}",
                    delta=f"{delta_val:+.3f} {delta_label} vs. original" if delta_val == delta_val else None,
                    delta_color=delta_color,
                )

        _quarter_pipeline_fig = build_ablation_bars_by_quarter(all_results, metric)
        if _quarter_pipeline_fig.data:
            st.plotly_chart(
                _quarter_pipeline_fig, width="stretch",
                key="ablation_bars_by_quarter_chart",
            )
        # delta_df já calculado acima pros cards de métrica — reaproveitado
        # aqui, não recomputado.

        def _color_ablation_delta(val, metric_name):
            direction = ABLATION_METRIC_DIRECTIONS.get(metric_name, "higher")
            if direction == "higher":
                return "color: green" if val > 0 else "color: red" if val < 0 else ""
            if direction == "lower":
                return "color: green" if val < 0 else "color: red" if val > 0 else ""
            return "color: red" if abs(val) > 1e-9 else ""

        delta_col = f"Δ{metric}"
        styled = delta_df.style
        if delta_col in delta_df.columns:
            styled = styled.map(lambda v: _color_ablation_delta(v, metric), subset=[delta_col])
        st.dataframe(
            styled, width="stretch", hide_index=True, key="ablation_delta_table",
        )

        with st.expander("Modelos clássicos — os 5 melhores por configuração"):
            lazy_top5_df = build_lazy_top5_per_arm(ablation_combos, panel_season, panel_cluster_choice)
            st.plotly_chart(
                build_lazy_top5_chart(lazy_top5_df),
                width="stretch", key="lazy_top5_chart",
            )
            if not lazy_top5_df.empty:
                st.dataframe(
                    lazy_top5_df[["arm", "rank", "Model", "R2"]].rename(
                        columns={"arm": "Configuração", "rank": "Posição"}
                    ),
                    width="stretch", hide_index=True, key="lazy_top5_table",
                )


# ── Seção 2: Spatial & Temporal Error Inspector ───────────────────────────────

@st.fragment
def _render_tab_inspector():
    selected_combos, global_cluster, global_season, global_station = _escopo()
    _obs_sdf = None
    _shared_cmin = _shared_cmax = None
    _arms_with_data = [a for a in ABLATION_ARMS if any(c["arm"] == a for c in ablation_combos)]
    if not _arms_with_data:
        st.info("Nenhum resultado sincronizado ainda.", icon="ℹ️")
    else:
        mc0, mc1, mc2, mc3 = st.columns(4)
        with mc0:
            multi_arm = st.selectbox("Configuração", _arms_with_data,
                                       format_func=lambda a: NOME_ARM.get(a, a),
                                       key="multi_map_arm")
        with mc1:
            multi_metric = st.selectbox("Métrica", list(INTERP_METRICS), key="multi_map_metric")
        with mc2:
            multi_method = st.selectbox("Suavização", list(INTERP_METHODS), key="multi_map_method")

        def _combo_for(pipeline: str, arm: str) -> dict | None:
            return next(
                (c for c in ablation_combos if c["pipeline"] == pipeline and c["arm"] == arm),
                None,
            )

        _mlp_combo = _combo_for("mlp", multi_arm)
        _lazy_combo = _combo_for("lazy", multi_arm)
        _lstm_combo = _combo_for("lstm", multi_arm)

        # Observado vem direto do INMET_Stratified.nc bruto (271 estações atuais),
        # não de um arm/pipeline já sincronizado — assim não fica preso ao
        # station-count antigo enquanto lazy/mlp/lstm ainda não foram re-treinados
        # com a rede expandida.
        _obs_sdf = load_inmet_observed(multi_metric, stations_geo_df)

        def _sdf_for_combo(combo: dict | None, value_label: str) -> pd.DataFrame | None:
            """Usa predictions_by_station.csv (granularidade real de estação)
            quando a pipeline já sincronizou esse arquivo pra esse arm; senão
            cai pro agregado por cluster_id espalhado (aggregate_cluster_values)
            — fallback só pra arms ainda não re-treinados com a granularidade
            nova. Antes, LazyPredict/LSTM sempre caíam no broadcast por
            cluster; agora que as duas também salvam predictions_by_station.csv,
            usam a mesma granularidade real que o MLP já tinha."""
            if combo is None:
                return None
            if (Path(combo["dir"]) / "predictions_by_station.csv").exists():
                return aggregate_station_values(combo["dir"], value_label, multi_metric)
            return aggregate_cluster_values(combo["dir"], value_label, multi_metric, stations_geo_df)

        _lazy_sdf = _sdf_for_combo(_lazy_combo, "Predicted")
        _mlp_sdf = _sdf_for_combo(_mlp_combo, "Predicted")
        _lstm_sdf = _sdf_for_combo(_lstm_combo, "Predicted")
        # ERA5 bruto (sem correção) — só o MLP sincroniza era5_wind_mag_max por
        # observação (predictions_by_station.csv); LazyPredict/LSTM só têm
        # y_true/y_pred no predictions.parquet. O valor em si não depende do
        # pipeline (é o insumo de entrada, não uma saída de modelo), então
        # reusar a fonte do MLP aqui não é diferente de qualquer outra feature
        # de entrada — só precisa que o MLP esteja sincronizado pra config atual.
        _era5_sdf = (
            aggregate_station_values(_mlp_combo["dir"], "ERA5 raw", multi_metric)
            if _mlp_combo else None
        )

        multi_panels = [
            ("Observed (INMET)", _obs_sdf),
            ("ERA5", _era5_sdf),
            ("ML Models", _lazy_sdf),
            ("MLP", _mlp_sdf),
            ("LSTM (TF dual-head)", _lstm_sdf),
        ]

        _all_cluster_ids = sorted({
            str(cid)
            for _, sdf in multi_panels if sdf is not None and not sdf.empty
            for cid in sdf["cluster_id"].unique()
        }, key=str)
        with mc3:
            multi_cluster = st.selectbox(
                "Área em foco", [TODAS_AS_AREAS] + [f"Área {c}" for c in _all_cluster_ids],
                key="multi_map_cluster_focus",
            )
        _chosen = (
            None if multi_cluster == TODAS_AS_AREAS
            else multi_cluster.removeprefix("Área ")
        )
        if _chosen is not None:
            multi_panels = [
                (
                    label,
                    sdf[sdf["cluster_id"].astype(str) == _chosen] if sdf is not None and not sdf.empty else sdf,
                )
                for label, sdf in multi_panels
            ]

        _all_vals = pd.concat(
            [sdf["value"] for _, sdf in multi_panels if sdf is not None and not sdf.empty],
            ignore_index=True,
        )
        _shared_cmin = float(_all_vals.min()) if not _all_vals.empty else None
        _shared_cmax = float(_all_vals.max()) if not _all_vals.empty else None

        # Junta o valor de cada estação em TODOS os painéis num único texto
        # de hover, pra comparar Observado/ERA5/LazyPredict/MLP/LSTM sem
        # precisar passar o mouse painel por painel.
        _values_by_station: dict[str, dict[str, float]] = {}
        for label, sdf in multi_panels:
            if sdf is None or sdf.empty:
                continue
            for est, v in zip(sdf["estacao"], sdf["value"]):
                _values_by_station.setdefault(est, {})[label] = v
        _hover_extra_by_station = {
            est: "<br>".join(
                f"{label}: {values[label]:.2f} m/s"
                for label, _ in multi_panels
                if label in values and np.isfinite(values[label])
            )
            for est, values in _values_by_station.items()
        }

        # Recorta o campo interpolado na forma real da bacia/cluster (em vez
        # de um retângulo) e enquadra o mapa exatamente nessa região — os 4
        # painéis compartilham a MESMA geometria/janela, senão cada um
        # ajustaria seu próprio zoom a partir das suas próprias estações.
        _mask_geom = _geom_for_cluster_choice(
            _chosen if multi_cluster != TODAS_AS_AREAS else None
        )
        _mgw, _mgs, _mge, _mgn = _mask_geom.bounds

        # Garante que NENHUMA estação fique cortada fora da janela do mapa: o
        # enquadramento é a UNIÃO do polígono da bacia/cluster com a extensão
        # real das estações de todos os painéis (uma estação raramente cai
        # ligeiramente fora do próprio polígono do cluster, perto da borda).
        _sta_lons = pd.concat(
            [sdf["longitude"] for _, sdf in multi_panels if sdf is not None and not sdf.empty],
            ignore_index=True,
        )
        _sta_lats = pd.concat(
            [sdf["latitude"] for _, sdf in multi_panels if sdf is not None and not sdf.empty],
            ignore_index=True,
        )
        if not _sta_lons.empty:
            _mgw = min(_mgw, float(_sta_lons.min()))
            _mge = max(_mge, float(_sta_lons.max()))
            _mgs = min(_mgs, float(_sta_lats.min()))
            _mgn = max(_mgn, float(_sta_lats.max()))

        _bpad = 0.3
        _shared_bounds = {
            "west": _mgw - _bpad, "east": _mge + _bpad,
            "south": _mgs - _bpad, "north": _mgn + _bpad,
        }

        # Só 1 colorbar pra linha inteira (no último painel com dado válido) — os
        # 4 painéis já compartilham a mesma escala (_shared_cmin/_shared_cmax),
        # repetir a barra 4x só desperdiçava espaço horizontal sem info nova.
        _last_valid_idx = max(
            (i for i, (_, sdf) in enumerate(multi_panels) if sdf is not None and not sdf.empty and len(sdf) >= 2),
            default=-1,
        )
        map_cols = st.columns(len(multi_panels))
        for i, (map_col, (label, sdf)) in enumerate(zip(map_cols, multi_panels)):
            with map_col:
                st.markdown(f"**{label}**")
                if sdf is None:
                    st.info(f"Não sincronizado para “{NOME_ARM.get(multi_arm, multi_arm)}”.", icon="ℹ️")
                elif sdf.empty or len(sdf) < 2:
                    st.info("Estações insuficientes para interpolar.", icon="ℹ️")
                else:
                    st.plotly_chart(
                        build_interp_map(
                            sdf, multi_method, False,
                            cmin_override=_shared_cmin, cmax_override=_shared_cmax, height=380,
                            mask_geom=_mask_geom, bounds_override=_shared_bounds,
                            show_colorbar=(i == _last_valid_idx),
                            hover_extra_by_station=_hover_extra_by_station,
                            stations_only=(label == "Observed (INMET)"),
                        ),
                        width="stretch", key=f"multi_map_{label}",
                    )

    st.divider()

    if not selected_combos:
        st.info("Escolha ao menos uma abordagem e uma configuração na barra lateral.", icon="ℹ️")
    else:
        st.caption(
            "Comparando: " + ", ".join(c["label"] for c in selected_combos)
            + f" — Área {global_cluster} — trimestre {global_season}"
        )

        _value_by_estacao = (
            _obs_sdf.set_index("estacao")["value"]
            if _obs_sdf is not None and not _obs_sdf.empty else None
        )
        col_map, col_density = st.columns(2)
        with col_map:
            event = st.plotly_chart(
                build_inspector_map(
                    stations_geo_df, global_station,
                    value_by_estacao=_value_by_estacao,
                    cmin=_shared_cmin, cmax=_shared_cmax,
                ),
                width="stretch", on_select="rerun", key="map_chart",
                selection_mode="points",
            )
            if event and hasattr(event, "selection") and event.selection.points:
                pt = event.selection.points[0]
                cd = pt.get("customdata")
                if cd and len(cd) >= 2:
                    clicked_station = str(cd[0])
                    if clicked_station in stations_geo_df["estacao"].tolist():
                        st.session_state["_pending_station"] = clicked_station
                        st.rerun()

        with col_density:
            st.plotly_chart(
                build_prediction_density(selected_combos, global_cluster),
                width="stretch", key="density_chart",
            )

        col_ts, col_box = st.columns(2)
        mlp_selected = [c for c in selected_combos if c["pipeline"] == "mlp"]
        other_selected = [c for c in selected_combos if c["pipeline"] != "mlp"]

        with col_ts:
            if mlp_selected:
                mlp_preds_df = load_mlp_predictions_by_station(mlp_selected[0]["dir"])
                st.plotly_chart(
                    build_residual_timeseries(global_station, mlp_preds_df),
                    width="stretch", key="residual_timeseries_chart",
                )
            else:
                st.info("Selecione a rede neural (MLP) para ver a série de resíduo real.", icon="ℹ️")

        with col_box:
            if other_selected:
                st.plotly_chart(
                    build_residual_boxplot(other_selected, global_cluster),
                    width="stretch", key="residual_boxplot_chart",
                )
            else:
                st.info("Selecione os modelos clássicos e/ou a rede recorrente (LSTM) para ver a distribuição do resíduo.", icon="ℹ️")


# ── Seção 3: Model Diagnostics & Explainability ───────────────────────────────

@st.fragment
def _render_tab_diag():
    selected_combos, global_cluster, global_season, global_station = _escopo()
    if not selected_combos:
        st.info("Escolha ao menos uma abordagem e uma configuração na barra lateral.", icon="ℹ️")
    else:
        for combo in selected_combos:
            st.subheader(combo["label"])
            preds = load_ablation_predictions(combo["dir"])
            if global_cluster is not None and not preds.empty and "cluster_id" in preds.columns:
                preds_c = preds[preds["cluster_id"] == global_cluster]
            else:
                preds_c = preds

            # Cards + scatter/OLS pra TODAS as pipelines (inclusive LSTM) — o
            # R² sozinho (Seção 1) não explica POR QUE ele é negativo pra
            # alguns arms; ver os pontos reais + a reta de regressão contra a
            # 1:1 é o que realmente mostra a causa (viés sistemático,
            # variância baixa demais, outliers, etc.).
            combo_metrics = _obs_pred_metrics(preds_c["y_true"], preds_c["y_pred"]) if not preds_c.empty else None
            if combo_metrics is not None and combo_metrics["n"] > 0:
                mcol0, mcol1, mcol2, mcol3 = st.columns(4)
                mcol0.metric("Bias", f"{combo_metrics['bias']:+.2f} m/s")
                mcol1.metric("RMSE", f"{combo_metrics['rmse']:.2f} m/s")
                mcol2.metric("MAE", f"{combo_metrics['mae']:.2f} m/s")
                mcol3.metric("Corr", f"{combo_metrics['corr']:.2f}")
            st.plotly_chart(
                build_ablation_scatter_with_ols(preds_c, combo["pipeline"], combo["arm"], metrics=combo_metrics),
                width="stretch", key=f"diag_scatter_{combo['pipeline']}_{combo['arm']}",
            )

            if combo["pipeline"] == "lstm":
                histories = load_lstm_histories(combo["dir"])
                season_for_lstm = global_season if global_season != "ALL" else "DJF"
                st.plotly_chart(
                    build_lstm_loss_curve(histories, global_cluster, season_for_lstm),
                    width="stretch", key=f"diag_loss_{combo['arm']}",
                )

            if combo["pipeline"] == "mlp":
                importance_df = load_mlp_importance(combo["dir"])
                if not importance_df.empty and global_cluster is not None:
                    st.plotly_chart(
                        build_importance(importance_df, global_cluster),
                        width="stretch", key=f"diag_importance_{combo['arm']}",
                    )

            st.divider()


# ── Seção 4: Corrected Grid Explorer ──────────────────────────────────────────

@st.fragment
def _render_tab_grid():
    selected_combos, global_cluster, global_season, global_station = _escopo()
    _available_versions = [
        v for v in CORRECTED_GRID_VERSIONS
        if CORRECTED_GRID_VERSIONS[v].exists()
    ]
    if not _available_versions:
        st.info(
            "Nenhuma versão do mapa corrigido está publicada neste painel. "
            "O mapa é gerado no repositório de pesquisa e copiado para cá.",
            icon="ℹ️",
        )
        return

    grid_version = st.selectbox(
        "Base de dados", _available_versions,
        index=_available_versions.index(CORRECTED_GRID_DEFAULT_VERSION)
        if CORRECTED_GRID_DEFAULT_VERSION in _available_versions else 0,
        key="grid_version_select",
        help=_ajuda_das_versoes(),
    )

    _grid_bounds = corrected_grid_date_bounds(grid_version)
    if _grid_bounds is None:
        st.info(f"Nenhum mapa corrigido encontrado para a versão “{grid_version}”.", icon="ℹ️")
    else:
        _gmin, _gmax = _grid_bounds

        # Seletor de estação PRÓPRIO (não usa o global_station da barra
        # lateral): esse é escopado ao cluster selecionado ali (relevante só
        # pra matriz de ablation), enquanto o grid é cluster-agnóstico —
        # reusar global_station faria a seleção "sumir" sempre que a estação
        # escolhida não pertencesse ao cluster atual da barra lateral.
        _grid_all_stations = stations_geo_df["estacao"].tolist()
        col_gdate, col_gstation = st.columns([1, 2])
        with col_gdate:
            grid_date = st.date_input(
                "Data do mapa", value=pd.Timestamp("2020-05-15"),
                min_value=_gmin.date(), max_value=_gmax.date(),
                key="grid_map_date",
            )
        with col_gstation:
            grid_station_choice = st.selectbox(
                "Estação", [NENHUMA] + _grid_all_stations,
                key="grid_station_select",
            )
        grid_station = None if grid_station_choice == NENHUMA else grid_station_choice

        grid_snapshot = corrected_grid_snapshot(str(grid_date), grid_version)

        col_map1, col_map2 = st.columns(2)
        with col_map1:
            st.plotly_chart(
                build_grid_map(
                    grid_snapshot, "ws_original", stations_geo_df,
                    grid_station, "ERA5", show_colorbar=False,
                ),
                width="stretch", key="grid_map_original",
            )
        with col_map2:
            st.plotly_chart(
                build_grid_map(
                    grid_snapshot, "rajada_max_corrigida", stations_geo_df,
                    grid_station, "AI", show_colorbar=True,
                ),
                width="stretch", key="grid_map_corrected",
            )

        if grid_station is None:
            st.info("Escolha uma estação acima para ver a série dela.", icon="ℹ️")
        else:
            _st_row = stations_geo_df[stations_geo_df["estacao"] == grid_station]
            if _st_row.empty:
                st.warning(f"A estação {grid_station} não consta no catálogo de estações.")
            else:
                _st_lat = float(_st_row["latitude"].iloc[0])
                _st_lon = float(_st_row["longitude"].iloc[0])
                grid_series = corrected_grid_station_series(_st_lat, _st_lon, grid_version)
                inmet_series = load_inmet_station_series(grid_station)
                st.plotly_chart(
                    build_grid_station_timeseries(
                        grid_station, grid_series, inmet_series, grid_date.year,
                    ),
                    width="stretch", key="grid_station_timeseries",
                )

        st.divider()
        extreme_metric = st.selectbox(
            "Métrica", list(INTERP_METRICS),
            index=list(INTERP_METRICS).index("P95"),
            key="grid_extreme_metric",
        )

        _grid_pct = grid_station_percentile_values(
            extreme_metric, stations_geo_df, grid_version,
        ).dropna(subset=["era5_original", "era5_corrected"])
        _inmet_pct = load_inmet_observed(extreme_metric, stations_geo_df)
        _grid_pct = _grid_pct.merge(
            _inmet_pct[["estacao", "value"]].rename(columns={"value": "inmet_value"}),
            on="estacao", how="inner",
        ).dropna(subset=["inmet_value"])

        if len(_grid_pct) < 2:
            st.info("Estações insuficientes para montar os mapas.", icon="ℹ️")
        else:
            # Valor de vento (não erro) das duas bases no percentil escolhido —
            # mais interpretativo: mostra diretamente "quanto vento" cada base
            # captura na cauda extrema, em vez de uma métrica abstrata de erro.
            # Os CÍRCULOS de estação mostram o valor REAL do INMET (verdade-
            # terreno), não o pixel do ERA5 — só o campo de fundo (raster IDW)
            # vem do ERA5; assim dá pra comparar visualmente modelo vs. real.
            _inmet_vals = _grid_pct["inmet_value"].to_numpy(float)
            _val_orig_sdf = _grid_pct[["estacao", "latitude", "longitude", "cluster_id", "era5_original"]].rename(
                columns={"era5_original": "value"},
            )
            _val_corr_sdf = _grid_pct[["estacao", "latitude", "longitude", "cluster_id", "era5_corrected"]].rename(
                columns={"era5_corrected": "value"},
            )
            _val_all = pd.concat(
                [_val_orig_sdf["value"], _val_corr_sdf["value"], pd.Series(_inmet_vals)], ignore_index=True,
            )
            _val_cmin, _val_cmax = float(_val_all.min()), float(_val_all.max())

            col_val1, col_val2 = st.columns(2)
            with col_val1:
                st.markdown(f"**ERA5 bruto e INMET observado — {extreme_metric}**")
                st.plotly_chart(
                    build_interp_map(
                        _val_orig_sdf, "IDW (original)", False,
                        cmin_override=_val_cmin, cmax_override=_val_cmax,
                        height=380, show_colorbar=False,
                        station_val_override=_inmet_vals,
                    ),
                    width="stretch", key="grid_extreme_value_original",
                )
            with col_val2:
                st.markdown(f"**Corrigido pelos modelos — {extreme_metric}**")
                st.plotly_chart(
                    build_interp_map(
                        _val_corr_sdf, "IDW (original)", False,
                        cmin_override=_val_cmin, cmax_override=_val_cmax,
                        height=380, show_colorbar=True,
                    ),
                    width="stretch", key="grid_extreme_value_corrected",
                )


