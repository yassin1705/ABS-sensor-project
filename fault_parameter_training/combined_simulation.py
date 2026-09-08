"""Rejoue une simulation dans l'ancien MSP et la nouvelle fusion CNN/GRU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model_training.data import SPEED_COLUMNS, VALID_COLUMNS, WHEELS
from model_training.realtime_spc_replay import (
    DEFAULT_LIMITS,
    FAULT_COLUMNS,
    load_control_limits,
    plot_static_replay,
    replay_simulation,
    select_faulty_demo_simulation_id,
    summarize_replay,
)
from model_training.spc_calibration import (
    CSV_COLUMNS,
    DEFAULT_CHECKPOINT,
    DEFAULT_METADATA,
    PROJECT_ROOT,
    PhysicalUnitGRUPredictor,
    iter_selected_simulations,
)

from .inference import CNNGRUFusionPredictor


DEFAULT_FUSION_EXPERIMENTS = Path(
    os.environ.get(
        "PFA_FAULT_EXPERIMENTS",
        PROJECT_ROOT / "fault_parameter_training" / "experiments",
    )
)
DEFAULT_FUSION_CACHE = Path(
    os.environ.get(
        "PFA_FAULT_CACHE",
        PROJECT_ROOT / "fault_parameter_training" / "cache" / "fault_parameter_dataset.npz",
    )
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "fault_parameter_training" / "combined_replay"


def latest_faulty_pair() -> tuple[Path, Path]:
    results = PROJECT_ROOT / "ABS_SoH_Simulator" / "simulation_results"
    datasets = sorted(results.glob("abs_faulty_braking_dataset_*.csv"))
    if not datasets:
        raise FileNotFoundError("Aucun dataset fautif trouve.")
    dataset = datasets[-1]
    manifest = results / dataset.name.replace("_dataset_", "_manifest_")
    if not manifest.exists():
        raise FileNotFoundError(f"Manifeste fautif absent : {manifest}")
    return dataset, manifest


def old_spc_class(wheel_summary: dict[str, object]) -> str:
    if int(wheel_summary["localized_suspect_sample_count"]) > 0:
        return "SUSPECTED_FAULTY"
    if int(wheel_summary["alarm_sample_count"]) > 0:
        return "CROSS_ALARM"
    return "HEALTHY"


def _boolean_series(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"U", "S", "O"}:
        return np.isin(np.char.lower(values.astype(str)), ["1", "true"])
    return values.astype(bool)


def _ground_truth(simulation, wheel: str) -> dict[str, object]:
    active_column = f"fault_active_{wheel}"
    if active_column not in simulation:
        return {"class": "HEALTHY"}
    active = _boolean_series(simulation[active_column].to_numpy())
    if not active.any():
        return {"class": "HEALTHY"}
    rows = simulation.loc[active]
    return {
        "class": "FAULTY",
        "fault_type": str(rows[f"fault_type_{wheel}"].iloc[0]),
        "fault_start_s": float(rows["time_s"].min()),
        "fault_end_s": float(rows["time_s"].max()),
        "fault_severity": float(rows[f"fault_severity_{wheel}"].max()),
        "dropped_edge_count": int(simulation[f"dropped_edges_{wheel}"].sum()),
    }


def plot_combined_diagnostic_summary(
    wheels: dict[str, dict[str, object]],
    simulation_id: int,
    output_path: str | Path,
):
    """Affiche explicitement les classes et sorties des deux pipelines."""
    probabilities = [
        float(wheels[wheel]["new_cnn_gru_fusion"]["fault_probability"])
        for wheel in WHEELS
    ]
    new_classes = [
        str(wheels[wheel]["new_cnn_gru_fusion"]["class"])
        for wheel in WHEELS
    ]
    colors = ["#d62728" if value == "FAULTY" else "#2ca02c" for value in new_classes]
    figure, (probability_axis, table_axis) = plt.subplots(
        2,
        1,
        figsize=(13, 7.5),
        gridspec_kw={"height_ratios": [2.1, 2.4]},
    )
    bars = probability_axis.bar(WHEELS, probabilities, color=colors, alpha=0.85)
    probability_axis.axhline(
        0.5, color="#d62728", linestyle="--", linewidth=1.4, label="Seuil 0.5"
    )
    probability_axis.set_ylim(0, 1.08)
    probability_axis.set_ylabel("Probabilite de defaut")
    probability_axis.set_title("Nouvelle fusion CNN/GRU — classification par roue")
    probability_axis.grid(True, axis="y", alpha=0.25)
    probability_axis.legend(loc="upper right")
    for bar, probability, class_name in zip(
        bars, probabilities, new_classes, strict=True
    ):
        probability_axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.03, probability + 0.035),
            f"{class_name}\n{probability:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    rows = []
    for wheel in WHEELS:
        result = wheels[wheel]
        old = result["old_spc"]
        new = result["new_cnn_gru_fusion"]
        truth = result["ground_truth_evaluation_only"]
        parameters = new["fault_parameters"]
        rows.append(
            [
                wheel,
                str(old["class"]),
                str(new["class"]),
                "-" if parameters is None else f"{parameters['fault_start_s']:.3f}",
                "-" if parameters is None else f"{parameters['fault_end_s']:.3f}",
                "-" if parameters is None else f"{parameters['fault_severity']:.3f}",
                str(truth["class"]),
            ]
        )
    table_axis.axis("off")
    table = table_axis.table(
        cellText=rows,
        colLabels=(
            "Roue",
            "Ancien MSP",
            "Fusion",
            "Debut (s)",
            "Fin (s)",
            "Severite",
            "Verite eval.",
        ),
        cellLoc="center",
        loc="center",
        colWidths=[0.08, 0.22, 0.13, 0.12, 0.12, 0.12, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.75)
    for column in range(7):
        table[(0, column)].set_facecolor("#d9eaf7")
        table[(0, column)].set_text_props(fontweight="bold")
    for row_index, class_name in enumerate(new_classes, start=1):
        table[(row_index, 2)].set_facecolor(
            "#f4cccc" if class_name == "FAULTY" else "#d9ead3"
        )
        table[(row_index, 2)].set_text_props(fontweight="bold")
    table_axis.set_title(
        "Ancien GRU t+1 + MSP versus nouvelle fusion et parametres",
        pad=8,
    )
    figure.suptitle(f"Diagnostic ABS combine — simulation {simulation_id}", fontsize=15)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    return figure


def run_combined_simulation(arguments: argparse.Namespace) -> dict[str, object]:
    default_dataset, default_manifest = latest_faulty_pair()
    dataset_csv = arguments.dataset_csv or default_dataset
    fault_manifest = arguments.fault_manifest or default_manifest
    simulation_id = arguments.simulation_id or select_faulty_demo_simulation_id(
        fault_manifest
    )
    columns = (*CSV_COLUMNS, *FAULT_COLUMNS)
    simulation = next(
        iter_selected_simulations(dataset_csv, [simulation_id], columns=columns)
    )

    limits_artifact = load_control_limits(arguments.limits)
    old_predictor = PhysicalUnitGRUPredictor.load(
        arguments.old_checkpoint,
        arguments.old_metadata,
        device=arguments.old_device,
    )
    old_results, dashboard = replay_simulation(
        simulation,
        old_predictor,
        limits_artifact["limits"],
        live=not arguments.no_live,
        playback_speed=arguments.playback_speed,
        render_every=arguments.render_every,
        visible_seconds=arguments.visible_seconds,
    )
    old_wheels = summarize_replay(old_results)

    fusion = CNNGRUFusionPredictor(
        arguments.fusion_experiments,
        arguments.fusion_cache,
        device=arguments.fusion_device,
        fault_threshold=arguments.fault_threshold,
    )
    wheels: dict[str, dict[str, object]] = {}
    for wheel, speed_column, valid_column in zip(
        WHEELS, SPEED_COLUMNS, VALID_COLUMNS, strict=True
    ):
        speed = simulation[speed_column].to_numpy(dtype=np.float32)
        valid = _boolean_series(simulation[valid_column].to_numpy())
        fusion_output = fusion.predict(speed, valid)
        wheels[wheel] = {
            "old_spc": {
                "class": old_spc_class(old_wheels[wheel]),
                **old_wheels[wheel],
            },
            "new_cnn_gru_fusion": {
                "class": (
                    "FAULTY" if fusion_output["fault_detected"] else "HEALTHY"
                ),
                **fusion_output,
            },
            "ground_truth_evaluation_only": _ground_truth(simulation, wheel),
        }

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"combined_faulty_simulation_{simulation_id}"
    old_csv = arguments.output_directory / f"{stem}_old_spc.csv"
    chart = arguments.output_directory / f"{stem}_old_spc.png"
    combined_chart = arguments.output_directory / f"{stem}_diagnostic.png"
    summary_path = arguments.output_directory / f"{stem}.summary.json"
    old_results.to_csv(old_csv, index=False)
    if dashboard is None:
        plot_static_replay(old_results, limits_artifact["limits"], chart)
    else:
        dashboard.save(chart)
    combined_figure = plot_combined_diagnostic_summary(
        wheels, simulation_id, combined_chart
    )
    if dashboard is not None:
        combined_figure.show()
        combined_figure.canvas.draw_idle()
        combined_figure.canvas.flush_events()
    else:
        plt.close(combined_figure)

    summary: dict[str, object] = {
        "simulation_id": simulation_id,
        "requested_phenomenon": str(simulation["requested_phenomenon"].iloc[0]),
        "observed_phenomenon": str(simulation["observed_phenomenon"].iloc[0]),
        "old_pipeline": "GRU t+1 residual + healthy SPC limits + 3-of-5 alarm",
        "new_pipeline": "CNN/GRU late fusion on one complete 500-sample sensor series",
        "fault_labels_used_by_diagnostics": False,
        "wheels": wheels,
        "artifacts": {
            "old_spc_csv": str(old_csv.resolve()),
            "old_spc_chart": str(chart.resolve()),
            "combined_diagnostic_chart": str(combined_chart.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSimulation {simulation_id} | {summary['observed_phenomenon']}")
    print("=" * 88)
    for wheel in WHEELS:
        result = wheels[wheel]
        old = result["old_spc"]
        new = result["new_cnn_gru_fusion"]
        truth = result["ground_truth_evaluation_only"]
        print(f"\n[{wheel}]")
        print(
            "  OLD SPC : "
            f"class={old['class']}, alarms={old['alarm_sample_count']}, "
            f"localized={old['localized_suspect_sample_count']}, "
            f"first_alarm_s={old['first_alarm_time_s']}"
        )
        print(
            "  NEW CNN+GRU : "
            f"class={new['class']}, probability={new['fault_probability']:.6f}"
        )
        print(
            "  NEW OUTPUT  : "
            + json.dumps(new["fault_parameters"], ensure_ascii=False)
        )
        print("  TRUTH (evaluation only): " + json.dumps(truth, ensure_ascii=False))
    print(f"\nCombined summary : {summary_path.resolve()}")
    print(f"Old SPC chart    : {chart.resolve()}")
    print(f"Combined chart   : {combined_chart.resolve()}")

    if (
        dashboard is not None
        and not arguments.close_on_finish
        and plt.fignum_exists(dashboard.figure.number)
    ):
        dashboard.hold()
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path)
    parser.add_argument("--fault-manifest", type=Path)
    parser.add_argument("--simulation-id", type=int)
    parser.add_argument("--old-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--old-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument(
        "--fusion-experiments", type=Path, default=DEFAULT_FUSION_EXPERIMENTS
    )
    parser.add_argument("--fusion-cache", type=Path, default=DEFAULT_FUSION_CACHE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--old-device", default="cpu")
    parser.add_argument("--fusion-device", default="cpu")
    parser.add_argument("--fault-threshold", type=float, default=0.5)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--visible-seconds", type=float, default=2.0)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--close-on-finish", action="store_true")
    return parser


def main() -> None:
    run_combined_simulation(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
