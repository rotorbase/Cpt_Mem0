"""Command line interface.

``cpt-mem0 plan`` is the command that matters: it computes the
configuration for the local hardware and prints it without starting
anything. Every other command is a thin wrapper around the same data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalogue import (
    Catalogue,
    catalogue_from_dict,
    default_catalogue_path,
    load_catalogue,
)
from .hardware import detect_hardware
from .planner import Hardware, PlannerConfig, PlanResult, plan_for_hardware

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NO_VIABLE_CONFIG = 3


def _load_catalogue(path: str | None) -> Catalogue:
    if path:
        return load_catalogue(path)
    default = default_catalogue_path()
    if default.is_file():
        return load_catalogue(default)
    return Catalogue(version=1, updated="", entries=[])


def _format_plan(result: PlanResult, hardware: Hardware) -> str:
    lines: list[str] = []
    lines.append(f"GPU     {hardware.name or 'unknown'}".ljust(42)
                 + f"{hardware.total_vram_mib} MiB total")
    if result.viable and result.candidate:
        c = result.candidate
        lines.append(f"LLM     {c.model_id} {c.quantisation}".ljust(42)
                     + f"{c.file_size_mib:.0f} MiB")
        lines.append(f"        weights {c.weights_vram_mib:.0f} MiB  "
                     f"KV {c.kv_vram_mib:.0f} MiB "
                     f"({c.kv_bytes_per_token} B/tok, {c.cache_type})")
        lines.append(f"        n_ctx {c.n_ctx}  ({c.n_ctx_slot} per slot, "
                     f"{c.parallel} slots)")
        # Print the flags that are actually used. A flag shown here that
        # does not exist in the target build would make the server abort
        # during argument parsing, before it loads anything.
        lines.append(f"        --fit off  --flash-attn on  "
                     f"--reasoning-format none  --kv-unified")
        if c.recurrent_state_mib > 0:
            lines.append(f"        recurrent state     {c.recurrent_state_mib:.0f} MiB "
                         f"(hybrid model, fixed)")
        lines.append(f"        ranking basis: {c.model_rank_basis}")
        lines.append(f"EMBED   {c.embedder_backend}".ljust(42)
                     + f"{'0 MiB (in-process ONNX)' if c.embedder_backend == 'fastembed' else 'on GPU'}")
        if result.budget:
            b = result.budget
            lines.append(f"TOTAL   {b.used_mib:.0f} MiB used  +  "
                         f"{b.reserve_mib} reserve  +  {b.headroom_mib} headroom"
                         f"  =  {b.used_mib + b.reserve_mib + b.headroom_mib:.0f}"
                         f" / {b.total_mib:.0f}")
    else:
        diag = result.diagnostics
        lines.append(f"NO VIABLE CONFIGURATION: {diag.get('reason', 'unknown')}")
        if diag.get("shortfall_mib"):
            lines.append(f"        short by {diag['shortfall_mib']} MiB")
        for option in diag.get("options", []):
            lines.append(f"        - {option}")
    if result.diagnostics.get("degraded"):
        lines.append("")
        lines.append(f"WARNING: {result.diagnostics.get('warning', '')}")
    return "\n".join(lines)


def _plan_json(result: PlanResult, hardware: Hardware) -> dict:
    data: dict = {
        "gpu": {
            "name": hardware.name,
            "vendor": hardware.vendor,
            "total_mib": hardware.total_vram_mib,
        },
        "viable": result.viable,
    }
    if result.viable and result.candidate:
        c = result.candidate
        data["llm"] = {
            "model": c.model_id,
            "quantisation": c.quantisation,
            "file": c.file_name,
            "size_mib": round(c.file_size_mib),
            "weights_vram_mib": round(c.weights_vram_mib),
            "kv_vram_mib": round(c.kv_vram_mib),
            "kv_bytes_per_token": c.kv_bytes_per_token,
            "cache_type": c.cache_type,
            "n_ctx": c.n_ctx,
            "n_ctx_slot": c.n_ctx_slot,
            "parallel": c.parallel,
            "rank_basis": c.model_rank_basis,
            "flags": c.as_flags(),
        }
        data["embedder"] = {"backend": c.embedder_backend}
    if result.budget:
        b = result.budget
        data["budget"] = {
            "total_mib": round(b.total_mib),
            "used_mib": round(b.used_mib),
            "reserved_mib": b.reserve_mib,
            "headroom_mib": b.headroom_mib,
            "free_mib": round(b.free_mib),
        }
    if result.diagnostics:
        data["diagnostics"] = result.diagnostics
    return data


def cmd_plan(args: argparse.Namespace) -> int:
    hardware = None
    if args.vram:
        hardware = Hardware(total_vram_mib=int(args.vram), name=args.gpu_name or "")
    else:
        hardware = detect_hardware()
    if hardware is None:
        print("No GPU detected. Pass --vram to plan for a specific card.",
              file=sys.stderr)
        return EXIT_ERROR

    catalogue = _load_catalogue(args.catalogue)
    config = PlannerConfig(
        reserve_mib=args.reserve,
        headroom_mib=args.headroom,
        allow_degraded=args.allow_degraded,
    )
    result = plan_for_hardware(hardware, catalogue.sorted_entries(), config)

    if args.json:
        print(json.dumps(_plan_json(result, hardware), indent=2))
    else:
        print(_format_plan(result, hardware))
    return EXIT_OK if result.viable else EXIT_NO_VIABLE_CONFIG


def cmd_models(args: argparse.Namespace) -> int:
    catalogue = _load_catalogue(args.catalogue)
    if args.json:
        print(json.dumps(catalogue.to_dict(), indent=2))
        return EXIT_OK
    print(f"{'model':26} {'params':>8} {'quant':>10} {'MiB':>9}  basis")
    for entry in catalogue.sorted_entries():
        for quant in entry.sorted_quantisations():
            print(f"{entry.model_id:26} {entry.params/1e9:7.1f}B "
                  f"{quant.name:>10} {quant.file_size_mib:9.0f}  "
                  f"{quant.kv_source}")
    return EXIT_OK


def cmd_validate_catalogue(args: argparse.Namespace) -> int:
    catalogue = _load_catalogue(args.catalogue)
    problems: list[str] = []
    for entry in catalogue.entries:
        if not entry.quantisations:
            problems.append(f"{entry.model_id}: no quantisations")
        derived = entry.kv_bytes_from_geometry()
        for quant in entry.quantisations:
            if quant.kv_source == "calibrate" and quant.kv_bytes_per_token:
                problems.append(
                    f"{entry.model_id}/{quant.name}: marked calibrate but has a constant")
            if quant.kv_source in ("measured", "formula") and not quant.kv_bytes_per_token:
                problems.append(
                    f"{entry.model_id}/{quant.name}: {quant.kv_source} but no constant")
            if derived is not None and quant.kv_bytes_per_token != derived:
                problems.append(
                    f"{entry.model_id}/{quant.name}: constant "
                    f"{quant.kv_bytes_per_token} != geometry {derived}")
    for problem in problems:
        print(f"  {problem}")
    if problems:
        return EXIT_ERROR
    print(f"catalogue OK: {len(catalogue.entries)} models, "
          f"{sum(len(e.quantisations) for e in catalogue.entries)} quantisations")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpt-mem0",
        description="Self-contained local mem0 memory service for consumer GPUs.",
    )
    parser.add_argument("--catalogue", help="path to catalogue.json")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="compute the configuration, start nothing")
    plan.add_argument("--vram", type=int, help="plan for a card with this much VRAM (MiB)")
    plan.add_argument("--gpu-name", help="name to report in the output")
    plan.add_argument("--reserve", type=int, help="VRAM to reserve (MiB)")
    plan.add_argument("--headroom", type=int, help="VRAM headroom (MiB)")
    plan.add_argument("--allow-degraded", action="store_true",
                      help="permit configurations below the context floor")
    plan.add_argument("--json", action="store_true", help="machine-readable output")
    plan.set_defaults(func=cmd_plan)

    models = sub.add_parser("models", help="show the model catalogue")
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=cmd_models)

    validate = sub.add_parser("validate-catalogue",
                              help="check the catalogue for internal consistency")
    validate.set_defaults(func=cmd_validate_catalogue)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
