"""HarvestToConfig node — parses harvest JSON, emits human-readable report.

Parses the JSON payload from MiniMaxH3SigmaHarvest and prints a calibration
report: fitted A, beta, r², fit health, and recommended delta-optimal
transition steps per preset. Text-only — does not emit a typed config object,
so it cannot be wired into a downstream sampler; use the report to copy the
calibrated power_A / power_beta and transition_steps into the sampler's widgets,
or wire the harvest JSON into a Schedule node.
"""

from __future__ import annotations

import json


class MiniMaxH3HarvestToConfig:
    DESCRIPTION = (
        "Parses the JSON payload from MiniMaxH3SigmaHarvest and prints a "
        "calibration report: fitted A, beta, r², fit health, and recommended "
        "delta-optimal transition steps per preset."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "parse"
    CATEGORY = "sampling/minimax_h3_speed/diagnostics"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "harvest_json": ("STRING", {"default": "{}"}),
            },
        }

    def parse(self, harvest_json):
        try:
            data = json.loads(harvest_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Invalid harvest JSON: {exc}") from exc

        for key in ("overall_fit_A", "overall_fit_beta"):
            if key not in data:
                raise ValueError(f"Harvest JSON missing key: {key}")
        if not (0 < data["overall_fit_A"] < 1e9):
            raise ValueError(f"Invalid A: {data['overall_fit_A']}")
        if not (-5 <= data["overall_fit_beta"] < 10):
            raise ValueError(f"Invalid beta: {data['overall_fit_beta']}")

        A = data["overall_fit_A"]
        beta = data["overall_fit_beta"]
        r2 = data.get("overall_fit_r2", None)
        n_sigmas = len(data.get("sigma_levels", []))
        r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else "N/A"
        mode = data.get("fit_mode", "?")
        health = data.get("fit_health", "?")

        lines = [f"Calibrated: A={A:.3f}  beta={beta:.3f}  r²={r2_str}  "
                 f"({n_sigmas} sigma levels, fit_mode={mode})"]
        if health in ("suspect", "weak", "invalid"):
            lines.append(
                f"WARNING: fit is {health.upper()} — beta={'%.3f' % beta} with "
                f"r²={'%.3f' % r2 if isinstance(r2, float) else 'N/A'}. "
                "Power is not cleanly decaying. Trust the transition_steps below "
                "with caution, or change fit_mode / capture_every and re-run."
            )

        rec = data.get("recommended_config")
        if isinstance(rec, dict) and rec:
            lines.append("Recommended delta-optimal transition_steps (paste into sampler):")
            for name, preset in rec.items():
                lines.append(
                    f"  {name}: scales={preset['scales']}  "
                    f"transition_steps={preset['transition_steps']}"
                )
        else:
            lines.append("(no recommended_config in harvest; re-run the harvest node "
                         "to include per-preset transition_steps)")

        # Per-sigma diagnostic table: exposes how the fit evolves with sigma.
        sigma_fits = data.get("sigma_fits")
        if isinstance(sigma_fits, list) and sigma_fits:
            rows = []
            for sf in sigma_fits[:12]:  # cap to keep the report readable
                rows.append(f"    sigma={sf['sigma']:.3f}  "
                            f"beta={sf['beta']:+.3f}  r²={sf['r_squared']:.3f}")
            if len(sigma_fits) > 12:
                rows.append(f"    … {len(sigma_fits) - 12} more levels")
            lines.append("Per-sigma velocity fits (diagnostic):")
            lines.append("\n".join(rows))

        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {"MiniMaxH3HarvestToConfig": MiniMaxH3HarvestToConfig}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3HarvestToConfig": "MiniMax H3 SPEED — Harvest → Config"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MiniMaxH3HarvestToConfig"]
