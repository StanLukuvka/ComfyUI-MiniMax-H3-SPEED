"""Committed workflow examples use the reset-only RES adapter."""

import json
from pathlib import Path


WORKFLOW_DIR = Path(__file__).parents[2] / "workflows"
WORKFLOW_NAMES = (
    "video_minimax_h3_SPEED_Sigma_Calculated.json",
    "video_minimax_h3_SPEED_Sigma_Manual.json",
    "video_minimax_h3_SPEED_SIGMA.json",
)
NODE_TYPES = {
    "MiniMaxH3SPEEDSampler",
    "MiniMaxH3SPEEDSamplerManual",
    "MiniMaxH3HarvestToConfig",
}
SAMPLER_INDEX = {
    "MiniMaxH3SPEEDSampler": -1,
    "MiniMaxH3SPEEDSamplerManual": -1,
    "MiniMaxH3HarvestToConfig": 0,
}


def _nodes(value):
    if isinstance(value, dict):
        if value.get("type") in NODE_TYPES:
            yield value
        for child in value.values():
            yield from _nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nodes(child)


def test_committed_examples_select_euler_for_each_public_node():
    for name in WORKFLOW_NAMES:
        workflow = json.loads((WORKFLOW_DIR / name).read_text())
        nodes = list(_nodes(workflow))
        assert nodes
        for node in nodes:
            assert node.get("widgets_values_named", {}).get("sampler_name") == "euler"
            assert node["widgets_values"][SAMPLER_INDEX[node["type"]]] == "euler"
            if node["type"] in {"MiniMaxH3SPEEDSampler", "MiniMaxH3SPEEDSamplerManual"}:
                assert "res_" + "history_mode" not in node["widgets_values_named"]
                assert "reset" not in node["widgets_values"]
