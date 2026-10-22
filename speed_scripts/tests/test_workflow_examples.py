"""The shipped workflow examples wire the public nodes as documented."""

import json
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).parents[2] / "workflows").glob("*.json"))
OUTPUTS = {
    "MiniMaxH3SPEEDSampler": ("output", "denoised_output"),
    "MiniMaxH3SPEEDSamplerManual": ("output", "denoised_output"),
    "MiniMaxH3HarvestToConfig": ("calibration", "diagnostic_latent"),
}


def _graphs(value):
    """Yield every graph (top level or subgraph) that has its own node list."""
    if isinstance(value, dict):
        if isinstance(value.get("nodes"), list):
            yield value
        values = value.values()
    else:
        values = value if isinstance(value, list) else ()
    for child in values:
        yield from _graphs(child)


def test_all_five_examples_are_shipped():
    assert {path.name for path in WORKFLOWS} == {
        "video_minimax_h3_SPEED_Sigma_Calculated.json",
        "video_minimax_h3_SPEED_Sigma_Manual.json",
        "video_minimax_h3_SPEED_SIGMA.json",
        "image_moss_SPEED.json",
        "image_kroma_SPEED.json",
    }


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_example_feeds_each_public_node_from_ksampler_select_euler(path):
    found = 0
    for graph in _graphs(json.loads(path.read_text())):
        nodes = {node["id"]: node for node in graph["nodes"]}
        # Links are dict records in subgraphs and [id, origin, ...] arrays at the top level.
        origins = dict(
            (link["id"], link["origin_id"]) if isinstance(link, dict) else link[:2]
            for link in graph.get("links", [])
        )
        for node in graph["nodes"]:
            if node.get("type") not in OUTPUTS:
                continue
            found += 1
            # SamplerCustomAdvanced order: noise, guider, sampler, ...
            names = [i["name"] for i in node["inputs"]]
            assert names.count("sampler") == 1 and names.index("sampler") == 2
            sampler = node["inputs"][2]
            assert sampler["type"] == "SAMPLER" and sampler["link"] is not None
            source = nodes[origins[sampler["link"]]]
            assert (source["type"], source["widgets_values"]) == ("KSamplerSelect", ["euler"])
            assert tuple(output["name"] for output in node["outputs"]) == OUTPUTS[node["type"]]
    assert found
