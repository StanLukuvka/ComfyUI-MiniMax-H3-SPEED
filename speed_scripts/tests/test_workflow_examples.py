"""Committed workflow examples match the public node contracts."""

import json
from pathlib import Path


WORKFLOW_DIR = Path(__file__).parents[2] / "workflows"
WORKFLOW_NAMES = (
    "video_minimax_h3_SPEED_Sigma_Calculated.json",
    "video_minimax_h3_SPEED_Sigma_Manual.json",
    "video_minimax_h3_SPEED_SIGMA.json",
    "image_moss_SPEED.json",
    "image_kroma_SPEED.json",
)
NODE_TYPES = {
    "MiniMaxH3SPEEDSampler",
    "MiniMaxH3SPEEDSamplerManual",
    "MiniMaxH3HarvestToConfig",
}
EXPECTED_OUTPUTS = {
    "MiniMaxH3SPEEDSampler": ("output", "denoised_output"),
    "MiniMaxH3SPEEDSamplerManual": ("output", "denoised_output"),
    "MiniMaxH3HarvestToConfig": ("calibration", "diagnostic_latent"),
}


def _graphs(value):
    """Yield every graph (top level or subgraph) that has its own node list."""
    if isinstance(value, dict):
        if isinstance(value.get("nodes"), list):
            yield value
        for child in value.values():
            yield from _graphs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _graphs(child)


def _link_origin(link):
    """Return ``(link_id, origin_node_id)`` for dict or array link records."""
    if isinstance(link, dict):
        return link["id"], link["origin_id"]
    return link[0], link[1]


def _public_nodes_with_graph(workflow):
    for graph in _graphs(workflow):
        for node in graph["nodes"]:
            if node.get("type") in NODE_TYPES:
                yield graph, node


def test_committed_examples_feed_each_public_node_from_ksampler_select_euler():
    for name in WORKFLOW_NAMES:
        workflow = json.loads((WORKFLOW_DIR / name).read_text())
        found = list(_public_nodes_with_graph(workflow))
        assert found, name
        for graph, node in found:
            nodes_by_id = {other["id"]: other for other in graph["nodes"]}
            origins = dict(_link_origin(link) for link in graph.get("links", []))

            input_names = [i["name"] for i in node["inputs"]]
            assert input_names.count("sampler") == 1, (name, node["id"])
            # SamplerCustomAdvanced order: noise, guider, sampler, ...
            assert input_names.index("sampler") == 2, (name, node["id"])
            sampler_input = node["inputs"][2]
            assert sampler_input["type"] == "SAMPLER"
            assert sampler_input["link"] is not None
            source = nodes_by_id[origins[sampler_input["link"]]]
            assert source["type"] == "KSamplerSelect"
            assert source["widgets_values"] == ["euler"]

            assert tuple(output["name"] for output in node["outputs"]) == EXPECTED_OUTPUTS[node["type"]]
