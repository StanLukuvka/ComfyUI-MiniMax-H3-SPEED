"""The shipped workflow examples wire the public nodes as documented."""

import importlib
import json
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).parents[2] / "workflows").glob("*.json"))
IMAGE_WORKFLOWS = [path for path in WORKFLOWS if path.name.startswith("image_")]
VIDEO_WORKFLOW = next(path for path in WORKFLOWS if path.name == "video_h3_SPEED.json")
NODE_MODULES = {
    "SPEEDSampler": "sampler_node",
    "SPEEDSamplerManual": "sampler_node_manual",
    "SPEEDSigmaHarvest": "sampler_sigma_harvest_node",
    "SPEEDKSampler": "ksampler_node",
    "SPEEDKSamplerManual": "ksampler_node_manual",
    "SPEEDKSamplerHarvest": "ksampler_sigma_harvest_node",
}
SPEED_TYPES = set(NODE_MODULES)
KSAMPLER_TYPES = {"SPEEDKSampler", "SPEEDKSamplerManual", "SPEEDKSamplerHarvest"}
ADVANCED_TYPES = {"SPEEDSampler", "SPEEDSamplerManual", "SPEEDSigmaHarvest"}
ADVANCED_HELPERS = {"RandomNoise", "CFGGuider", "KSamplerSelect", "BasicScheduler"}
OUTPUTS = {
    "SPEEDSampler": ("output", "denoised_output"),
    "SPEEDSamplerManual": ("output", "denoised_output"),
    "SPEEDSigmaHarvest": ("calibration", "diagnostic_latent"),
    "SPEEDKSampler": ("output", "denoised_output"),
    "SPEEDKSamplerManual": ("output", "denoised_output"),
    "SPEEDKSamplerHarvest": ("calibration", "diagnostic_latent"),
}
DEPRECATED_IDS = {"MiniMaxH3SPEEDSampler", "MiniMaxH3SPEEDSamplerManual", "MiniMaxH3HarvestToConfig"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}
CONTROL = "control_after_generate"
WILDCARD = "*"


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


def _node_class(type_name):
    return getattr(importlib.import_module(NODE_MODULES[type_name]), type_name)


def _entries(cls):
    """INPUT_TYPES entries as (name, type, config), required then optional, in key order."""
    spec = cls.INPUT_TYPES()
    entries = []
    for section in ("required", "optional"):
        for name, entry in (spec.get(section) or {}).items():
            config = entry[1] if len(entry) > 1 else {}
            entries.append((name, entry[0], config))
    return entries


def _is_widget(type_):
    return isinstance(type_, list) or type_ in WIDGET_TYPES


def _expected_widgets(cls):
    """Widget order: widget-typed inputs by INPUT_TYPES key order; seed-like INTs gain control_after_generate."""
    names = []
    for name, type_, config in _entries(cls):
        if _is_widget(type_):
            names.append(name)
            if type_ == "INT" and config.get(CONTROL):
                names.append(CONTROL)
    return names


def _expected_link_inputs(cls):
    return [name for name, type_, _ in _entries(cls) if not _is_widget(type_)]


def _link_record(link):
    """Normalize a links-list entry to (id, origin_id, origin_slot, target_id, target_slot)."""
    if isinstance(link, dict):
        return (link["id"], link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"])
    return tuple(link[:5])


def test_all_three_examples_are_shipped():
    assert {path.name for path in WORKFLOWS} == {
        "video_h3_SPEED.json",
        "image_moss_SPEED.json",
        "image_krea2_SPEED.json",
    }


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_speed_node_widgets_match_input_types(path):
    found = 0
    for graph in _graphs(json.loads(path.read_text())):
        for node in graph["nodes"]:
            # Examples use the current IDs, never the deprecated ones.
            assert node.get("type") not in DEPRECATED_IDS, node["type"]
            if node.get("type") not in SPEED_TYPES:
                continue
            found += 1
            expected = _expected_widgets(_node_class(node["type"]))
            widgets = node["widgets_values"]
            assert len(widgets) == len(expected), (node["type"], expected, widgets)
            named = node.get("widgets_values_named")
            if named is not None:
                assert list(named.keys()) == expected, (node["type"], expected, list(named))
            assert tuple(output["name"] for output in node["outputs"]) == OUTPUTS[node["type"]]
    assert found


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_every_referenced_link_resolves_and_endpoints_agree(path):
    for graph in _graphs(json.loads(path.read_text())):
        nodes = {node["id"]: node for node in graph["nodes"]}
        # Links are dict records in subgraphs and [id, origin, ...] arrays at the top level.
        links = {_link_record(link)[0]: _link_record(link)[1:] for link in graph.get("links", [])}
        for link_id, (origin, origin_slot, target, target_slot) in links.items():
            assert origin in nodes and target in nodes, link_id
            assert origin_slot < len(nodes[origin].get("outputs") or []), link_id
            assert target_slot < len(nodes[target].get("inputs") or []), link_id
        for node in graph["nodes"]:
            for slot, inp in enumerate(node.get("inputs") or []):
                if inp.get("link") is None:
                    continue
                assert inp["link"] in links, (node["id"], inp["name"], inp["link"])
                origin, origin_slot, target, target_slot = links[inp["link"]]
                assert (target, target_slot) == (node["id"], slot), (inp["link"], node["id"], slot)
                output = nodes[origin]["outputs"][origin_slot]
                # Inputs may be wildcards or type unions (e.g. "FLOAT,INT,BOOLEAN").
                assert output["type"] == WILDCARD or inp["type"] == WILDCARD \
                    or output["type"] in inp["type"].split(","), (inp["link"], output["type"], inp["type"])
            for slot, output in enumerate(node.get("outputs") or []):
                for link in output.get("links") or []:
                    assert link in links, (node["id"], output["name"], link)
                    origin, origin_slot, target, target_slot = links[link]
                    assert (origin, origin_slot) == (node["id"], slot), (link, node["id"], slot)
                    inp = nodes[target]["inputs"][target_slot]
                    assert inp["link"] == link, (link, node["id"], slot)


@pytest.mark.parametrize("path", IMAGE_WORKFLOWS, ids=lambda p: p.stem)
def test_image_workflows_use_the_ksampler_family(path):
    types = [node["type"] for graph in _graphs(json.loads(path.read_text())) for node in graph["nodes"]]
    assert not set(types) & (ADVANCED_TYPES | ADVANCED_HELPERS | DEPRECATED_IDS)
    assert set(types) >= KSAMPLER_TYPES
    automatic = [node for graph in _graphs(json.loads(path.read_text()))
                 for node in graph["nodes"] if node["type"] == "SPEEDKSampler"]
    assert any(node.get("mode", 0) == 0 for node in automatic)


def test_video_workflow_keeps_the_advanced_family():
    types = [node["type"] for graph in _graphs(json.loads(VIDEO_WORKFLOW.read_text())) for node in graph["nodes"]]
    assert {"SPEEDSampler", "SPEEDSigmaHarvest"} <= set(types)
    assert not set(types) & KSAMPLER_TYPES


@pytest.mark.parametrize("path", IMAGE_WORKFLOWS, ids=lambda p: p.stem)
def test_image_speed_nodes_wire_model_and_conditions_directly(path):
    for graph in _graphs(json.loads(path.read_text())):
        for node in graph["nodes"]:
            if node["type"] not in KSAMPLER_TYPES:
                continue
            names = [inp["name"] for inp in node["inputs"]]
            assert names == _expected_link_inputs(_node_class(node["type"])), (node["type"], names)
            assert set(names).isdisjoint({"noise", "guider", "sampler", "sigmas"})
            assert all(inp["link"] is not None for inp in node["inputs"])


def test_video_advanced_nodes_are_fed_from_ksampler_select_euler():
    found = 0
    for graph in _graphs(json.loads(VIDEO_WORKFLOW.read_text())):
        nodes = {node["id"]: node for node in graph["nodes"]}
        origins = {_link_record(link)[0]: _link_record(link)[1] for link in graph.get("links", [])}
        for node in graph["nodes"]:
            if node.get("type") not in ADVANCED_TYPES:
                continue
            found += 1
            # SamplerCustomAdvanced order: noise, guider, sampler, ...
            names = [inp["name"] for inp in node["inputs"]]
            assert names.count("sampler") == 1 and names.index("sampler") == 2
            sampler = node["inputs"][2]
            assert sampler["type"] == "SAMPLER" and sampler["link"] is not None
            source = nodes[origins[sampler["link"]]]
            assert (source["type"], source["widgets_values"]) == ("KSamplerSelect", ["euler"])
    assert found
