"""Simulate ComfyUI's node-loading to prove the rename fixed the collision.

ComfyUI's core has its own `nodes` module (ComfyUI/nodes.py). This script
replicates a fake copy BEFORE importing our package, which is exactly the
condition that broke helper-node registration.
"""
import sys
import types
import importlib.util

# Fake ComfyUI built-in nodes module (this is what was silently winning).
fake_comfy_nodes = types.ModuleType("nodes")
fake_comfy_nodes.NODE_CLASS_MAPPINGS = {"FakeComfyBuiltin": object}
fake_comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS = {"FakeComfyBuiltin": "Fake Builtin"}
sys.modules["nodes"] = fake_comfy_nodes

# Fake comfy package bits used by sampler_node.py import chain.
comfy = types.ModuleType("comfy")
samplers = types.ModuleType("comfy.samplers")
utils = types.ModuleType("comfy.utils")
nested_tensor = types.ModuleType("comfy.nested_tensor")
samplers.sampler_object = lambda name: ("sampler", name)
utils.PROGRESS_BAR_ENABLED = True
nested_tensor.NestedTensor = type("NestedTensor", (), {})
sys.modules["comfy"] = comfy
sys.modules["comfy.samplers"] = samplers
sys.modules["comfy.utils"] = utils
sys.modules["comfy.nested_tensor"] = nested_tensor

# Load the repo root __init__.py the way ComfyUI does it.
repo_root = "/agent/projects/minimax-quickfile/ComfyUI-MiniMaxH3-SPEED-Sampler"
spec = importlib.util.spec_from_file_location(
    "ComfyUI-MiniMaxH3-SPEED-Sampler", repo_root + "/__init__.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

expected = {
    "MiniMaxH3SPEEDSampler",
    "MiniMaxH3SigmaHarvest",
    "MiniMaxH3HarvestToConfig",
    "MiniMaxH3SPEEDSchedule",
    "MiniMaxH3Inspect",
    "MiniMaxH3PowerSpectrum",
    "MiniMaxH3DCTLowpass",
    "MiniMaxH3TransitionMath",
    "MiniMaxH3SpectralExpand",
    "MiniMaxH3XFidelityProbe",
    "MiniMaxH3AVReentryOracle",
}
got = set(mod.NODE_CLASS_MAPPINGS.keys())
got -= {"FakeComfyBuiltin"}  # tolerate ComfyUI merging its own builtins

missing = expected - got
extra = got - expected
print(f"registered: {sorted(got)}")
print(f"missing: {sorted(missing)}" if missing else "missing: none ✓")
print(f"extra: {sorted(extra)}" if extra else "extra: none ✓")
assert not missing, f"FAIL — nodes missing: {missing}"
assert not extra, f"FAIL — unexpected nodes: {extra}"
print("PASS — all 11 nodes register despite ComfyUI's 'nodes' module collision")
