"""The package must register every shipped ComfyUI node."""

import importlib.util
from pathlib import Path


def test_package_registers_all_shipped_nodes():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("speed_pack_test", root / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert set(module.NODE_CLASS_MAPPINGS) == {
        "MiniMaxH3SPEEDSampler",
        "MiniMaxH3SPEEDSamplerManual",
        "MiniMaxH3HarvestToConfig",
    }
    assert set(module.NODE_DISPLAY_NAME_MAPPINGS) == set(module.NODE_CLASS_MAPPINGS)
