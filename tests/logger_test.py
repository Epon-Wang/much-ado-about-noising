"""Tests for checkpoint naming helpers in mip.logger."""

import fnmatch

import pytest
from omegaconf import OmegaConf

from mip.config import Config, LogConfig, NetworkConfig, OptimizationConfig, TaskConfig
from mip.logger import get_checkpoint_base_name

LEGACY = "lift_ph_state_flow_mlp_32_seed0"


def _config(**opt_kwargs):
    return Config(
        optimization=OptimizationConfig(loss_type="flow", **opt_kwargs),
        network=NetworkConfig(emb_dim=32),
        task=TaskConfig(),
        log=LogConfig(
            log_dir="./logs",
            wandb_mode="disabled",
            project="t",
            group="t",
            exp_name="t",
        ),
    )


def test_untagged_name_is_legacy_format():
    assert get_checkpoint_base_name(_config()) == LEGACY


def test_tag_is_appended():
    assert get_checkpoint_base_name(_config(ckpt_tag="flowenc")) == LEGACY + "_flowenc"


def test_dictconfig_without_ckpt_tag_key():
    cfg = OmegaConf.create(
        {
            "task": {"env_name": "lift", "env_type": "ph", "obs_type": "state"},
            "optimization": {"loss_type": "flow", "seed": 0},
            "network": {"network_type": "mlp", "emb_dim": 32},
        }
    )
    OmegaConf.set_struct(cfg, True)
    assert get_checkpoint_base_name(cfg) == LEGACY


@pytest.mark.parametrize("tag", ["a/b", "has space", "success9", "x_success"])
def test_invalid_tag_raises(tag):
    with pytest.raises(ValueError, match="ckpt_tag"):
        get_checkpoint_base_name(_config(ckpt_tag=tag))


def test_tagged_and_untagged_files_never_match_each_others_glob():
    tagged = get_checkpoint_base_name(_config(ckpt_tag="flowenc"))
    assert not fnmatch.fnmatch(f"{tagged}_success40.pt", f"{LEGACY}_success*.pt")
    assert not fnmatch.fnmatch(f"{LEGACY}_success40.pt", f"{tagged}_success*.pt")


@pytest.mark.parametrize("tag, suffix", [(2, "_2"), (0, "_0"), ("v1.5-a", "_v1.5-a")])
def test_non_string_tag_is_coerced(tag, suffix):
    cfg = OmegaConf.create(
        {
            "task": {"env_name": "lift", "env_type": "ph", "obs_type": "state"},
            "optimization": {"loss_type": "flow", "seed": 0, "ckpt_tag": tag},
            "network": {"network_type": "mlp", "emb_dim": 32},
        }
    )
    assert get_checkpoint_base_name(cfg) == LEGACY + suffix


def test_none_tag_means_untagged():
    cfg = OmegaConf.create(
        {
            "task": {"env_name": "lift", "env_type": "ph", "obs_type": "state"},
            "optimization": {"loss_type": "flow", "seed": 0, "ckpt_tag": None},
            "network": {"network_type": "mlp", "emb_dim": 32},
        }
    )
    assert get_checkpoint_base_name(cfg) == LEGACY
