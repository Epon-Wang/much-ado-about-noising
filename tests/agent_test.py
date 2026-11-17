"""Tests for TrainingAgent.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import pytest
import torch

from mip.agent import TrainingAgent
from mip.config import Config, LogConfig, NetworkConfig, OptimizationConfig, TaskConfig


class TestTrainingAgent:
    """Test suite for the TrainingAgent class."""

    def test_agent_different_loss_types(self):
        """Test agent with different loss types."""
        act_dim = 2
        Ta = 4
        obs_dim = 3
        To = 1
        bs = 8

        network_config = NetworkConfig(emb_dim=32, num_layers=2)
        task_config = TaskConfig(
            act_dim=act_dim, obs_dim=obs_dim, act_steps=Ta, obs_steps=To, horizon=Ta
        )
        log_config = LogConfig(
            log_dir="./logs",
            wandb_mode="disabled",
            project="test",
            group="test",
            exp_name="test",
        )

        for loss_type in ["flow", "regression", "tsd", "mip"]:
            optimization_config = OptimizationConfig(
                loss_type=loss_type, num_steps=5, device="cpu"
            )
            config = Config(
                optimization=optimization_config,
                network=network_config,
                task=task_config,
                log=log_config,
            )
            agent = TrainingAgent(config)

            act = torch.randn(bs, Ta, act_dim)
            obs = torch.randn(bs, To, obs_dim)
            delta_t = torch.rand(bs)
            act_0 = torch.randn(bs, Ta, act_dim)

            # Test update
            info = agent.update(act, obs, delta_t)
            assert isinstance(info, dict)
            assert "loss" in info

            # Test sample
            sampled_act = agent.sample(act_0, obs)
            assert sampled_act.shape == act_0.shape


def _make_config(loss_type="flow", obs_dim=3, **opt_kwargs):
    return Config(
        optimization=OptimizationConfig(
            loss_type=loss_type,
            num_steps=3,
            device="cpu",
            use_compile=False,
            **opt_kwargs,
        ),
        network=NetworkConfig(emb_dim=32, num_layers=2),
        task=TaskConfig(
            act_dim=2, obs_dim=obs_dim, act_steps=4, obs_steps=1, horizon=4
        ),
        log=LogConfig(
            log_dir="./logs",
            wandb_mode="disabled",
            project="test",
            group="test",
            exp_name="test",
        ),
    )


def _batch(obs_dim=3, bs=8):
    return torch.randn(bs, 4, 2), torch.randn(bs, 1, obs_dim), torch.rand(bs)


def _make_donor(tmp_path, obs_dim=3):
    """A trained-a-little agent, so encoder, encoder_ema and the initial weights all differ."""
    donor = TrainingAgent(_make_config("flow", obs_dim=obs_dim))
    for _ in range(3):
        donor.update(*_batch(obs_dim))
    path = tmp_path / "donor.pt"
    donor.save(path)
    return path, torch.load(path, weights_only=False)


def _assert_state_equal(module, state_dict):
    actual = module.state_dict()
    assert actual.keys() == state_dict.keys()
    for k in state_dict:
        assert torch.equal(actual[k], state_dict[k]), k


class TestFrozenEncoder:
    def test_unset_keeps_default_behavior(self):
        opt = OptimizationConfig()
        assert opt.frozen_encoder_path is None
        assert opt.ckpt_tag == ""
        agent = TrainingAgent(_make_config("regression"))
        assert agent.frozen_encoder is False
        n_params = len(list(agent.encoder.parameters())) + len(
            list(agent.flow_map.parameters())
        )
        assert sum(len(g["params"]) for g in agent.optimizer.param_groups) == n_params
        before = {k: v.clone() for k, v in agent.encoder.state_dict().items()}
        agent.update(*_batch())
        assert any(
            not torch.equal(before[k], v) for k, v in agent.encoder.state_dict().items()
        )

    def test_encoder_never_changes_while_head_trains(self, tmp_path):
        path, donor = _make_donor(tmp_path)
        assert any(
            not torch.equal(donor["encoder"][k], donor["encoder_ema"][k])
            for k in donor["encoder"]
        )
        agent = TrainingAgent(
            _make_config("regression", frozen_encoder_path=str(path), ckpt_tag="t")
        )
        assert agent.frozen_encoder is True
        _assert_state_equal(agent.encoder, donor["encoder_ema"])
        _assert_state_equal(agent.encoder_ema, donor["encoder_ema"])
        assert all(not p.requires_grad for p in agent.encoder.parameters())
        opt_ids = {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
        assert opt_ids == {id(p) for p in agent.flow_map.parameters()}
        # the head is a fresh init, not the donor's
        assert any(
            not torch.equal(v, donor["flow_map"][k])
            for k, v in agent.flow_map.state_dict().items()
        )

        head_before = {k: v.clone() for k, v in agent.flow_map.state_dict().items()}
        head_ema_before = {
            k: v.clone() for k, v in agent.flow_map_ema.state_dict().items()
        }
        for _ in range(5):
            agent.train()
            info = agent.update(*_batch())
            assert torch.isfinite(info["loss"])
            agent.eval()
            agent.sample(torch.randn(8, 4, 2), torch.randn(8, 1, 3))
        agent.train()

        _assert_state_equal(agent.encoder, donor["encoder_ema"])
        _assert_state_equal(agent.encoder_ema, donor["encoder_ema"])
        assert agent.encoder.training is False
        assert agent.encoder_ema.training is False
        assert agent.flow_map.training is True
        assert any(
            not torch.equal(head_before[k], v)
            for k, v in agent.flow_map.state_dict().items()
        )
        assert any(
            not torch.equal(head_ema_before[k], v)
            for k, v in agent.flow_map_ema.state_dict().items()
        )

    def test_requires_ckpt_tag(self, tmp_path):
        path, _ = _make_donor(tmp_path)
        with pytest.raises(ValueError, match="ckpt_tag"):
            TrainingAgent(_make_config("regression", frozen_encoder_path=str(path)))

    def test_save_load_roundtrip(self, tmp_path):
        path, donor = _make_donor(tmp_path)
        cfg = _make_config("flow", frozen_encoder_path=str(path), ckpt_tag="t")
        agent = TrainingAgent(cfg)
        for _ in range(2):
            agent.update(*_batch())
        ckpt = tmp_path / "frozen_run.pt"
        agent.save(ckpt, training_state={"n_gradient_step": 1})
        saved = torch.load(ckpt, weights_only=False)
        for k in donor["encoder_ema"]:
            assert torch.equal(saved["encoder"][k], donor["encoder_ema"][k])
            assert torch.equal(saved["encoder_ema"][k], donor["encoder_ema"][k])

        resumed = TrainingAgent(cfg)
        state = resumed.load(str(ckpt), load_optimizer=True)
        assert state == {"n_gradient_step": 1}
        _assert_state_equal(resumed.flow_map, saved["flow_map"])
        resumed.update(*_batch())
        _assert_state_equal(resumed.encoder, donor["encoder_ema"])
        _assert_state_equal(resumed.encoder_ema, donor["encoder_ema"])
        assert all(not p.requires_grad for p in resumed.encoder.parameters())

    def test_donor_shape_mismatch_raises(self, tmp_path):
        path, _ = _make_donor(tmp_path, obs_dim=3)
        with pytest.raises(RuntimeError):
            TrainingAgent(
                _make_config(
                    "regression", obs_dim=5, frozen_encoder_path=str(path), ckpt_tag="t"
                )
            )

    def test_frozen_checkpoint_loads_into_plain_agent_without_optimizer(self, tmp_path):
        """What mode=eval does: a plain agent loads a frozen-run checkpoint, weights only."""
        path, donor = _make_donor(tmp_path)
        frozen = TrainingAgent(
            _make_config("regression", frozen_encoder_path=str(path), ckpt_tag="t")
        )
        frozen.update(*_batch())
        ckpt = tmp_path / "frozen_run.pt"
        frozen.save(ckpt)

        plain = TrainingAgent(_make_config("regression"))
        plain.load(str(ckpt), load_optimizer=False)
        _assert_state_equal(plain.encoder, donor["encoder_ema"])
        _assert_state_equal(plain.encoder_ema, donor["encoder_ema"])
        _assert_state_equal(plain.flow_map_ema, frozen.flow_map_ema.state_dict())
        sampled = plain.sample(torch.randn(8, 4, 2), torch.randn(8, 1, 3))
        assert sampled.shape == (8, 4, 2)

    def test_frozen_checkpoint_optimizer_does_not_fit_plain_agent(self, tmp_path):
        """A restart that forgets frozen_encoder_path must fail loudly, not silently unfreeze."""
        path, _ = _make_donor(tmp_path)
        frozen = TrainingAgent(
            _make_config("regression", frozen_encoder_path=str(path), ckpt_tag="t")
        )
        frozen.update(*_batch())
        ckpt = tmp_path / "frozen_run.pt"
        frozen.save(ckpt)

        plain = TrainingAgent(_make_config("regression"))
        with pytest.raises(ValueError, match="parameter group"):
            plain.load(str(ckpt), load_optimizer=True)
