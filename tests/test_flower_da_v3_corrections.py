from __future__ import annotations

import subprocess
import sys

import torch

from gaps_flower.domain_adaptation import (
    ServerDomainAdaptation,
    cross_domain_same_class_phase_mmd2,
    wasserstein_feature_objective,
)
from gaps_flower.strategy import GapsStrategy
from model import GradientReversalLayer
from utils import compute_mmd2


def bare_trainer(hyperparams: dict[str, object]) -> ServerDomainAdaptation:
    trainer = object.__new__(ServerDomainAdaptation)
    trainer.device = torch.device("cpu")
    trainer.hp = hyperparams
    trainer.semantic_protos = torch.nn.ParameterDict()
    return trainer


def test_compute_mmd2_has_gradient_for_shifted_distributions() -> None:
    source = torch.tensor(
        [[-1.0, 0.0], [-0.5, 0.2], [0.0, -0.1], [0.5, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor(
        [[1.0, 0.0], [1.5, 0.2], [2.0, -0.1], [2.5, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss = compute_mmd2(source, target)
    loss.backward()

    assert loss.ndim == 0
    assert float(loss.detach()) > 0.0
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert target.grad is not None and torch.isfinite(target.grad).all()
    assert float(source.grad.abs().sum()) > 0.0
    assert float(target.grad.abs().sum()) > 0.0


def test_compute_mmd2_does_not_mutate_global_torch_rng() -> None:
    source = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float32).view(-1, 1)
    target = source + 0.25
    torch.manual_seed(20260712)
    state_before = torch.random.get_rng_state().clone()

    compute_mmd2(source, target, seed=42)

    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_cross_domain_stage_mmd2_matches_same_class_phase_cells_only() -> None:
    source = torch.tensor(
        [[0.0], [0.2], [4.0], [4.2], [20.0], [20.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    source_class = torch.tensor([0, 0, 0, 0, 1, 1])
    source_phase = torch.tensor([0, 0, 1, 1, 0, 0])
    target = torch.tensor(
        [[0.8], [1.0], [4.8], [5.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target_class = torch.tensor([0, 0, 0, 0])
    target_phase = torch.tensor([0, 0, 1, 1])

    loss = cross_domain_same_class_phase_mmd2(
        source,
        source_class,
        source_phase,
        target,
        target_class,
        target_phase,
        num_classes=2,
    )
    expected = torch.stack(
        [
            compute_mmd2(source[:2], target[:2]),
            compute_mmd2(source[2:4], target[2:4]),
        ]
    ).mean()
    loss.backward()

    assert torch.allclose(loss.detach(), expected.detach())
    assert source.grad is not None and target.grad is not None
    assert float(source.grad[:4].abs().sum()) > 0.0
    assert float(target.grad.abs().sum()) > 0.0
    assert torch.equal(source.grad[4:], torch.zeros_like(source.grad[4:]))


def test_wasserstein_feature_objective_reduces_fixed_critic_gap() -> None:
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(1.0)
    source = torch.tensor([[2.0], [3.0]], requires_grad=True)
    target = torch.tensor([[0.0], [1.0]], requires_grad=True)
    before = float((critic(source).mean() - critic(target).mean()).detach())

    loss = wasserstein_feature_objective(critic, source, target)
    loss.backward()
    with torch.no_grad():
        source -= 0.1 * source.grad
        target -= 0.1 * target.grad
    after = float((critic(source).mean() - critic(target).mean()).detach())

    assert after < before
    assert critic.weight.grad is None


def test_corrected_mmd_mode_uses_mmd2_without_outer_square() -> None:
    trainer = bare_trainer(
        {
            "USE_MMD_ALIGNMENT": True,
            "MMD_OBJECTIVE": "mmd2",
            "LAMBDA_PROTO_ANCHOR": 0.0,
            "NUM_CLASSES": 1,
        }
    )
    source = torch.tensor([[0.0], [0.2], [0.4]], requires_grad=True)
    target = torch.tensor([[1.0], [1.2], [1.4]], requires_grad=True)
    labels = torch.zeros(3, dtype=torch.long)

    global_loss, class_loss, anchor = trainer._compute_mmd_losses(
        source, labels, target, labels
    )
    expected = compute_mmd2(source, target)

    assert torch.allclose(global_loss, expected)
    assert torch.allclose(class_loss, expected)
    assert float(anchor) == 0.0


def test_corrected_stage_mode_uses_cross_domain_same_class_phase_cells() -> None:
    trainer = bare_trainer(
        {
            "USE_MMD_ALIGNMENT": True,
            "STAGE_ALIGNMENT": "cross_domain_same_class_phase",
            "NUM_CLASSES": 2,
        }
    )
    source = torch.tensor(
        [[0.0], [0.2], [4.0], [4.2], [20.0], [20.2]], requires_grad=True
    )
    source_class = torch.tensor([0, 0, 0, 0, 1, 1])
    source_phase = torch.tensor([0, 0, 1, 1, 0, 0])
    target = torch.tensor([[0.8], [1.0], [4.8], [5.0]], requires_grad=True)
    target_class = torch.tensor([0, 0, 0, 0])
    target_phase = torch.tensor([0, 0, 1, 1])

    loss = trainer._compute_stage_mmd_loss(
        source,
        source_class,
        source_phase,
        target,
        target_class,
        target_phase,
    )
    expected = cross_domain_same_class_phase_mmd2(
        source,
        source_class,
        source_phase,
        target,
        target_class,
        target_phase,
        num_classes=2,
    )

    assert torch.allclose(loss, expected)


def test_corrected_adversarial_mode_decreases_gap_without_critic_gradient() -> None:
    trainer = bare_trainer(
        {
            "ADV_CRITIC_ITERS": 0,
            "ADV_GRADIENT_PENALTY": 0.0,
            "ADV_CLASS_CONDITIONAL": False,
            "ADV_FEATURE_OBJECTIVE": "wasserstein_min",
        }
    )
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(1.0)
    trainer.domain_discriminator = critic
    trainer.disc_optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)
    trainer.grl = GradientReversalLayer(lambda_grl=1.0)
    source = torch.tensor([[2.0], [3.0]], requires_grad=True)
    target = torch.tensor([[0.0], [1.0]], requires_grad=True)
    before = float((critic(source).mean() - critic(target).mean()).detach())

    loss = trainer._compute_adversarial_loss(source, None, target, None)
    loss.backward()
    with torch.no_grad():
        source -= 0.1 * source.grad
        target -= 0.1 * target.grad
    after = float((critic(source).mean() - critic(target).mean()).detach())

    assert after < before
    assert critic.weight.grad is None


def test_gaps_strategy_records_corrected_da_modes(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    strategy = GapsStrategy(
        parameter_keys=list(model.state_dict()),
        reference_state=model.state_dict(),
        output_dir=str(tmp_path),
        run_name="v3_modes",
        da_mmd_objective="mmd2",
        da_stage_alignment="cross_domain_same_class_phase",
        da_adv_feature_objective="wasserstein_min",
    )

    assert strategy.da_mmd_objective == "mmd2"
    assert strategy.da_stage_alignment == "cross_domain_same_class_phase"
    assert strategy.da_adv_feature_objective == "wasserstein_min"


def test_server_cli_exposes_versioned_da_modes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gaps_flower.server_app", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--da-mmd-objective" in result.stdout
    assert "--da-stage-alignment" in result.stdout
    assert "--da-adv-feature-objective" in result.stdout


def test_da_method_definition_marks_detached_pair_l2_as_non_trainable() -> None:
    trainer = bare_trainer(
        {
            "MMD_OBJECTIVE": "mmd2",
            "STAGE_ALIGNMENT": "cross_domain_same_class_phase",
            "ADV_FEATURE_OBJECTIVE": "wasserstein_min",
            "USE_PROTO_MMD": True,
            "LAMBDA_PROTO_MMD": 0.2,
        }
    )

    definition = trainer._method_definition()

    assert definition["mmd_objective"] == "mmd2"
    assert definition["stage_alignment"] == "cross_domain_same_class_phase"
    assert definition["adv_feature_objective"] == "wasserstein_min"
    assert definition["proto_pair_l2_enabled"] is True
    assert definition["proto_pair_l2_trainable"] is False
