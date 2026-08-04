from collections import OrderedDict

import torch

from gaps_flower.state_fingerprint import model_parameter_inventory


def test_model_parameter_inventory_has_unambiguous_semantics():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model[1].bias.requires_grad_(False)

    inventory = model_parameter_inventory(model)

    assert inventory == {
        "total_parameter_count": 11,
        "trainable_parameter_count": 10,
        "fp32_model_bytes": 44,
    }


def test_checkpoint_provenance_renames_state_tensor_count(tmp_path):
    from gaps_flower.state_fingerprint import checkpoint_provenance

    checkpoint = tmp_path / "model.pth"
    torch.save(
        {
            "model_state": OrderedDict(
                [("weight", torch.ones(2, 3)), ("buffer", torch.zeros(1))]
            ),
            "parameter_keys": ["weight", "buffer"],
            "round": 25,
        },
        checkpoint,
    )

    identity = checkpoint_provenance(checkpoint)

    assert identity["state_tensor_count"] == 2
    assert "parameter_count" not in identity
