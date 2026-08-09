from scripts.deploy_iotj_strict_dataset import deployment_targets
from scripts.run_iotj_scientific_validation_queue import required_a0t_markers


def test_strict_dataset_deploys_to_exact_three_frozen_roots():
    targets = deployment_targets()
    assert targets == [
        ("root@121.40.139.213", "/root/GAPS/dataset"),
        ("gaps@192.168.137.172", "/home/gaps/GAPS/flower_runtime/dataset"),
        ("root@114.55.171.63", "/root/GAPS/confirmation_c2_data"),
    ]


def test_queue_requires_all_three_a0t_fixed_endpoints():
    paths = required_a0t_markers()
    assert len(paths) == 3
    assert {path.parent.name for path in paths} == {
        "CANONICAL-V1-A0T-C3", "CANONICAL-V1-A0T-C4", "CANONICAL-V1-A0T-C5"
    }
