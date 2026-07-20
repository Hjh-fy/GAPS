# ECS-C2 + Pi-C1 Execution Topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Provision a remote ECS C2 client, preserve frozen B2/B5 algorithm inputs, and run only B2/B5 two-round smoke gates after topology preflight passes.

**Architecture:** The controller gains a separate remote-C2 placement rather than relabeling the local PC. It deploys the existing content-addressed source archive, validates a C2-only data subset/dependency manifest, launches and samples C2 remotely, and recovers evidence below `raw/ecs_c2`. The controller retains its loopback server tunnel and adds an independent loopback reverse tunnel for ECS-C2, so Flower remains unexposed publicly. An execution-topology manifest binds placement and is separate from the algorithm manifest.

**Tech Stack:** Python 3.10, CPU PyTorch, Flower 1.23.0, protobuf 4.25.8, psutil 7.0.0, SSH/SCP, Ubuntu 22.04, pytest.

## Global Constraints

- Preserve source archive 52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68 and algorithm commit 2ef7aea77b9dfabdd09da4f38742907a37c58c30.
- Do not change data, C5 split, model, loss, optimizer, B2/B5 settings, rounds=25, epochs=5, batch size, learning rate, or server DA.
- Keep original ECS as server/DA, the Pi as C1, and use root@114.55.171.63 only for C2.
- Preserve PC-topology attempts. Never pool PC and ECS-C2 system timing.
- A failed preflight or smoke blocks the 25-round queue.

---

### Task 1: Bind an ECS-C2 execution topology

**Files:**
- Create: scripts/generate_iotj_ecs_c2_topology_manifest.py
- Modify: scripts/run_iotj_confirmation_observability.py
- Test: tests/test_iotj_confirmation_controller.py

**Interfaces:**
- Generator writes results/c2e_ecs_c2_topology/execution_topology_manifest.json.
- Controller accepts --c2-host, --c2-python, --c2-runtime-base and --execution-topology-manifest.
- Manifest binds `topology_id=ecs_c2_pi_c1`, `C2.host_id=ecs-c2`, `C2.ssh_host=root@114.55.171.63`, archive SHA-256 and the exact `run_id -> algorithm_config_sha256` mapping for all ten frozen runs. It additionally identifies the C2-only dataset-subset manifest and its hash.

- [ ] **Step 1: Write failing tests**

~~~python
def test_ecs_c2_topology_manifest_binds_all_frozen_run_hashes(tmp_path: Path) -> None:
    path = write_topology_manifest(
        tmp_path, archive_sha="a" * 64,
        config_by_run={"c12_to_c5__b2__s42": "b" * 64, "c12_to_c5__b5__s42": "c" * 64},
    )
    actual = controller.load_execution_topology_manifest(
        path, expected_archive_sha="a" * 64,
        expected_config_by_run={"c12_to_c5__b2__s42": "b" * 64, "c12_to_c5__b5__s42": "c" * 64},
    )
    assert actual["topology_id"] == "ecs_c2_pi_c1"
    assert actual["hosts"]["C2"]["host_id"] == "ecs-c2"


def test_ecs_c2_topology_manifest_rejects_run_config_mismatch(tmp_path: Path) -> None:
    path = write_topology_manifest(tmp_path, archive_sha="a" * 64, config_by_run={"c12_to_c5__b2__s42": "b" * 64})
    with pytest.raises(RuntimeError, match="algorithm config"):
        controller.load_execution_topology_manifest(
            path, expected_archive_sha="a" * 64,
            expected_config_by_run={"c12_to_c5__b2__s42": "c" * 64},
        )
~~~

- [ ] **Step 2: Verify RED**

Run: python -m pytest tests/test_iotj_confirmation_controller.py -k "ecs_c2_topology_manifest" -q
Expected: FAIL because generator and loader do not exist.

- [ ] **Step 3: Implement the contract**

~~~python
def load_execution_topology_manifest(
    path: Path, *, expected_archive_sha: str, expected_config_by_run: Mapping[str, str]
) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("topology_id") != "ecs_c2_pi_c1":
        raise RuntimeError("execution topology identifier mismatch")
    if payload.get("source_archive_sha256") != expected_archive_sha:
        raise RuntimeError("execution topology source archive mismatch")
    if payload.get("algorithm_config_sha256_by_run") != dict(expected_config_by_run):
        raise RuntimeError("execution topology algorithm config mapping mismatch")
    if payload.get("hosts", {}).get("C2") != {
        "host_id": "ecs-c2", "ssh_host": "root@114.55.171.63"
    }:
        raise RuntimeError("execution topology C2 host mismatch")
    return payload
~~~

Generator reads all ten run identities and hashes from the immutable protocol manifest, writes deterministic host records plus its own SHA-256, and rejects a missing, extra, or mismatched run. Legacy PC mode is unchanged unless `--c2-host` is supplied.

- [ ] **Step 4: Verify GREEN and commit**

Run: python -m pytest tests/test_iotj_confirmation_controller.py -k "ecs_c2_topology_manifest or frozen" -q
Expected: PASS.

~~~bash
git add scripts/generate_iotj_ecs_c2_topology_manifest.py scripts/run_iotj_confirmation_observability.py tests/test_iotj_confirmation_controller.py
git commit -m "feat: bind ECS C2 execution topology"
~~~

### Task 2: Remote C2 controller lifecycle

**Files:**
- Modify: scripts/run_iotj_confirmation_observability.py
- Modify: tests/test_iotj_confirmation_controller.py
- Modify: 代码文件介绍.md

**Interfaces:**
- Remote C2 uses `/root/gaps_c2_cpu_env/bin/python` and `/root/GAPS/confirmation_runtime_c2/<archive-sha>`.
- Deployment returns HostDeployment("ecs_c2", ...).
- C2 observer contexts use host_id="ecs-c2"; recovered evidence goes to raw/ecs_c2.
- Existing no---c2-host calls still return HostDeployment("pc", ...).

- [ ] **Step 1: Write failing deployment/lifecycle tests**

~~~python
def test_remote_c2_deployment_uses_only_remote_runtime(tmp_path: Path) -> None:
    archive, manifest = _source_fixture(tmp_path / "source")
    calls: list[tuple[str, str]] = []
    deployed = controller.deploy_source_archive(
        archive, manifest, ecs_host="root@server", pi_host="gaps@pi",
        c2_host="root@c2", c2_python="/root/gaps_c2_cpu_env/bin/python",
        c2_runtime_base="/root/GAPS/confirmation_runtime_c2",
        pc_runtime_root=tmp_path / "runtime", run=fake_run, ssh=fake_ssh,
        remote_python=lambda host, python, source, **kwargs:
            calls.append((host, python)) or remote_complete_report(manifest),
    )
    assert set(deployed) == {"ecs", "pi", "ecs_c2"}
    assert ("root@c2", "/root/gaps_c2_env/bin/python") in calls


def test_remote_c2_evidence_root_is_distinct(tmp_path: Path) -> None:
    assert controller.remote_c2_raw_path(tmp_path / "attempt") == tmp_path / "attempt/raw/ecs_c2"
~~~

- [ ] **Step 2: Verify RED**

Run: python -m pytest tests/test_iotj_confirmation_controller.py -k "remote_c2" -q
Expected: FAIL because remote-C2 parameters and helper do not exist.

- [ ] **Step 3: Implement minimal lifecycle**

Add allowed remote Python `/root/gaps_c2_cpu_env/bin/python`. Parameterize `deploy_source_archive`: when `c2_host` is present, reuse existing remote archive reserve/SCP/extract logic for `ecs_c2` and skip local PC runtime creation. Parameterize C2 preflight, launch, sampler stop, process ownership, evidence copy and cleanup to use remote C2. Add a controller-local tunnel helper that starts (1) the existing local `-L` server tunnel, (2) the Pi loopback `-R`, and (3) a separate ECS-C2 loopback `-R`, with all three owned and cleaned up by the same attempt. Do not create a second launcher or alter legacy PC behavior.

- [ ] **Step 4: Verify controller regressions and commit**

Run: python -m pytest tests/test_iotj_confirmation_controller.py tests/test_confirmation_resource_sampler.py tests/test_confirmation_observability.py -q
Expected: PASS.

~~~bash
git add scripts/run_iotj_confirmation_observability.py tests/test_iotj_confirmation_controller.py 代码文件介绍.md
git commit -m "feat: run confirmation C2 on remote ECS"
~~~

### Task 3: C2 provisioning, preflight and smoke gates

**Files:**
- Create: results/c2e_ecs_c2_topology/provisioning_record.json
- Create: results/c2e_ecs_c2_topology/execution_topology_manifest.json
- Create: results/c2e_ecs_c2_topology/preflight_report.json
- Create: results/c2e_ecs_c2_smoke_b2/
- Create: results/c2e_ecs_c2_smoke_b5/
- Modify: docs/experiments/iotj_system_experiment_notebook.md

**Interfaces:**
- C2 receives `/root/gaps_c2_cpu_env`, a 2 GiB swapfile, immutable archive/C2-only data, a dependency report and a fail-closed preflight. The discarded CUDA-oriented `/root/gaps_c2_env` is recorded as an unused provisioning attempt; it must not be selected by any controller option.
- Smoke outputs are noncanonical and label C2 as ecs-c2.

- [ ] **Step 1: Provision isolated host runtime**

Run on C2:

~~~bash
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
python3 -m venv /root/gaps_c2_cpu_env
/root/gaps_c2_cpu_env/bin/pip install --upgrade pip
/root/gaps_c2_cpu_env/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0+cpu
/root/gaps_c2_cpu_env/bin/pip install flwr==1.23.0 protobuf==4.25.8 psutil==7.0.0 numpy scikit-learn pandas
/root/gaps_c2_cpu_env/bin/python -c 'import torch,flwr,numpy,psutil; print(torch.__version__, flwr.__version__, numpy.__version__, psutil.__version__)'
~~~

Expected: swap active and imports succeed. Record exact versions; if this fails, stop before source/data transfer.

- [ ] **Step 2: Generate manifest, deploy and preflight**

Run the Task-1 generator from frozen input manifests, then generate and transfer only the C2 subset manifest/data. Run the controller with `--preflight-only`, `--c2-host root@114.55.171.63`, `--c2-python /root/gaps_c2_cpu_env/bin/python`, `--c2-runtime-base /root/GAPS/confirmation_runtime_c2` and the topology manifest.

Expected: ECS server, Pi C1 and ECS C2 agree on archive/member/dataset/config hashes, expected dependencies and zero active attempts. Preflight creates no Flower training output.

- [ ] **Step 3: Run B2 then B5 two-round smokes**

Run B2 seed 42 in a new noncanonical smoke root. Validate message, event, resource, process, checkpoint/schema and provenance contracts before B5. Run B5 only after B2 passes. Preserve failures and write smoke_gate_summary.md with topology, hashes, package/RAM/swap inventory, and two-round diagnostic timing/RSS; never claim multi-seed metrics.

- [ ] **Step 4: Commit lightweight provenance**

~~~bash
git add results/c2e_ecs_c2_topology docs/experiments/iotj_system_experiment_notebook.md
git commit -m "docs: record ECS C2 preflight and smoke gates"
~~~

## Self-review

- Task 1 binds execution topology to immutable algorithm inputs.
- Task 2 changes orchestration only and retains legacy PC behavior through regression tests.
- Task 3 installs isolated dependencies, validates transfer identity, and blocks the queue unless B2 then B5 smokes pass.
- No task changes the algorithm or treats smoke timing/metrics as confirmation evidence.
