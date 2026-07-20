# Raspberry Pi Experiment Wi-Fi Watchdog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a manually started Raspberry Pi service that detects the specific "gateway reachable but experiment PC unreachable" Wi-Fi failure and safely reassociates `wlan0`.

**Architecture:** Keep the policy in one dependency-free Python 3 program so its timing, counter, cooldown, and NetworkManager commands can be unit tested locally with pytest. Run it as a root-owned systemd service that is deliberately not enabled at boot; stdout becomes the service journal. Ship the program, its unit file, and operator instructions together under `scripts/pi_wifi_watchdog/` so installation is a small, auditable copy operation on the Pi.

**Tech Stack:** Python 3 standard library, pytest, Raspberry Pi OS systemd, NetworkManager `nmcli`, `ping`, `iw`.

## Global Constraints

- Watchdog runs on the Raspberry Pi only; do not add a Windows-side monitor or modify the Flower controller.
- It must be started and stopped manually with `sudo systemctl`; do not enable it at boot.
- Probe `wlan0`, gateway `192.168.31.1`, and controller PC `192.168.31.165` every 10 seconds.
- Recover only after 6 consecutive samples where gateway succeeds and controller PC fails.
- Recovery must be limited to `nmcli device disconnect wlan0` followed by `nmcli device connect wlan0`.
- Reset the failure counter for a gateway failure and never recover during a gateway outage.
- Enforce a 90-second recovery cooldown and log all decisions and NetworkManager outcomes to journald.
- Do not restart Flower, SSH, the whole Pi, or alter SSID/BSSID/power-save settings.

---

## File structure

- `scripts/pi_wifi_watchdog/gaps_wifi_watchdog.py` - dependency-free watchdog policy and process entry point.
- `scripts/pi_wifi_watchdog/gaps-wifi-watchdog.service` - root-owned, non-enabled systemd unit for the policy.
- `scripts/pi_wifi_watchdog/README.md` - installation, manual lifecycle, log inspection, rollback, and safe test instructions.
- `tests/test_pi_wifi_watchdog.py` - deterministic pytest coverage of the policy and static unit-file contract.
- `docs/experiments/iotj_system_experiment_notebook.md` - concise experiment-run checklist referring operators to the watchdog README.

### Task 1: Implement the testable watchdog policy

**Files:**

- Create: `tests/test_pi_wifi_watchdog.py`
- Create: `scripts/pi_wifi_watchdog/gaps_wifi_watchdog.py`

**Interfaces:**

- Produces `WatchdogConfig`, `Watchdog`, and `main()` from `scripts.pi_wifi_watchdog.gaps_wifi_watchdog`.
- `Watchdog.run_once()` performs one probe-and-possibly-recover cycle and returns `None`.
- `Watchdog` accepts injected `runner`, `sleep`, `clock`, and `log` callables so tests never invoke real network commands.

- [ ] **Step 1: Write the failing policy tests**

Create `tests/test_pi_wifi_watchdog.py` with the following exact test scaffold. It describes all safety-critical outcomes without requiring a Pi:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.pi_wifi_watchdog.gaps_wifi_watchdog import Watchdog, WatchdogConfig


@dataclass
class Result:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeRunner:
    def __init__(self, gateway: list[bool], controller: list[bool]) -> None:
        self.gateway = iter(gateway)
        self.controller = iter(controller)
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], _timeout: float) -> Result:
        self.calls.append(args)
        if args[0] == "ping":
            ok = next(self.gateway if args[-1] == "192.168.31.1" else self.controller)
            return Result(returncode=0 if ok else 1)
        if args[0] == "iw":
            return Result(stdout="Connected to 44:F7:70:2A:AC:0F\n")
        return Result()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_watchdog(runner: FakeRunner, clock: Clock) -> tuple[Watchdog, list[str]]:
    logs: list[str] = []
    config = WatchdogConfig(
        interval_seconds=10,
        failures_before_recover=6,
        cooldown_seconds=90,
        reconnect_wait_seconds=0,
    )
    return Watchdog(config, runner, lambda _seconds: None, clock, logs.append), logs


def test_healthy_sample_resets_failure_counter_without_recovery() -> None:
    runner = FakeRunner([True], [True])
    watchdog, _logs = make_watchdog(runner, Clock())

    watchdog.consecutive_controller_failures = 5
    watchdog.run_once()

    assert watchdog.consecutive_controller_failures == 0
    assert not any(call[:3] == ["nmcli", "device", "disconnect"] for call in runner.calls)


def test_gateway_outage_never_recovers_and_resets_counter() -> None:
    runner = FakeRunner([False], [False])
    watchdog, logs = make_watchdog(runner, Clock())

    watchdog.consecutive_controller_failures = 5
    watchdog.run_once()

    assert watchdog.consecutive_controller_failures == 0
    assert not any(call[0] == "nmcli" for call in runner.calls)
    assert any("gateway unreachable; recovery suppressed" in line for line in logs)


def test_six_qualified_failures_reassociate_wlan0_once() -> None:
    runner = FakeRunner([True] * 7, [False] * 7)
    watchdog, logs = make_watchdog(runner, Clock())

    for _ in range(6):
        watchdog.run_once()

    assert [call for call in runner.calls if call[0] == "nmcli"] == [
        ["nmcli", "device", "disconnect", "wlan0"],
        ["nmcli", "device", "connect", "wlan0"],
    ]
    assert any("recovery triggered after 6 qualified failures" in line for line in logs)


def test_cooldown_blocks_a_second_reassociation() -> None:
    runner = FakeRunner([True] * 14, [False] * 14)
    clock = Clock()
    watchdog, _logs = make_watchdog(runner, clock)

    for _ in range(12):
        watchdog.run_once()

    assert len([call for call in runner.calls if call[0] == "nmcli"]) == 2


def test_systemd_unit_is_manual_and_runs_the_installed_program() -> None:
    unit = Path("scripts/pi_wifi_watchdog/gaps-wifi-watchdog.service").read_text(encoding="utf-8")

    assert "ExecStart=/usr/bin/python3 /usr/local/lib/gaps-wifi-watchdog/gaps_wifi_watchdog.py" in unit
    assert "User=root" in unit
    assert "WantedBy=" not in unit
    assert "Restart=" not in unit
```

- [ ] **Step 2: Run the new tests and verify the expected failure**

Run:

```powershell
python -m pytest tests/test_pi_wifi_watchdog.py -v
```

Expected: collection fails because `scripts.pi_wifi_watchdog.gaps_wifi_watchdog` does not exist yet.

- [ ] **Step 3: Write the minimal watchdog implementation**

Create `scripts/pi_wifi_watchdog/gaps_wifi_watchdog.py` with this implementation:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Protocol


class CommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[list[str], float], CommandResult]
Logger = Callable[[str], None]


@dataclass(frozen=True)
class WatchdogConfig:
    interface: str = "wlan0"
    gateway: str = "192.168.31.1"
    controller: str = "192.168.31.165"
    interval_seconds: int = 10
    failures_before_recover: int = 6
    cooldown_seconds: int = 90
    reconnect_wait_seconds: int = 8
    ping_timeout_seconds: int = 2


def run_command(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr=str(error))


class Watchdog:
    def __init__(
        self,
        config: WatchdogConfig,
        runner: Runner = run_command,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        log: Logger | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.sleep = sleep
        self.clock = clock
        self.log = log if log is not None else self._log
        self.consecutive_controller_failures = 0
        self.cooldown_until = 0.0
        self.gateway_reachable: bool | None = None
        self.controller_reachable: bool | None = None

    @staticmethod
    def _log(message: str) -> None:
        print(f"gaps-wifi-watchdog: {message}", flush=True)

    def _probe(self, address: str) -> bool:
        result = self.runner(
            [
                "ping", "-I", self.config.interface, "-c", "1", "-W",
                str(self.config.ping_timeout_seconds), address,
            ],
            float(self.config.ping_timeout_seconds + 2),
        )
        return result.returncode == 0

    def _log_transition(self, name: str, previous: bool | None, current: bool) -> None:
        if previous is None or previous != current:
            self.log(f"{name} {'reachable' if current else 'unreachable'}")

    def _wifi_summary(self) -> str:
        result = self.runner(["iw", "dev", self.config.interface, "link"], 5.0)
        summary = " ".join(result.stdout.split()) if result.returncode == 0 else result.stderr.strip()
        return summary or "unavailable"

    def _recover(self) -> None:
        self.log(
            f"recovery triggered after {self.config.failures_before_recover} qualified failures; "
            f"wifi={self._wifi_summary()}"
        )
        for command in (
            ["nmcli", "device", "disconnect", self.config.interface],
            ["nmcli", "device", "connect", self.config.interface],
        ):
            result = self.runner(command, 30.0)
            self.log(
                f"command={' '.join(command)} returncode={result.returncode} "
                f"stderr={result.stderr.strip() or '-'}"
            )
            if command[2] == "disconnect":
                self.sleep(2)
        self.sleep(self.config.reconnect_wait_seconds)
        post_gateway = self._probe(self.config.gateway)
        post_controller = self._probe(self.config.controller)
        self.log(
            f"post-recovery gateway={'up' if post_gateway else 'down'} "
            f"controller={'up' if post_controller else 'down'}"
        )

    def run_once(self) -> None:
        gateway = self._probe(self.config.gateway)
        controller = self._probe(self.config.controller)
        self._log_transition("gateway", self.gateway_reachable, gateway)
        self._log_transition("controller", self.controller_reachable, controller)
        self.gateway_reachable = gateway
        self.controller_reachable = controller

        if not gateway:
            self.consecutive_controller_failures = 0
            self.log("gateway unreachable; recovery suppressed")
            return
        if controller:
            self.consecutive_controller_failures = 0
            return

        self.consecutive_controller_failures += 1
        self.log(f"qualified controller failure count={self.consecutive_controller_failures}")
        if self.consecutive_controller_failures < self.config.failures_before_recover:
            return
        if self.clock() < self.cooldown_until:
            self.log("recovery cooldown active; recovery suppressed")
            return

        self._recover()
        self.consecutive_controller_failures = 0
        self.cooldown_until = self.clock() + self.config.cooldown_seconds
        self.log(f"recovery cooldown active for {self.config.cooldown_seconds} seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run one probe cycle and exit")
    args = parser.parse_args()
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    watchdog = Watchdog(WatchdogConfig())
    watchdog.log("service started")
    while not stop_requested:
        watchdog.run_once()
        if args.once:
            break
        watchdog.sleep(watchdog.config.interval_seconds)
    watchdog.log("service stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Also create an empty `scripts/pi_wifi_watchdog/__init__.py` so the test import is portable across supported Python versions.

- [ ] **Step 4: Run the policy tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_pi_wifi_watchdog.py -v
```

Expected: the first four policy tests pass; the unit-file test still fails until Task 2 creates the unit.

- [ ] **Step 5: Commit the policy and its tests**

```powershell
git add tests/test_pi_wifi_watchdog.py scripts/pi_wifi_watchdog/gaps_wifi_watchdog.py scripts/pi_wifi_watchdog/__init__.py
git commit -m "feat: add Pi Wi-Fi watchdog policy"
```

### Task 2: Add the manual systemd service and operator instructions

**Files:**

- Create: `scripts/pi_wifi_watchdog/gaps-wifi-watchdog.service`
- Create: `scripts/pi_wifi_watchdog/README.md`
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`
- Modify: `tests/test_pi_wifi_watchdog.py`

**Interfaces:**

- Consumes the installed program path `/usr/local/lib/gaps-wifi-watchdog/gaps_wifi_watchdog.py` from Task 1.
- Produces a manually runnable service `gaps-wifi-watchdog.service` and the exact install/control commands used by operators.

- [ ] **Step 1: Extend the failing static service contract**

Add this test to `tests/test_pi_wifi_watchdog.py`:

```python
def test_service_orders_after_networkmanager_and_preserves_manual_lifecycle() -> None:
    unit = Path("scripts/pi_wifi_watchdog/gaps-wifi-watchdog.service").read_text(encoding="utf-8")

    assert "After=NetworkManager.service" in unit
    assert "Wants=NetworkManager.service" in unit
    assert "Type=simple" in unit
    assert "Restart=" not in unit
    assert "[Install]" not in unit
```

- [ ] **Step 2: Run the service-contract test and verify it fails**

Run:

```powershell
python -m pytest tests/test_pi_wifi_watchdog.py::test_systemd_unit_is_manual_and_runs_the_installed_program tests/test_pi_wifi_watchdog.py::test_service_orders_after_networkmanager_and_preserves_manual_lifecycle -v
```

Expected: FAIL because `gaps-wifi-watchdog.service` is absent.

- [ ] **Step 3: Create the unit and instructions**

Create `scripts/pi_wifi_watchdog/gaps-wifi-watchdog.service`:

```ini
[Unit]
Description=GAPS experiment Wi-Fi reachability watchdog
Wants=NetworkManager.service
After=NetworkManager.service

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /usr/local/lib/gaps-wifi-watchdog/gaps_wifi_watchdog.py
KillSignal=SIGTERM
TimeoutStopSec=10
```

Create `scripts/pi_wifi_watchdog/README.md` with these exact operational commands and explanations:

```markdown
# Pi experiment Wi-Fi watchdog

Install while the Pi is reachable. These commands copy the reviewed files, load the unit, and do **not** enable or start it:

```bash
scp scripts/pi_wifi_watchdog/gaps_wifi_watchdog.py scripts/pi_wifi_watchdog/gaps-wifi-watchdog.service gaps@192.168.31.184:/tmp/
ssh -t gaps@192.168.31.184
sudo install -D -m 0755 /tmp/gaps_wifi_watchdog.py /usr/local/lib/gaps-wifi-watchdog/gaps_wifi_watchdog.py
sudo install -D -m 0644 /tmp/gaps-wifi-watchdog.service /etc/systemd/system/gaps-wifi-watchdog.service
sudo systemctl daemon-reload
sudo systemctl is-enabled gaps-wifi-watchdog.service
```

The last command must print `disabled` or `static`; do not run `systemctl enable`.

Before each experiment:

```bash
sudo systemctl start gaps-wifi-watchdog.service
sudo systemctl status gaps-wifi-watchdog.service --no-pager
```

After the experiment:

```bash
sudo systemctl stop gaps-wifi-watchdog.service
sudo systemctl status gaps-wifi-watchdog.service --no-pager
```

Review an incident:

```bash
sudo journalctl -u gaps-wifi-watchdog.service --since 'today' --no-pager
sudo journalctl -k --since 'today' --no-pager | grep -Ei 'brcmfmac|wlan0|deauth|disconnect'
```

Safe preflight before an experiment:

```bash
ping -I wlan0 -c 1 -W 2 192.168.31.1
ping -I wlan0 -c 1 -W 2 192.168.31.165
sudo /usr/bin/python3 /usr/local/lib/gaps-wifi-watchdog/gaps_wifi_watchdog.py --once
```

To remove it, first stop the unit, then remove its two installed files and reload systemd:

```bash
sudo systemctl stop gaps-wifi-watchdog.service
sudo rm -f /etc/systemd/system/gaps-wifi-watchdog.service /usr/local/lib/gaps-wifi-watchdog/gaps_wifi_watchdog.py
sudo systemctl daemon-reload
```
```

Append a short "Experiment Wi-Fi watchdog" subsection to `docs/experiments/iotj_system_experiment_notebook.md` that links to `scripts/pi_wifi_watchdog/README.md`, states it is manual-only, and records the two lifecycle commands.

- [ ] **Step 4: Run the complete watchdog test module**

Run:

```powershell
python -m pytest tests/test_pi_wifi_watchdog.py -v
```

Expected: PASS, including all policy and service-contract tests.

- [ ] **Step 5: Commit the unit, instructions, and documentation**

```powershell
git add scripts/pi_wifi_watchdog/gaps-wifi-watchdog.service scripts/pi_wifi_watchdog/README.md docs/experiments/iotj_system_experiment_notebook.md tests/test_pi_wifi_watchdog.py
git commit -m "docs: add Pi Wi-Fi watchdog operations"
```

### Task 3: Verify on the recovered Raspberry Pi and perform a controlled recovery test

**Files:**

- Verify: `scripts/pi_wifi_watchdog/gaps_wifi_watchdog.py`
- Verify: `scripts/pi_wifi_watchdog/gaps-wifi-watchdog.service`
- Verify: `scripts/pi_wifi_watchdog/README.md`

**Interfaces:**

- Consumes the Task 1 program and Task 2 unit installed at `/usr/local/lib/gaps-wifi-watchdog/` and `/etc/systemd/system/`.
- Produces evidence in `journalctl -u gaps-wifi-watchdog.service` that start/stop and one controlled reassociation work without enabling at boot.

- [ ] **Step 1: Run repository verification before deployment**

Run:

```powershell
python -m pytest tests/test_pi_wifi_watchdog.py -v
git diff --check HEAD~2..HEAD
```

Expected: pytest PASS and no whitespace errors. If the repository has unrelated modified/untracked files, do not stage or clean them.

- [ ] **Step 2: Confirm Pi preconditions without changing state**

Once SSH is reachable, run:

```bash
ssh gaps@192.168.31.184 "nmcli -t -f DEVICE,TYPE,STATE device; nmcli -g GENERAL.CONNECTION device show wlan0; ping -I wlan0 -c 1 -W 2 192.168.31.1; ping -I wlan0 -c 1 -W 2 192.168.31.165"
```

Expected: `wlan0` is connected, the connection is `LAB_T504_5G`, and both pings succeed. Stop and diagnose instead of deploying if any condition fails.

- [ ] **Step 3: Install, but do not enable, the service**

Follow the install commands in `scripts/pi_wifi_watchdog/README.md`. During the interactive SSH session, verify:

```bash
sudo systemctl daemon-reload
sudo systemctl is-enabled gaps-wifi-watchdog.service
sudo systemctl cat gaps-wifi-watchdog.service
```

Expected: `is-enabled` reports `disabled` or `static`; `systemctl cat` shows the approved unit. Do not run `enable`.

- [ ] **Step 4: Test normal manual lifecycle and journal output**

Run:

```bash
sudo systemctl start gaps-wifi-watchdog.service
sleep 12
sudo systemctl status gaps-wifi-watchdog.service --no-pager
sudo journalctl -u gaps-wifi-watchdog.service -n 30 --no-pager
sudo systemctl stop gaps-wifi-watchdog.service
sudo systemctl is-active gaps-wifi-watchdog.service
```

Expected: while active, the journal reports gateway and controller reachable; after stop, `is-active` prints `inactive`.

- [ ] **Step 5: Perform one controlled recovery test with an explicit operator window**

With no Flower run active, temporarily block only Pi-to-controller ICMP for just over one minute using a reversible iptables rule, while leaving gateway traffic untouched:

```bash
sudo iptables -I OUTPUT -o wlan0 -d 192.168.31.165 -p icmp --icmp-type echo-request -j DROP
sudo systemctl start gaps-wifi-watchdog.service
sleep 75
sudo iptables -D OUTPUT -o wlan0 -d 192.168.31.165 -p icmp --icmp-type echo-request -j DROP
sleep 15
sudo journalctl -u gaps-wifi-watchdog.service -n 100 --no-pager
sudo systemctl stop gaps-wifi-watchdog.service
```

Expected: exactly one logged recovery trigger, one `nmcli device disconnect wlan0`, one `nmcli device connect wlan0`, and a later controller-reachable transition. If the stop or rule removal cannot be performed, immediately reconnect locally and run the matching `iptables -D` command before proceeding.

- [ ] **Step 6: Commit only any corrections required by verified deployment**

If deployment required source/doc corrections, add only those named files and commit with a focused message, for example:

```powershell
git add scripts/pi_wifi_watchdog tests/test_pi_wifi_watchdog.py docs/experiments/iotj_system_experiment_notebook.md
git commit -m "fix: harden Pi Wi-Fi watchdog deployment"
```

If no corrections were needed, do not create an empty commit.

## Plan self-review

- **Spec coverage:** Task 1 implements both-endpoint probes, six-sample qualification, gateway suppression, one-interface NetworkManager recovery, cooldown, stable logging, and testable stop handling. Task 2 enforces root-owned, manual-only systemd lifecycle and operator diagnostics. Task 3 verifies NetworkManager ownership, real reachability, disabled-at-boot behavior, logs, and one controlled recovery.
- **No placeholders:** Checked for deferred-work markers and generic testing instructions; none remain.
- **Interface consistency:** The tests import `WatchdogConfig` and `Watchdog` from the package path created in Task 1; the service `ExecStart` matches the Task 2 installation destination; README commands use the identical service name and program path.
