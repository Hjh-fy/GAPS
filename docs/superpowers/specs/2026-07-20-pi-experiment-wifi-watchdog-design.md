# Raspberry Pi experiment Wi-Fi watchdog - design

**Date:** 2026-07-20
**Status:** approved design; awaiting implementation-plan review

## Purpose

Keep a long-running Flower experiment recoverable when the Raspberry Pi remains associated with the laboratory Wi-Fi and can reach its default gateway, but the Wi-Fi access point stops forwarding traffic between the Pi and the experiment controller PC.

The watchdog is deliberately opt-in: it runs only after a researcher starts it for an experiment and stops immediately when the researcher stops it.

## Observed failure this addresses

The Pi (`192.168.31.184`) has previously retained its address and reported `wlan0` connected while SSH from the controller PC (`192.168.31.165`) timed out. The Pi's kernel log subsequently recorded an AP deauthentication (reason 6), after which it reassociated with the same AP and recovered in about two seconds.

This points to a Wi-Fi/AP association or forwarding-state desynchronization, rather than an SSH daemon failure. A gateway-only check would not detect that state: the gateway (`192.168.31.1`) can still answer while the PC is unreachable.

## Scope

### In scope

- A watchdog running on the Raspberry Pi only.
- Manual `start`, `stop`, and `status` control for each experiment.
- Probing both the local gateway and the experiment controller PC.
- Reassociating `wlan0` through NetworkManager after a sustained, targeted failure.
- Timestamped diagnostic logs suitable for later incident review.

### Out of scope

- A Windows-side watchdog or an automated controller integration.
- Automatic enablement at boot.
- Changing Wi-Fi SSID, BSSID preference, DHCP reservations, SSH settings, or radio power-save configuration.
- Restarting the Flower client, SSH service, or the whole Pi.
- Recovery when the gateway itself is unreachable; this is treated as a broader network outage and only logged.

## Fixed environment values

| Item | Value |
| --- | --- |
| Wi-Fi interface | `wlan0` |
| Wi-Fi connection | `LAB_T504_5G` |
| Gateway | `192.168.31.1` |
| Experiment controller PC | `192.168.31.165` |
| Pi account | `gaps` |

The controller PC is confirmed to remain powered on and to accept ICMP echo requests throughout an experiment.

## Recommended architecture

Install one root-owned, **disabled-by-default** systemd service named `gaps-wifi-watchdog.service`. The service runs a small shell watchdog script on the Pi.

The service is not enabled at boot and has no relationship to login sessions or SSH terminals. A researcher explicitly starts it before the experiment and stops it after the experiment:

```bash
sudo systemctl start gaps-wifi-watchdog
sudo systemctl status gaps-wifi-watchdog --no-pager
sudo systemctl stop gaps-wifi-watchdog
```

`journalctl -u gaps-wifi-watchdog` is the canonical event log. This is preferred to a background script because it keeps running after the launching SSH session drops, has a dependable stop action, and permits privileged NetworkManager recovery without passwordless `sudo` workarounds.

## Detection and recovery state machine

The service samples the two endpoints every 10 seconds. Each ping uses `wlan0`, sends one ICMP packet, and has a short timeout so a failed check cannot stall the loop.

| Gateway | Controller PC | Action |
| --- | --- | --- |
| reachable | reachable | Reset the PC-failure counter; log only state transitions. |
| unreachable | either | Reset the counter and log a gateway outage; do not re-associate Wi-Fi. |
| reachable | unreachable | Increment the consecutive failure counter. |

At six consecutive `gateway reachable + controller unreachable` samples (nominally one minute), the service performs a recovery attempt:

1. Write a journal event with the failed-sample count and current Wi-Fi diagnostics.
2. Issue `nmcli device disconnect wlan0`.
3. Wait briefly, then issue `nmcli device connect wlan0` so NetworkManager reassociates using `LAB_T504_5G`.
4. Allow a bounded reconnection window, then probe both endpoints again and record the outcome.
5. Enter a 90-second cooldown. During cooldown the service continues to log state but cannot issue another re-association.

After any recovery attempt, the PC-failure counter resets. The service returns to ordinary sampling after the cooldown.

The threshold and cooldown are constants exposed near the beginning of the script, so they can be adjusted later without changing the control interface.

## Logging and diagnosability

All meaningful events are sent to the service journal with a stable prefix:

- service started/stopped;
- gateway-up/down transitions;
- controller-PC-up/down transitions;
- each qualifying consecutive failure count;
- recovery trigger, NetworkManager command results, and post-recovery probe results;
- current `iw dev wlan0 link` summary at recovery time, where available.

Useful post-incident commands:

```bash
sudo journalctl -u gaps-wifi-watchdog --since '2026-07-20 17:00' --no-pager
sudo journalctl -k --since '2026-07-20 17:00' --no-pager | grep -Ei 'brcmfmac|wlan0|deauth|disconnect'
```

## Safety properties

- The service has no `enable` step, so a reboot leaves it inactive.
- It never reconnects merely because the gateway is unavailable.
- Six consecutive qualified failures protect against isolated packet loss.
- The 90-second cooldown prevents repeated disconnection loops.
- Its recovery is limited to `wlan0`; it does not restart services, terminate Flower jobs, or reboot the Pi.
- Stop is immediate and cancels future checks/reassociations.

## Acceptance criteria

1. `start` launches exactly one running watchdog and `stop` leaves no watchdog process.
2. With both endpoints reachable, the service does not invoke NetworkManager recovery.
3. A single PC ping failure, or fewer than six consecutive qualifying failures, does not invoke recovery.
4. Six consecutive controller failures while gateway probes succeed invoke exactly one `wlan0` re-association.
5. A gateway outage never invokes a `wlan0` re-association.
6. A second qualifying failure during the 90-second cooldown does not invoke a second recovery.
7. The service journal provides enough timestamps and command outcomes to reconstruct each recovery.
8. The Pi can regain its normal IP connectivity after a controlled recovery test.

## Installation preconditions

Before installation, verify on the recovered Pi that:

- NetworkManager is the active Wi-Fi manager and owns `wlan0`.
- `nmcli device disconnect/connect wlan0` works when invoked as root.
- the configured connection name is still `LAB_T504_5G`;
- the controller PC responds to `ping -I wlan0 192.168.31.165`.

The initial installation and each manual start/stop use `sudo`; this avoids granting background network-control privileges to the ordinary `gaps` account.
