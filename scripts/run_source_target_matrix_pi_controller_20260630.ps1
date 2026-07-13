param(
    [string]$EcsHost = "root@121.40.139.213",
    [string]$PiHost = "gaps@192.168.31.184",
    [string]$EcsProject = "/root/GAPS",
    [string]$PiProject = "/home/gaps/GAPS/flower_runtime",
    [string]$EcsResultsRoot = "results/source_target_classification_matrix_20260630",
    [string]$PiLogRoot = "results/source_target_classification_matrix_20260630_local_client_logs",
    [string]$CommandRoot = "results/source_target_classification_matrix_20260630_commands",
    [int]$PollSeconds = 120
)

$ErrorActionPreference = "Stop"

$runs = @(
    @{ Id = "F1_C1_to_C5_fixed_da_strong_r25"; DataRoot = "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"; Clients = @(1) },
    @{ Id = "F2_C12_to_C5_fixed_da_strong_r25"; DataRoot = "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"; Clients = @(1, 2) },
    @{ Id = "F3_C123_to_C5_fixed_da_strong_r25"; DataRoot = "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"; Clients = @(1, 2, 3) },
    @{ Id = "F4_C1234_to_C5_fixed_da_strong_r25"; DataRoot = "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"; Clients = @(1, 2, 3, 4) },
    @{ Id = "F5_C1_to_C2345_fixed_da_strong_r25"; DataRoot = "client_data_c1src_c2345tgt_2080_timeaware_60_170_window_fullgrid"; Clients = @(1) },
    @{ Id = "R1_C5_to_C1_fixed_da_strong_r25"; DataRoot = "client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid"; Clients = @(5) },
    @{ Id = "R2_C45_to_C1_fixed_da_strong_r25"; DataRoot = "client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid"; Clients = @(4, 5) },
    @{ Id = "R3_C345_to_C1_fixed_da_strong_r25"; DataRoot = "client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid"; Clients = @(3, 4, 5) },
    @{ Id = "R4_C2345_to_C1_fixed_da_strong_r25"; DataRoot = "client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid"; Clients = @(2, 3, 4, 5) }
)

$controllerLog = Join-Path (Get-Location) "results/source_target_classification_matrix_20260630_controller.log"
New-Item -ItemType Directory -Force -Path (Split-Path $controllerLog) | Out-Null

function Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Output $line
    Add-Content -Path $controllerLog -Value $line
}

function SshEcs([string]$Command) {
    & ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 $EcsHost $Command
}

function SshPi([string]$Command) {
    & ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 $PiHost $Command
}

function Remote-PythonCommand([string]$PythonCode) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($PythonCode)
    $encoded = [Convert]::ToBase64String($bytes)
    return "python3 -c 'import base64; exec(base64.b64decode(`"$encoded`").decode(`"utf-8`"))'"
}

function Ensure-Tunnels {
    $localForward = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match "127\.0\.0\.1:18080:127\.0\.0\.1:8080" -and $_.CommandLine -match "121\.40\.139\.213"
    }
    if (-not $localForward) {
        $p = Start-Process -FilePath ssh -ArgumentList @(
            "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-N",
            "-L", "127.0.0.1:18080:127.0.0.1:8080",
            $EcsHost
        ) -WindowStyle Hidden -PassThru
        Log "started PC->ECS tunnel pid=$($p.Id)"
        Start-Sleep -Seconds 3
    }

    $reverseForward = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match "-R 127\.0\.0\.1:18080:127\.0\.0\.1:18080" -and $_.CommandLine -match "172\.31\.139\.224"
    }
    if (-not $reverseForward) {
        $p = Start-Process -FilePath ssh -ArgumentList @(
            "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-N",
            "-R", "127.0.0.1:18080:127.0.0.1:18080",
            $PiHost
        ) -WindowStyle Hidden -PassThru
        Log "started PC->Pi reverse tunnel pid=$($p.Id)"
        Start-Sleep -Seconds 3
    }
}

function Is-RunComplete([string]$RunId) {
    $cmd = "test -f '$EcsResultsRoot/$RunId/server_latest_adapted.pth' && test -f '$EcsResultsRoot/$RunId/history.json' && test -f '$EcsResultsRoot/$RunId/client_stats_round_025.json' && echo complete || true"
    $out = SshEcs "cd '$EcsProject' && $cmd"
    return ($out -join "`n") -match "complete"
}

function Server-Pids([string]$RunId) {
    $cmd = "ps -eo pid,cmd | grep gaps_flower.server_app | grep '$RunId' | grep -v grep || true"
    return SshEcs $cmd
}

function Launch-Server([string]$RunId) {
    $py = @"
import os
import subprocess

cwd = r'''$EcsProject'''
run_id = r'''$RunId'''
output_dir = os.path.join(cwd, r'''$EcsResultsRoot''', run_id)
command_script = os.path.join(cwd, r'''$CommandRoot''', run_id, "server_command.sh")
os.makedirs(output_dir, exist_ok=True)
subprocess.run(["sed", "-i", "s/\\r$//", command_script], check=False)
log = open(os.path.join(output_dir, "server_launch.log"), "ab", buffering=0)
p = subprocess.Popen(
    ["bash", command_script],
    cwd=cwd,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=log,
    start_new_session=True,
    close_fds=True,
)
print(p.pid)
"@
    $cmd = Remote-PythonCommand $py
    $out = SshEcs $cmd
    Log "server launch requested run=$RunId output=$($out -join ' ')"
    Start-Sleep -Seconds 15
}

function Launch-Client([string]$RunId, [string]$DataRoot, [int]$ClientId) {
    $py = @"
import os
import subprocess

cwd = r'''$PiProject'''
run_id = r'''$RunId'''
client_id = "$ClientId"
log_dir = os.path.join(cwd, r'''$PiLogRoot''', run_id)
os.makedirs(log_dir, exist_ok=True)
log = open(os.path.join(log_dir, f"client_{client_id}.log"), "ab", buffering=0)
cmd = [
    "/home/gaps/GAPS/gaps_rpi_env/bin/python",
    "-m",
    "gaps_flower.client_app",
    "--server-address",
    "127.0.0.1:18080",
    "--client-id",
    client_id,
    "--data-root",
    os.path.join(cwd, "dataset", r'''$DataRoot'''),
    "--device",
    "cpu",
    "--local-epochs",
    "5",
    "--batch-size",
    "32",
    "--profile",
    "strong_cls",
]
p = subprocess.Popen(
    cmd,
    cwd=cwd,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=log,
    start_new_session=True,
    close_fds=True,
)
print(p.pid)
"@
    $cmd = Remote-PythonCommand $py
    $out = SshPi $cmd
    Log "client launch requested run=$RunId C$ClientId output=$($out -join ' ')"
}

function Start-Run($Run) {
    $runId = $Run.Id
    if (Is-RunComplete $runId) {
        Log "skip completed run=$runId"
        return
    }

    Ensure-Tunnels

    $pids = Server-Pids $runId
    if (-not $pids) {
        Launch-Server $runId
    } else {
        Log "server already running run=$runId pids=$($pids -join ' | ')"
    }

    foreach ($clientId in $Run.Clients) {
        $clientRunning = SshPi "ps -eo pid,cmd | grep gaps_flower.client_app | grep -- '--client-id $clientId' | grep '$($Run.DataRoot)' | grep -v grep || true"
        if (-not $clientRunning) {
            Launch-Client $runId $Run.DataRoot $clientId
        } else {
            Log "client already running run=$runId C$clientId"
        }
    }

    while ($true) {
        Start-Sleep -Seconds $PollSeconds
        $complete = Is-RunComplete $runId
        $server = Server-Pids $runId
        $rounds = SshEcs "cd '$EcsProject' && ls '$EcsResultsRoot/$runId'/client_stats_round_*.json 2>/dev/null | wc -l"
        $adapt = SshEcs "cd '$EcsProject' && ls '$EcsResultsRoot/$runId'/domain_adapt_round_*.json 2>/dev/null | wc -l"
        $health = SshPi "vcgencmd measure_temp 2>/dev/null; vcgencmd get_throttled 2>/dev/null; free -h | sed -n '2,3p'"
        Log "progress run=$runId complete=$complete server_running=$([bool]$server) client_stats=$($rounds -join '') domain_adapt=$($adapt -join '') pi_health=$($health -join ' | ')"
        if ($complete -and -not $server) {
            Log "finished run=$runId"
            break
        }
        if (-not $server -and -not $complete) {
            Log "ERROR run=$runId server exited before completion"
            break
        }
    }
}

Log "controller started"
foreach ($run in $runs) {
    Start-Run $run
}
Log "controller finished"
