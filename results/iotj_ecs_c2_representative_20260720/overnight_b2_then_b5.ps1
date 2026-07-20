$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$python = 'D:\anaconda3\python.exe'
$resultRoot = 'results/iotj_ecs_c2_representative_20260720'
$rawRoot = "$resultRoot/raw"
$b2Attempt = Join-Path $repo "$rawRoot/c12_to_c5__b2__s42/c12_to_c5__b2__s42__a006"
$b2StatusPath = Join-Path $b2Attempt 'attempt_status.json'
$b5Attempt = Join-Path $repo "$rawRoot/c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001"
$logPath = Join-Path $PSScriptRoot 'overnight_b2_then_b5.log'
$controllerPath = Join-Path $repo 'scripts/run_iotj_confirmation_observability.py'
$expectedControllerSha = '46efea0bd55a616101911100d2e90d35bd9704ffdc3979206d27eaf089014ee3'

function Write-WatcherLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $logPath -Value $line
}

function Assert-ControllerIdentity {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $controllerPath).Hash.ToLower()
    if ($actual -ne $expectedControllerSha) {
        throw "controller file SHA-256 changed: $actual"
    }
}

function Write-Utf8JsonExclusive([string]$Path, [object]$Value) {
    $json = $Value | ConvertTo-Json -Depth 8
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        $writer = New-Object System.IO.StreamWriter($stream, $encoding)
        try {
            $writer.Write($json)
        }
        finally {
            $writer.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

Set-Location -LiteralPath $repo
Assert-ControllerIdentity
$lockPath = Join-Path $PSScriptRoot 'overnight_b2_then_b5.lock.json'
$lock = [ordered]@{
    schema_version = 'iotj.overnight_sequence_lock.v1'
    process_id = $PID
    started_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    b2_attempt_id = 'c12_to_c5__b2__s42__a006'
    conditional_b5_attempt_id = 'c12_to_c5__b5__s42__a001'
}
Write-Utf8JsonExclusive -Path $lockPath -Value $lock
Write-WatcherLog 'watcher_start b2=c12_to_c5__b2__s42__a006'
$deadline = (Get-Date).AddHours(24)
while ((Get-Date) -lt $deadline) {
    if (-not (Test-Path -LiteralPath $b2StatusPath)) {
        Start-Sleep -Seconds 30
        continue
    }
    $b2 = Get-Content -Raw -LiteralPath $b2StatusPath | ConvertFrom-Json
    if ($b2.state -eq 'running') {
        Start-Sleep -Seconds 30
        continue
    }
    if ($b2.state -ne 'canonical' -or $b2.event_type -ne 'attempt_end' -or $b2.reason -ne 'validator_accepted') {
        Write-WatcherLog "stop_b2_not_canonical state=$($b2.state) event=$($b2.event_type) reason=$($b2.reason)"
        exit 2
    }
    if ([string]$b2.audit_sha256 -notmatch '^[0-9a-f]{64}$') {
        Write-WatcherLog 'stop_b2_missing_audit_sha256'
        exit 3
    }
    $gate = [ordered]@{
        schema_version = 'iotj.overnight_sequence_gate.v1'
        b2_attempt_id = $b2.attempt_id
        b2_state = $b2.state
        b2_audit_sha256 = $b2.audit_sha256
        b2_status_wall_time_utc = $b2.wall_time_utc
        watcher_gate_time_utc = (Get-Date).ToUniversalTime().ToString('o')
        next_run_id = 'c12_to_c5__b5__s42'
    }
    $gatePath = Join-Path $PSScriptRoot 'b2_a006_to_b5_gate.json'
    Write-Utf8JsonExclusive -Path $gatePath -Value $gate
    Write-WatcherLog "b2_canonical audit=$($b2.audit_sha256)"
    break
}

if ((Get-Date) -ge $deadline) {
    Write-WatcherLog 'stop_b2_wait_timeout'
    exit 4
}

Assert-ControllerIdentity
if (Test-Path -LiteralPath $b5Attempt) {
    throw "refusing existing B5 attempt path: $b5Attempt"
}

$common = @(
    '-m', 'scripts.run_iotj_confirmation_observability',
    '--protocol-manifest', 'results/c2e_summary/confirmation_protocol_manifest.json',
    '--source-archive-manifest', 'results/c2e_summary/source_archive_manifest.json',
    '--dataset-manifest', 'results/c2e_summary/dataset_manifest.json',
    '--command-root', 'results/c2e_commands',
    '--source-archive', 'results/c2e/source/confirmation_source.tar',
    '--raw-root', $rawRoot,
    '--runs', 'B5:42',
    '--ecs-host', 'root@121.40.139.213',
    '--pi-hosts', 'gaps@192.168.137.172',
    '--wait-for-pi-minutes', '30',
    '--pi-retry-seconds', '10',
    '--c2-host', 'root@114.55.171.63',
    '--c2-python', '/root/gaps_c2_cpu_env/bin/python',
    '--c2-data-root', '/root/GAPS/confirmation_c2_data/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid',
    '--c2-dataset-subset-manifest', 'results/c2e_ecs_c2_topology/c2_dataset_subset_manifest.json',
    '--execution-topology-manifest', 'results/c2e_ecs_c2_topology/execution_topology_manifest.json',
    '--run-timeout-seconds', '172800',
    '--poll-seconds', '30'
)

Write-WatcherLog 'b5_preflight_start'
& $python @common '--preflight-only' 1> (Join-Path $PSScriptRoot 'b5_s42_a001_preflight.stdout.log') 2> (Join-Path $PSScriptRoot 'b5_s42_a001_preflight.stderr.log')
if ($LASTEXITCODE -ne 0) {
    Write-WatcherLog "stop_b5_preflight_exit=$LASTEXITCODE"
    exit 5
}
Write-WatcherLog 'b5_preflight_passed controller_start'

Assert-ControllerIdentity
& $python @common 1> (Join-Path $PSScriptRoot 'b5_s42_a001_controller.stdout.log') 2> (Join-Path $PSScriptRoot 'b5_s42_a001_controller.stderr.log')
$controllerExit = $LASTEXITCODE
if (-not (Test-Path -LiteralPath (Join-Path $b5Attempt 'attempt_status.json'))) {
    Write-WatcherLog "stop_b5_missing_status controller_exit=$controllerExit"
    exit 6
}
$b5 = Get-Content -Raw -LiteralPath (Join-Path $b5Attempt 'attempt_status.json') | ConvertFrom-Json
Write-WatcherLog "b5_terminal controller_exit=$controllerExit state=$($b5.state) reason=$($b5.reason) audit=$($b5.audit_sha256)"
if ($controllerExit -ne 0 -or $b5.state -ne 'canonical') {
    exit 7
}
exit 0
