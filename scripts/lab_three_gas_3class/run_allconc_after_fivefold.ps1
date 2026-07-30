param(
    [string]$QueuePidFile = (
        "results/lab_3gas_p2src_norm_three_node_20260730/" +
        "remaining_folds_launcher/queue.pid"
    ),
    [string]$QueueStdout = (
        "results/lab_3gas_p2src_norm_three_node_20260730/" +
        "remaining_folds_launcher/queue.stdout.log"
    ),
    [string]$QueueStderr = (
        "results/lab_3gas_p2src_norm_three_node_20260730/" +
        "remaining_folds_launcher/queue.stderr.log"
    )
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pidPath = Join-Path $projectRoot $QueuePidFile
$stdoutPath = Join-Path $projectRoot $QueueStdout
$stderrPath = Join-Path $projectRoot $QueueStderr
$queuePid = [int](Get-Content -LiteralPath $pidPath -Raw).Trim()

Write-Output "$(Get-Date -Format 's') waiting_for_fivefold pid=$queuePid"
while (Get-Process -Id $queuePid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$queueOutput = Get-Content -LiteralPath $stdoutPath -Raw
$queueError = Get-Content -LiteralPath $stderrPath -Raw
if ($queueOutput -notmatch "queue_complete_all folds=2,3,4,5") {
    throw "Five-fold queue did not complete successfully; all-concentration run blocked"
}
if (-not [string]::IsNullOrWhiteSpace($queueError)) {
    throw "Five-fold queue stderr is non-empty; all-concentration run blocked"
}

Write-Output "$(Get-Date -Format 's') fivefold_gate_passed"
$controller = Join-Path $PSScriptRoot "run_lab_three_node_fold.ps1"
& $controller `
    -Direction P2_to_P3 `
    -Fold 1 `
    -Rounds 25 `
    -LocalEpochs 3 `
    -Seed 42 `
    -SourceSha "aaa6de1c9b119102bab82e1cac854edadb33956fa56ba9c735a638790ff1abba" `
    -DatasetName "client_data_lab_3gas_allconc_timepurged_p2src_v1" `
    -RunLabel "lab3gas_allconc_timepurged_v1" `
    -ResultNamespace "lab_three_gas_allconc_timepurged_p2src_r25le3_20260730" `
    -PollSeconds 30 `
    -TimeoutHours 8
Write-Output "$(Get-Date -Format 's') allconc_run_complete"
