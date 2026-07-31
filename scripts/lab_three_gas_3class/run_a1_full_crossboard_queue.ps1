param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{64}$")]
    [string]$SourceSha,
    [int]$Rounds = 25,
    [int]$LocalEpochs = 1,
    [int]$Seed = 42,
    [string]$ResultNamespace = "lab_3gas_a1_full_crossboard_seed42_20260731",
    [switch]$ContractOnly
)

$ErrorActionPreference = "Stop"
$serverDaSteps = 100
$selectionPolicy = "last_round"
$experiments = @(
    [ordered]@{
        experiment_id = "A1-FULL-E1-P2P3-S42"
        protocol = "A1"
        direction = "P2_to_P3"
        dataset = "client_data_lab_3gas_a1_full_crossboard_p2p3_v1"
        run_label = "a1_full_p2p3"
    },
    [ordered]@{
        experiment_id = "A4-CTRL-E2-P2P3-LE1-S42"
        protocol = "A4"
        direction = "P2_to_P3"
        dataset = "client_data_lab_3gas_a4_crossboard_p2p3_eval_v1"
        run_label = "a4_ctrl_le1_p2p3"
    },
    [ordered]@{
        experiment_id = "A1-FULL-E3-P1P3-S42"
        protocol = "A1"
        direction = "P1_to_P3"
        dataset = "client_data_lab_3gas_a1_full_crossboard_p1p3_v1"
        run_label = "a1_full_p1p3"
    },
    [ordered]@{
        experiment_id = "A1-FULL-E4-P12P3-S42"
        protocol = "A1"
        direction = "P12_to_P3"
        dataset = "client_data_lab_3gas_a1_full_crossboard_p12p3_v1"
        run_label = "a1_full_p12p3"
    },
    [ordered]@{
        experiment_id = "A1-FULL-E5-P2P1-S42"
        protocol = "A1"
        direction = "P2_to_P1"
        dataset = "client_data_lab_3gas_a1_full_crossboard_p2p1_v1"
        run_label = "a1_full_p2p1"
    },
    [ordered]@{
        experiment_id = "A1-FULL-E6-P3P1-S42"
        protocol = "A1"
        direction = "P3_to_P1"
        dataset = "client_data_lab_3gas_a1_full_crossboard_p3p1_v1"
        run_label = "a1_full_p3p1"
    }
)

if ($ContractOnly) {
    [ordered]@{
        source_sha = $SourceSha
        rounds = $Rounds
        local_epochs = $LocalEpochs
        server_da_steps = $serverDaSteps
        seed = $Seed
        selection_policy = $selectionPolicy
        experiments = $experiments
    } | ConvertTo-Json -Depth 5 -Compress
    return
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$controller = Join-Path $PSScriptRoot "run_lab_three_node_fold.ps1"
$queueId = "a1_full_crossboard_queue_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$queueRoot = Join-Path $projectRoot "results\${ResultNamespace}_controller\$queueId"
if (Test-Path -LiteralPath $queueRoot) {
    throw "Refusing to overwrite queue evidence: $queueRoot"
}
New-Item -ItemType Directory -Path $queueRoot | Out-Null
$queueLog = Join-Path $queueRoot "queue.log"

function Write-QueueLog([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Output $line
    Add-Content -LiteralPath $queueLog -Value $line
}

$contractPath = Join-Path $queueRoot "queue_contract.json"
[ordered]@{
    source_sha = $SourceSha
    rounds = $Rounds
    local_epochs = $LocalEpochs
    server_da_steps = $serverDaSteps
    seed = $Seed
    selection_policy = $selectionPolicy
    experiments = $experiments
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $contractPath -Encoding UTF8

Write-QueueLog "queue_start id=$queueId experiments=$($experiments.Count) source_sha=$SourceSha"
foreach ($experiment in $experiments) {
    $common = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $controller,
        "-Direction", $experiment.direction,
        "-Fold", "1",
        "-Rounds", "$Rounds",
        "-LocalEpochs", "$LocalEpochs",
        "-Seed", "$Seed",
        "-SourceSha", $SourceSha,
        "-DatasetName", $experiment.dataset,
        "-RunLabel", $experiment.run_label,
        "-ResultNamespace", $ResultNamespace,
        "-Profile", "proto_replay",
        "-DaMode", "corrected_b2",
        "-TargetCeWeight", "0",
        "-InputDim", "6",
        "-SelectionPolicy", $selectionPolicy,
        "-PollSeconds", "60",
        "-TimeoutHours", "8"
    )

    Write-QueueLog "preflight_start experiment=$($experiment.experiment_id) direction=$($experiment.direction)"
    & powershell.exe @common -PreflightOnly
    if ($LASTEXITCODE -ne 0) {
        throw "Preflight failed: experiment=$($experiment.experiment_id) exit=$LASTEXITCODE"
    }
    Write-QueueLog "preflight_pass experiment=$($experiment.experiment_id)"

    Write-QueueLog "run_start experiment=$($experiment.experiment_id)"
    & powershell.exe @common
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: experiment=$($experiment.experiment_id) exit=$LASTEXITCODE"
    }
    Write-QueueLog "run_complete experiment=$($experiment.experiment_id)"
}
Write-QueueLog "queue_complete experiments=$($experiments.Count)"
