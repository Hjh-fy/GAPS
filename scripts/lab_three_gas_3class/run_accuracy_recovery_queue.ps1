param(
    [string]$SourceSha = "ce45f3edd58b7068a5d47631830afdf0165ca7be81b7ad443b2deabc9788d9ff",
    [int]$Rounds = 25,
    [int]$LocalEpochs = 3,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$controller = Join-Path $PSScriptRoot "run_lab_three_node_fold.ps1"
$queueId = "accuracy_recovery_queue_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$queueRoot = Join-Path $projectRoot "results\lab_3gas_accuracy_recovery_20260730\$queueId"
New-Item -ItemType Directory -Path $queueRoot | Out-Null
$queueLog = Join-Path $queueRoot "queue.log"

function Write-QueueLog([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Output $line
    Add-Content -LiteralPath $queueLog -Value $line
}

$experiments = @(
    @{
        Id = "REC-A1-CB2"
        Dataset = "client_data_lab_3gas_allconc_timepurged_p2src_v1"
        RunLabel = "rec_a1_corrected_b2"
        Namespace = "lab_3gas_accuracy_recovery_corrected_b2_20260730"
        InputDim = 6
        TargetCeWeight = 0.0
    },
    @{
        Id = "REC-A3-COND"
        Dataset = "client_data_lab_3gas_allconc_timepurged_p2src_conductance_v1"
        RunLabel = "rec_a3_conductance"
        Namespace = "lab_3gas_accuracy_recovery_conductance_20260730"
        InputDim = 6
        TargetCeWeight = 0.0
    },
    @{
        Id = "REC-A4-STABLE150"
        Dataset = "client_data_lab_3gas_allconc_timepurged_p2src_stable150_v1"
        RunLabel = "rec_a4_stable150"
        Namespace = "lab_3gas_accuracy_recovery_stable150_20260730"
        InputDim = 6
        TargetCeWeight = 0.0
    },
    @{
        Id = "REC-A5-NOCH2"
        Dataset = "client_data_lab_3gas_allconc_timepurged_p2src_noch2_v1"
        RunLabel = "rec_a5_noch2"
        Namespace = "lab_3gas_accuracy_recovery_noch2_20260730"
        InputDim = 5
        TargetCeWeight = 0.0
    },
    @{
        Id = "REC-A2-TCE"
        Dataset = "client_data_lab_3gas_allconc_timepurged_p2src_v1"
        RunLabel = "rec_a2_targetce"
        Namespace = "lab_3gas_accuracy_recovery_targetce_20260730"
        InputDim = 6
        TargetCeWeight = 1.0
    }
)

foreach ($experiment in $experiments) {
    $common = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $controller,
        "-Direction", "P2_to_P3",
        "-Fold", "1",
        "-Rounds", "$Rounds",
        "-LocalEpochs", "$LocalEpochs",
        "-Seed", "$Seed",
        "-SourceSha", $SourceSha,
        "-DatasetName", $experiment.Dataset,
        "-RunLabel", $experiment.RunLabel,
        "-ResultNamespace", $experiment.Namespace,
        "-Profile", "proto_replay",
        "-DaMode", "corrected_b2",
        "-TargetCeWeight", "$($experiment.TargetCeWeight)",
        "-InputDim", "$($experiment.InputDim)",
        "-SelectionPolicy", "last_round",
        "-PollSeconds", "60",
        "-TimeoutHours", "8"
    )

    Write-QueueLog "preflight_start experiment=$($experiment.Id)"
    & powershell.exe @common -PreflightOnly
    if ($LASTEXITCODE -ne 0) {
        throw "Preflight failed for $($experiment.Id): exit=$LASTEXITCODE"
    }
    Write-QueueLog "preflight_pass experiment=$($experiment.Id)"

    Write-QueueLog "run_start experiment=$($experiment.Id)"
    & powershell.exe @common
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for $($experiment.Id): exit=$LASTEXITCODE"
    }
    Write-QueueLog "run_complete experiment=$($experiment.Id)"
}

Write-QueueLog "queue_complete experiments=$($experiments.Count)"
