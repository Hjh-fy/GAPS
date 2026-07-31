param(
    [string]$SourceSha = "e104a05e9e341305cd04e6270edae768087e5260aa691a037a3eed45ed0bc38d",
    [int]$Rounds = 25,
    [int]$LocalEpochs = 3,
    [int]$Seed = 42,
    [string]$ResultNamespace = "lab_3gas_a4_crossboard_seed42_20260731"
)

$ErrorActionPreference = "Stop"
$controller = Join-Path $PSScriptRoot "run_lab_three_node_fold.ps1"
$experiments = @(
    @{
        Direction = "P2_to_P1"
        Dataset = "client_data_lab_3gas_a4_crossboard_p2p1_v1"
        Label = "a4_cross_p2p1"
    },
    @{
        Direction = "P1_to_P3"
        Dataset = "client_data_lab_3gas_a4_crossboard_p1p3_v1"
        Label = "a4_cross_p1p3"
    },
    @{
        Direction = "P12_to_P3"
        Dataset = "client_data_lab_3gas_a4_crossboard_p12p3_v1"
        Label = "a4_cross_p12p3"
    }
)

foreach ($experiment in $experiments) {
    Write-Output "$(Get-Date -Format 's') START $($experiment.Direction)"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $controller `
        -Direction $experiment.Direction `
        -Fold 1 `
        -Rounds $Rounds `
        -LocalEpochs $LocalEpochs `
        -Seed $Seed `
        -SourceSha $SourceSha `
        -DatasetName $experiment.Dataset `
        -RunLabel $experiment.Label `
        -ResultNamespace $ResultNamespace `
        -Profile "proto_replay" `
        -DaMode "corrected_b2" `
        -TargetCeWeight 0 `
        -InputDim 6 `
        -SelectionPolicy "last_round" `
        -PollSeconds 60 `
        -TimeoutHours 8
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $($experiment.Direction)"
    }
    Write-Output "$(Get-Date -Format 's') COMPLETE $($experiment.Direction)"
}
