param(
    [int[]]$Folds = @(2, 3, 4, 5)
)

$ErrorActionPreference = "Stop"
$controller = Join-Path $PSScriptRoot "run_lab_three_node_fold.ps1"

foreach ($fold in $Folds) {
    Write-Output "$(Get-Date -Format 's') queue_start fold=$fold"
    & $controller `
        -Direction P2_to_P3 `
        -Fold $fold `
        -Rounds 25 `
        -LocalEpochs 3 `
        -Seed 42 `
        -SourceSha "aaa6de1c9b119102bab82e1cac854edadb33956fa56ba9c735a638790ff1abba" `
        -DatasetName "client_data_lab_3gas_5fold_nominal_p2src_v2" `
        -RunLabel "lab3gas_nominal_p2srcnorm_v2" `
        -ResultNamespace "lab_three_gas_p2src_norm_three_node_r25le3_20260730" `
        -PollSeconds 30 `
        -TimeoutHours 8
    Write-Output "$(Get-Date -Format 's') queue_complete fold=$fold"
}

Write-Output "$(Get-Date -Format 's') queue_complete_all folds=$($Folds -join ',')"
