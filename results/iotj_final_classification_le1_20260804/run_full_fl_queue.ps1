$ErrorActionPreference = 'Stop'
$worktree = 'D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-final-classification-le1'
$experimentIds = @(
    'FCL-E1-FEDPROX',
    'FCL-E1-SCAFFOLD',
    'FCL-E3-GAPS-C3',
    'FCL-E3-GAPS-C4',
    'FCL-E3-GAPS-C5',
    'FCL-E4-A1',
    'FCL-E4-A2',
    'FCL-E4-A3',
    'FCL-E4-A4',
    'FCL-E4-A5'
)
Set-Location -LiteralPath $worktree
foreach ($experimentId in $experimentIds) {
    Write-Output "[$(Get-Date -Format o)] START $experimentId"
    & python -m scripts.run_iotj_final_classification_le1 run --experiment-id $experimentId --timeout-hours 10
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen experiment failed: $experimentId (exit=$LASTEXITCODE)"
    }
    Write-Output "[$(Get-Date -Format o)] COMPLETE $experimentId"
}
Write-Output "[$(Get-Date -Format o)] QUEUE_COMPLETE"
