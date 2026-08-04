$ErrorActionPreference = 'Stop'
$worktree = 'D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-final-classification-le1'
$resultRoot = Join-Path $worktree 'results\iotj_final_classification_le1_20260804'
$queuePid = 18872
Set-Location -LiteralPath $worktree

while ($true) {
    $markers = @(Get-ChildItem -LiteralPath $resultRoot -Filter 'fixed_endpoint_complete.json' -Recurse -File -ErrorAction SilentlyContinue)
    if ($markers.Count -eq 21) {
        Write-Output "[$(Get-Date -Format o)] ENDPOINTS_COMPLETE count=21"
        break
    }
    if (-not (Get-Process -Id $queuePid -ErrorAction SilentlyContinue)) {
        throw "Frozen queue exited before all endpoints completed (count=$($markers.Count))"
    }
    Write-Output "[$(Get-Date -Format o)] WAIT endpoint_count=$($markers.Count)"
    Start-Sleep -Seconds 60
}

& python -m scripts.finalize_iotj_final_classification_le1 --finalize
if ($LASTEXITCODE -ne 0) { throw "Final evaluation failed" }
& python -m scripts.audit_iotj_final_classification_le1 --stage post-run --strict
if ($LASTEXITCODE -ne 0) { throw "Post-run strict audit failed" }
& python -m compileall -q gaps_flower scripts tests
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }
& python -m pytest tests/test_scaffold_canonical.py tests/test_canonical_uda_and_target_gate.py tests/test_ablation_loss_activity.py tests/test_selective_warmup_boundary.py tests/test_iotj_final_classification_evaluation.py tests/test_iotj_final_classification_runner.py -q --basetemp=.tmp_pytest_postrun_publish
if ($LASTEXITCODE -ne 0) { throw "Post-run tests failed" }

& git add scripts/finalize_iotj_final_classification_le1.py tests/test_iotj_final_classification_evaluation.py
if ($LASTEXITCODE -ne 0) { throw "git add code failed" }
$pathspecs = @(
    ':(glob)results/iotj_final_classification_le1_20260804/**/*.json',
    ':(glob)results/iotj_final_classification_le1_20260804/**/*.jsonl',
    ':(glob)results/iotj_final_classification_le1_20260804/**/*.csv',
    ':(glob)results/iotj_final_classification_le1_20260804/**/*.md',
    ':(glob)results/iotj_final_classification_le1_20260804/**/*.png',
    ':(glob)results/iotj_final_classification_le1_20260804/**/*.pdf',
    ':(glob)results/iotj_final_classification_le1_20260804/**/*.log',
    ':(glob)results/iotj_final_classification_le1_20260804/**/*.ps1',
    ':(glob)results/iotj_final_classification_le1_20260804/*/adapted_step_100.pth',
    ':(glob)results/iotj_final_classification_le1_20260804/*/remote_server/server_latest.pth',
    ':(glob)results/iotj_final_classification_le1_20260804/*/remote_server/server_latest_adapted.pth'
)
& git add -f -- $pathspecs
if ($LASTEXITCODE -ne 0) { throw "git add result evidence failed" }
& git commit -m 'results: complete IoT-J final classification seed42 suite'
if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
& git push origin codex/iotj-final-classification-le1
if ($LASTEXITCODE -ne 0) { throw "git push failed" }
Write-Output "[$(Get-Date -Format o)] FINALIZE_AUDIT_COMMIT_PUSH_COMPLETE"
