@echo off
setlocal
cd /d "D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-confirmation-observability"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "results\iotj_ecs_c2_representative_20260720\overnight_b2_then_b5.ps1" 1>"results\iotj_ecs_c2_representative_20260720\overnight_watcher.stdout.log" 2>"results\iotj_ecs_c2_representative_20260720\overnight_watcher.stderr.log"
exit /b %errorlevel%
