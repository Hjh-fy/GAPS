param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [string]$AttemptId,
    [switch]$Once,
    [int]$IntervalSeconds = 30
)

$ErrorActionPreference = 'Continue'
$attempt = Join-Path $PSScriptRoot "raw/$RunId/$AttemptId"
$bindingPath = Join-Path $attempt 'remote_attempt_binding.json'
if (-not (Test-Path -LiteralPath $bindingPath)) {
    throw "Missing remote binding: $bindingPath"
}
$binding = Get-Content -Raw -LiteralPath $bindingPath | ConvertFrom-Json

do {
    Write-Output "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $AttemptId ====="
    $status = Get-Content -Raw -LiteralPath (Join-Path $attempt 'attempt_status.json') | ConvertFrom-Json
    $status | ConvertTo-Json -Depth 5

    $ecs = [string]$binding.remote_roots.ecs
    $pi = [string]$binding.remote_roots.pi
    $c2 = [string]$binding.remote_roots.ecs_c2
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 root@121.40.139.213 "printf 'completed_rounds='; grep -c fit_round_end '$ecs/raw/server/events.jsonl' 2>/dev/null || true; tail -n 8 '$ecs/raw/server/server.log'"
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 gaps@192.168.137.172 "printf 'PI: '; tail -n 2 '$pi/raw/client_c1/client.log'; stat -c 'resource_bytes=%s' '$pi/raw/client_c1/resource.jsonl' 2>/dev/null || true"
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 root@114.55.171.63 "printf 'ECS-C2: '; tail -n 2 '$c2/raw/client_c2/client.log'; stat -c 'resource_bytes=%s' '$c2/raw/client_c2/resource.jsonl' 2>/dev/null || true"

    if ($status.state -ne 'running') {
        Write-Output "Terminal attempt state reached: $($status.state)"
        break
    }
    if (-not $Once) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while (-not $Once)
