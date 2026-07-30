param(
    [ValidateSet("P12_to_P3", "P2_to_P3")]
    [string]$Direction = "P12_to_P3",
    [ValidateRange(1, 5)]
    [int]$Fold = 1,
    [ValidateRange(1, 100)]
    [int]$Rounds = 25,
    [ValidateRange(1, 20)]
    [int]$LocalEpochs = 3,
    [int]$Seed = 42,
    [ValidateSet("strong_cls", "proto_replay")]
    [string]$Profile = "strong_cls",
    [ValidateSet("legacy_strong", "corrected_b2")]
    [string]$DaMode = "legacy_strong",
    [ValidateRange(0.0, 100.0)]
    [double]$TargetCeWeight = 0.0,
    [ValidateRange(1, 18)]
    [int]$InputDim = 6,
    [ValidateSet("last_round", "source_calibration")]
    [string]$SelectionPolicy = "last_round",
    [string]$SourceSha = "aaa6de1c9b119102bab82e1cac854edadb33956fa56ba9c735a638790ff1abba",
    [string]$ServerHost = "root@121.40.139.213",
    [string]$CloudBHost = "root@114.55.171.63",
    [string]$PiHost = "gaps@192.168.137.172",
    [string]$DatasetName = "client_data_lab_3gas_5fold_nominal_v1",
    [string]$RunLabel = "lab3gas_nominal",
    [string]$ResultNamespace = "lab_three_gas_nominal_three_node_r25le3_20260729",
    [int]$PollSeconds = 60,
    [ValidateRange(5, 120)]
    [int]$StallMinutes = 15,
    [int]$TimeoutHours = 8,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$sourceClients = if ($Direction -eq "P12_to_P3") { @(1, 2) } else { @(2) }
$sourceCsv = $sourceClients -join ","
$runId = "${RunLabel}_${Direction}_fold${Fold}_s${Seed}_r${Rounds}le${LocalEpochs}"
$controllerInstanceId = if ($PreflightOnly) {
    "${runId}__preflight_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
} else {
    $runId
}
$serverRuntime = "/root/GAPS/lab_3gas_confirmation_runtime/$SourceSha/src"
$cloudBRuntime = "/root/GAPS/confirmation_runtime_c2/$SourceSha/src"
$piRuntime = "/home/gaps/GAPS/confirmation_runtime/$SourceSha/src"
$serverData = "/root/GAPS/dataset/$DatasetName/fold_$Fold"
$cloudBData = "/root/GAPS/lab_3gas_data/$DatasetName/fold_$Fold"
$piData = "/home/gaps/GAPS/lab_3gas_data/$DatasetName/fold_$Fold"
$serverResultsRoot = "/root/GAPS/results/$ResultNamespace"
$serverRunDir = "$serverResultsRoot/$runId"
$cloudBLogRoot = "/root/GAPS/results/${ResultNamespace}_client_logs"
$piLogRoot = "/home/gaps/GAPS/results/${ResultNamespace}_client_logs"
$localControllerRoot = Join-Path $projectRoot "results\${ResultNamespace}_controller"
$localRunDir = Join-Path $localControllerRoot $controllerInstanceId

if (Test-Path -LiteralPath $localRunDir) {
    throw "Refusing to overwrite local controller output: $localRunDir"
}
New-Item -ItemType Directory -Path $localRunDir -Force | Out-Null
$controllerLog = Join-Path $localRunDir "controller.log"

function Write-RunLog([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Output $line
    Add-Content -LiteralPath $controllerLog -Value $line
}

function Invoke-Remote(
    [string]$HostSpec,
    [string]$Command,
    [int]$Attempts = 3
) {
    $lastOutput = @()
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        $savedErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $output = & ssh -n `
                -o BatchMode=yes `
                -o ConnectTimeout=20 `
                -o ServerAliveInterval=15 `
                -o ServerAliveCountMax=2 `
                $HostSpec $Command 2>&1
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $savedErrorActionPreference
        }
        $lastOutput = @($output)
        if ($exitCode -eq 0) {
            return $lastOutput
        }
        Write-RunLog "WARN ssh host=$HostSpec attempt=$attempt/$Attempts exit=$exitCode"
        Start-Sleep -Seconds ([Math]::Min(10, 2 * $attempt))
    }
    throw "SSH failed host=$HostSpec command=$Command output=$($lastOutput -join ' ')"
}

function Start-Tunnel(
    [string]$Name,
    [string[]]$Arguments
) {
    $stdout = Join-Path $localRunDir "$Name.stdout.log"
    $stderr = Join-Path $localRunDir "$Name.stderr.log"
    $process = Start-Process `
        -FilePath "ssh" `
        -ArgumentList $Arguments `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    Start-Sleep -Seconds 2
    $process.Refresh()
    if ($process.HasExited) {
        throw "SSH tunnel $Name exited early; see $stderr"
    }
    Write-RunLog "tunnel_started name=$Name pid=$($process.Id)" | Out-Host
    return $process
}

function Current-RunServer {
    return Invoke-Remote $ServerHost `
        "pgrep -af '[g]aps_flower.server_app.*$runId' || true"
}

function Current-RunClient([string]$HostSpec, [int]$ClientId) {
    return Invoke-Remote $HostSpec `
        "pgrep -af '[g]aps_flower.client_app.*--client-id $ClientId.*--run-tag $runId' || true"
}

function Stop-CurrentRun {
    try {
        Invoke-Remote $ServerHost `
            "pkill -f '[g]aps_flower.server_app.*$runId' || true" | Out-Null
    } catch {
        Write-RunLog "WARN failed to stop current server: $($_.Exception.Message)"
    }
    try {
        Invoke-Remote $CloudBHost `
            "pkill -f '[g]aps_flower.client_app.*--client-id 2.*--run-tag $runId' || true" | Out-Null
    } catch {
        Write-RunLog "WARN failed to stop C2: $($_.Exception.Message)"
    }
    if ($sourceClients -contains 1) {
        try {
            Invoke-Remote $PiHost `
                "pkill -f '[g]aps_flower.client_app.*--client-id 1.*--run-tag $runId' || true" | Out-Null
        } catch {
            Write-RunLog "WARN failed to stop C1: $($_.Exception.Message)"
        }
    }
}

$tunnels = @()
try {
    $destinationState = Invoke-Remote $ServerHost `
        "if test -e '$serverRunDir'; then echo EXISTS; else echo FREE; fi"
    if (($destinationState -join "`n") -notmatch "FREE") {
        throw "Refusing to overwrite remote run directory: $serverRunDir"
    }

    foreach ($probe in @(
        @{ Host = $ServerHost; Command = "pgrep -af '[g]aps_flower.server_app|[g]aps_flower.client_app' || true"; Role = "server" },
        @{ Host = $CloudBHost; Command = "pgrep -af '[g]aps_flower.server_app|[g]aps_flower.client_app' || true"; Role = "C2" },
        @{ Host = $PiHost; Command = "pgrep -af '[g]aps_flower.server_app|[g]aps_flower.client_app' || true"; Role = "C1" }
    )) {
        $active = Invoke-Remote $probe.Host $probe.Command
        if ($active) {
            throw "Residual Flower process on $($probe.Role): $($active -join ' ')"
        }
    }

    $runtimeProbes = @(
        @{
            Host = $ServerHost
            Role = "server"
            Command = "cd '$serverRuntime' && /root/gaps_env/bin/python scripts/lab_three_gas_3class/remote_runtime_preflight.py --runtime-src '$serverRuntime' --role server --profile '$Profile' --input-dim '$InputDim'"
        },
        @{
            Host = $ServerHost
            Role = "P3/target-data"
            Command = "cd '$serverRuntime' && /root/gaps_env/bin/python scripts/lab_three_gas_3class/remote_runtime_preflight.py --runtime-src '$serverRuntime' --role client --data-root '$serverData' --client-id 3 --profile '$Profile' --input-dim '$InputDim'"
        },
        @{
            Host = $CloudBHost
            Role = "C2/P2"
            Command = "cd '$cloudBRuntime' && /root/gaps_c2_cpu_env/bin/python scripts/lab_three_gas_3class/remote_runtime_preflight.py --runtime-src '$cloudBRuntime' --role client --data-root '$cloudBData' --client-id 2 --profile '$Profile' --input-dim '$InputDim'"
        }
    )
    if ($sourceClients -contains 1) {
        $runtimeProbes += @{
            Host = $PiHost
            Role = "C1/P1"
            Command = "cd '$piRuntime' && /home/gaps/GAPS/gaps_rpi_env/bin/python scripts/lab_three_gas_3class/remote_runtime_preflight.py --runtime-src '$piRuntime' --role client --data-root '$piData' --client-id 1 --profile '$Profile' --input-dim '$InputDim'"
        }
    }
    foreach ($probe in $runtimeProbes) {
        $report = Invoke-Remote $probe.Host $probe.Command
        Write-RunLog "runtime_preflight role=$($probe.Role) report=$($report -join ' ')"
    }
    if ($PreflightOnly) {
        Write-RunLog "preflight_passed run=$runId source_sha=$SourceSha"
        return
    }

    $common = @(
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-N"
    )
    $tunnels += Start-Tunnel "controller_to_server" (
        $common + @("-L", "127.0.0.1:18080:127.0.0.1:8080", $ServerHost)
    )
    $tunnels += Start-Tunnel "controller_to_cloud_b" (
        $common + @("-R", "127.0.0.1:18080:127.0.0.1:18080", $CloudBHost)
    )
    if ($sourceClients -contains 1) {
        $tunnels += Start-Tunnel "controller_to_pi" (
            $common + @("-R", "127.0.0.1:18080:127.0.0.1:18080", $PiHost)
        )
    }

    $serverLaunch = @(
        "/root/gaps_env/bin/python '$serverRuntime/scripts/remote_launch_flower_server_clean.py'",
        "--project '$serverRuntime'",
        "--results-root '$serverResultsRoot'",
        "--run-id '$runId'",
        "--data-root '$serverData'",
        "--source-clients '$sourceCsv'",
        "--target-clients '3'",
        "--rounds '$Rounds'",
        "--profile '$Profile'",
        "--da-mode '$DaMode'",
        "--target-ce-weight '$TargetCeWeight'",
        "--num-classes '3'",
        "--input-dim '$InputDim'",
        "--num-clients '3'",
        "--num-phases '1'",
        "--server-address '127.0.0.1:8080'",
        "--python-bin '/root/gaps_env/bin/python'"
    ) -join " "
    $serverPid = Invoke-Remote $ServerHost $serverLaunch
    Write-RunLog "server_started pid=$($serverPid -join '') run=$runId"
    Start-Sleep -Seconds 8

    if ($sourceClients -contains 1) {
        $piLaunch = @(
            "/home/gaps/GAPS/gaps_rpi_env/bin/python '$piRuntime/scripts/remote_launch_flower_client_clean.py'",
            "--project '$piRuntime'",
            "--log-root '$piLogRoot'",
            "--run-id '$runId'",
            "--data-root '$piData'",
            "--client-id '1'",
            "--profile '$Profile'",
            "--local-epochs '$LocalEpochs'",
            "--batch-size '32'",
            "--num-classes '3'",
            "--input-dim '$InputDim'",
            "--num-clients '3'",
            "--num-phases '1'",
            "--eval-split 'calibration'",
            "--python-bin '/home/gaps/GAPS/gaps_rpi_env/bin/python'",
            "--server-address '127.0.0.1:18080'"
        ) -join " "
        $piPid = Invoke-Remote $PiHost $piLaunch
        Write-RunLog "C1_started pid=$($piPid -join '')"
    }

    $cloudBLaunch = @(
        "/root/gaps_c2_cpu_env/bin/python '$cloudBRuntime/scripts/remote_launch_flower_client_clean.py'",
        "--project '$cloudBRuntime'",
        "--log-root '$cloudBLogRoot'",
        "--run-id '$runId'",
        "--data-root '$cloudBData'",
        "--client-id '2'",
        "--profile '$Profile'",
        "--local-epochs '$LocalEpochs'",
        "--batch-size '32'",
        "--num-classes '3'",
        "--input-dim '$InputDim'",
        "--num-clients '3'",
        "--num-phases '1'",
        "--eval-split 'calibration'",
        "--python-bin '/root/gaps_c2_cpu_env/bin/python'",
        "--server-address '127.0.0.1:18080'"
    ) -join " "
    $cloudBPid = Invoke-Remote $CloudBHost $cloudBLaunch
    Write-RunLog "C2_started pid=$($cloudBPid -join '')"

    $deadline = (Get-Date).AddHours($TimeoutHours)
    $lastObservedRound = 0
    $lastRoundProgressAt = Get-Date
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds $PollSeconds
        $roundCount = Invoke-Remote $ServerHost `
            "find '$serverRunDir' -maxdepth 1 -name 'server_round_*.pth' ! -name '*adapted*' -type f 2>/dev/null | wc -l"
        $roundNumber = [int](($roundCount -join "").Trim())
        if ($roundNumber -gt $lastObservedRound) {
            $lastObservedRound = $roundNumber
            $lastRoundProgressAt = Get-Date
        }
        $serverActive = Current-RunServer
        $lastRound = Invoke-Remote $ServerHost `
            "ls '$serverRunDir'/server_round_*.pth 2>/dev/null | grep -v adapted | tail -1 || true"
        $health = if ($sourceClients -contains 1) {
            Invoke-Remote $PiHost `
                "vcgencmd measure_temp 2>/dev/null; vcgencmd get_throttled 2>/dev/null"
        } else {
            @("not_participating")
        }
        Write-RunLog (
            "progress rounds=$($roundCount -join '')/$Rounds " +
            "server_running=$([bool]$serverActive) last=$($lastRound -join '') " +
            "pi=$($health -join ' ')"
        )
        $failureState = Invoke-Remote $ServerHost `
            "if test -f '$serverRunDir/history.json' && grep -Eq '\""(fit|evaluate)_failures\"": [1-9]' '$serverRunDir/history.json'; then echo FAILURE; else echo OK; fi"
        if (($failureState -join "") -match "FAILURE") {
            throw "Flower history recorded a fit/evaluate failure"
        }
        if (
            $serverActive -and
            ((Get-Date) - $lastRoundProgressAt).TotalMinutes -ge $StallMinutes
        ) {
            throw (
                "No new base checkpoint for $StallMinutes minutes; " +
                "last_round=$lastObservedRound"
            )
        }
        if ($roundNumber -ge $Rounds -and -not $serverActive) {
            break
        }
        if (-not $serverActive) {
            throw "Server exited before completing $Rounds rounds"
        }
    }

    $finalRounds = Invoke-Remote $ServerHost `
        "find '$serverRunDir' -maxdepth 1 -name 'server_round_*.pth' ! -name '*adapted*' -type f | wc -l"
    if ([int](($finalRounds -join "").Trim()) -ne $Rounds) {
        throw "Run did not complete exactly $Rounds rounds"
    }

    $evaluationCommand = @(
        "cd '$serverRuntime'",
        "&& /root/gaps_env/bin/python scripts/lab_three_gas_3class/evaluate_source_target_run.py",
        "--run-dir '$serverRunDir'",
        "--data-root '$serverData'",
        "--source-clients '$sourceCsv'",
        "--target-client '3'",
        "--rounds '$Rounds'",
        "--selection-policy '$SelectionPolicy'",
        "--device 'cpu'",
        "> '$serverRunDir/evaluation.stdout.log'",
        "2> '$serverRunDir/evaluation.stderr.log'"
    ) -join " "
    Invoke-Remote $ServerHost $evaluationCommand | Out-Null

    $auditCommand = @(
        "cd '$serverRuntime'",
        "&& /root/gaps_env/bin/python scripts/lab_three_gas_3class/validate_three_node_run.py",
        "--run-dir '$serverRunDir'",
        "--direction '$Direction'",
        "--rounds '$Rounds'",
        "--local-epochs '$LocalEpochs'",
        "--da-steps '100'",
        "--input-dim '$InputDim'",
        "--profile '$Profile'",
        "--da-mode '$DaMode'",
        "--target-ce-weight '$TargetCeWeight'",
        "--selection-policy '$SelectionPolicy'",
        "--target-data-dir '$serverData/client_3'",
        "--evaluation-dir '$serverRunDir/formal_evaluation'",
        "--output '$serverRunDir/postflight_attempt_audit.json'"
    ) -join " "
    Invoke-Remote $ServerHost $auditCommand | Out-Null

    $localSummary = Join-Path $localRunDir "formal_evaluation_summary.json"
    & scp -q `
        "${ServerHost}:$serverRunDir/formal_evaluation/summary.json" `
        $localSummary
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to recover formal evaluation summary"
    }
    $localAudit = Join-Path $localRunDir "postflight_attempt_audit.json"
    & scp -q `
        "${ServerHost}:$serverRunDir/postflight_attempt_audit.json" `
        $localAudit
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to recover postflight attempt audit"
    }
    Write-RunLog "completed run=$runId remote=$serverRunDir summary=$localSummary"
}
catch {
    Write-RunLog "FAILED run=$runId error=$($_.Exception.Message)"
    Stop-CurrentRun
    throw
}
finally {
    foreach ($tunnel in $tunnels) {
        if ($tunnel -is [System.Diagnostics.Process]) {
            $tunnel.Refresh()
            if (-not $tunnel.HasExited) {
                Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
