param(
    [ValidateRange(1, 5)]
    [int]$Fold = 1,
    [ValidateRange(1, 100)]
    [int]$Rounds = 1,
    [ValidateRange(1, 100)]
    [int]$LocalEpochs = 1,
    [ValidateRange(1, 65535)]
    [int]$Port = 18081,
    [ValidateRange(10, 3600)]
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonExe = (Get-Command python -ErrorAction Stop).Source
$dataRoot = Join-Path $projectRoot "dataset\client_data_lab_3gas_5fold_nominal_v1\fold_$Fold"
$outputRoot = Join-Path $projectRoot "results\lab_3gas_flower_smoke_fold$Fold"

if (-not (Test-Path -LiteralPath $dataRoot)) {
    throw "Missing generated fold data: $dataRoot"
}
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$commonShapeArgs = @(
    "--num-classes", "3",
    "--input-dim", "6",
    "--num-clients", "3",
    "--num-phases", "1"
)
$serverArgs = @(
    "-m", "gaps_flower.server_app",
    "--server-address", "127.0.0.1:$Port",
    "--rounds", "$Rounds",
    "--min-clients", "3",
    "--output-dir", $outputRoot,
    "--run-name", "lab_3gas_fold${Fold}_smoke",
    "--strategy", "fedavg",
    "--profile", "smoke"
) + $commonShapeArgs

function ConvertTo-CommandLineArgument {
    param([string]$Value)
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Start-HiddenPython {
    param(
        [string[]]$Arguments,
        [string]$StdoutPath,
        [string]$StderrPath
    )
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $pythonExe
    $startInfo.Arguments = (
        $Arguments |
            ForEach-Object { ConvertTo-CommandLineArgument -Value $_ }
    ) -join " "
    $startInfo.WorkingDirectory = $projectRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Failed to start Python process"
    }
    return [PSCustomObject]@{
        Process = $process
        StdoutTask = $process.StandardOutput.ReadToEndAsync()
        StderrTask = $process.StandardError.ReadToEndAsync()
        StdoutPath = $StdoutPath
        StderrPath = $StderrPath
    }
}

function Save-ProcessLogs {
    param($Handle)
    [System.IO.File]::WriteAllText(
        $Handle.StdoutPath,
        $Handle.StdoutTask.GetAwaiter().GetResult()
    )
    [System.IO.File]::WriteAllText(
        $Handle.StderrPath,
        $Handle.StderrTask.GetAwaiter().GetResult()
    )
}

$processes = @()
try {
    $server = Start-HiddenPython `
        -Arguments $serverArgs `
        -StdoutPath (Join-Path $outputRoot "server.stdout.log") `
        -StderrPath (Join-Path $outputRoot "server.stderr.log")
    $processes += $server
    Start-Sleep -Seconds 3

    foreach ($clientId in 1..3) {
        $clientArgs = @(
            "-m", "gaps_flower.client_app",
            "--server-address", "127.0.0.1:$Port",
            "--client-id", "$clientId",
            "--data-root", $dataRoot,
            "--device", "cpu",
            "--local-epochs", "$LocalEpochs",
            "--batch-size", "64",
            "--profile", "smoke"
        ) + $commonShapeArgs
        $client = Start-HiddenPython `
            -Arguments $clientArgs `
            -StdoutPath (Join-Path $outputRoot "client_$clientId.stdout.log") `
            -StderrPath (Join-Path $outputRoot "client_$clientId.stderr.log")
        $processes += $client
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        foreach ($handle in $processes) {
            $handle.Process.Refresh()
        }
        if (-not ($processes | Where-Object { -not $_.Process.HasExited })) {
            break
        }
        Start-Sleep -Seconds 1
    }

    $running = @($processes | Where-Object { -not $_.Process.HasExited })
    if ($running.Count -gt 0) {
        foreach ($handle in $running) {
            Stop-Process -Id $handle.Process.Id -Force -ErrorAction SilentlyContinue
        }
        throw "Flower smoke test timed out after $TimeoutSeconds seconds"
    }

    foreach ($handle in $processes) {
        Save-ProcessLogs -Handle $handle
    }
    $failed = @($processes | Where-Object { $_.Process.ExitCode -ne 0 })
    if ($failed.Count -gt 0) {
        $details = $failed | ForEach-Object {
            "PID=$($_.Process.Id), exit=$($_.Process.ExitCode)"
        }
        throw "Flower smoke process failure: $($details -join '; ')"
    }

    $checkpoint = Join-Path $outputRoot "server_latest.pth"
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        throw "Flower processes exited successfully but checkpoint is missing: $checkpoint"
    }
    Write-Output "Flower smoke test passed: $checkpoint"
}
finally {
    foreach ($handle in $processes) {
        if (-not $handle.Process.HasExited) {
            Stop-Process -Id $handle.Process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
