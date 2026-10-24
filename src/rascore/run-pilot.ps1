param([int]$SampleSize = 5000, [int]$Seed = 42, [int]$Epochs = 50)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $workspace 'work\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Project environment missing. Run setup.ps1 first.' }
$resultPath = Join-Path $PSScriptRoot "results\pilot_${SampleSize}_seed${Seed}"
$arguments = @('-u', (Join-Path $PSScriptRoot 'pilot.py'), '--sample-size', $SampleSize, '--seed', $Seed, '--epochs', $Epochs, '--output', $resultPath)
if (Test-Path -LiteralPath (Join-Path $resultPath 'pilot_features.npz')) { $arguments += '--reuse-audit' }
& $pythonExe @arguments
if ($LASTEXITCODE -ne 0) { throw "Pilot failed: $LASTEXITCODE" }
