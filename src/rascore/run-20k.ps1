param([string]$OutputName = 'experiment_20000_three_seeds')
$ErrorActionPreference = 'Stop'
$workspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $workspace 'work\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Project environment missing. Run setup.ps1 first.' }
$resultPath = Join-Path $PSScriptRoot "results\$OutputName"
& $pythonExe -u (Join-Path $PSScriptRoot 'experiment_20k.py') --output $resultPath
if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $LASTEXITCODE" }
& $pythonExe (Join-Path $PSScriptRoot 'verify_20k.py') --output $resultPath
if ($LASTEXITCODE -ne 0) { throw 'Result verification failed.' }
& $pythonExe (Join-Path $PSScriptRoot 'report_20k.py') --output $resultPath
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
