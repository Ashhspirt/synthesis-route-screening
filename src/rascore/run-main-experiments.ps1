$ErrorActionPreference = 'Stop'
$workspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $workspace 'work\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Project environment missing. Run setup.ps1 first.' }
& $pythonExe -u (Join-Path $PSScriptRoot 'main_experiments.py')
if ($LASTEXITCODE -ne 0) { throw 'Main experiments failed; inspect saved training logs.' }
& $pythonExe (Join-Path $PSScriptRoot 'report_main_experiments.py')
if ($LASTEXITCODE -ne 0) { throw 'Aggregation or report generation failed.' }
