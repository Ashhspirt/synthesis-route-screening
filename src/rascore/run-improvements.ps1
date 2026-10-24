param([string]$OutputName = 'improvement_round2', [switch]$AnalyzeOnly)
$ErrorActionPreference = 'Stop'
if ($OutputName -notmatch '^[A-Za-z0-9_-]+$') { throw 'OutputName must be a simple directory name.' }
$workspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $workspace 'work\.venv\Scripts\python.exe'
$resultDir = Join-Path $PSScriptRoot "results\$OutputName"
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Project environment missing. Run setup.ps1 first.' }
if (-not $AnalyzeOnly) {
    & $pythonExe -X utf8 -u (Join-Path $PSScriptRoot 'improvement_experiments.py') --output $resultDir
    if ($LASTEXITCODE -ne 0) { throw 'Training failed. Existing experiments are never overwritten.' }
}
& $pythonExe -X utf8 -u (Join-Path $PSScriptRoot 'analyze_improvements.py') --output $resultDir
if ($LASTEXITCODE -ne 0) { throw 'Verification or analysis failed.' }
