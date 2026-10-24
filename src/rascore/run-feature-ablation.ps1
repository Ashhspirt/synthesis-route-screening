param([string]$OutputName = 'error_analysis_and_feature_ablation', [switch]$AnalyzeOnly)
$ErrorActionPreference = 'Stop'
if ($OutputName -notmatch '^[A-Za-z0-9_-]+$') { throw 'OutputName must be a simple directory name.' }
$workspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $workspace 'work\.venv\Scripts\python.exe'
$resultDir = Join-Path $PSScriptRoot "results\$OutputName"
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Project environment missing. Run setup.ps1 first.' }
if (-not $AnalyzeOnly) {
    & $pythonExe -u (Join-Path $PSScriptRoot 'feature_ablation.py') --output $resultDir
    if ($LASTEXITCODE -ne 0) { throw 'Ablation training failed. Existing results are never overwritten.' }
}
& $pythonExe -u (Join-Path $PSScriptRoot 'analyze_feature_ablation.py') --output $resultDir
if ($LASTEXITCODE -ne 0) { throw 'Verification or analysis failed.' }
