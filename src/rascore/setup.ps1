param([string]$PythonExe = 'python')
$ErrorActionPreference = 'Stop'
$workspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$venv = Join-Path $workspace 'work\.venv'
$projectPython = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    & $PythonExe -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment.' }
}
& $projectPython -m pip install --only-binary=:all: -r (Join-Path $PSScriptRoot 'requirements-lock.txt') --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
