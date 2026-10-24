param([string]$PythonExe = '')
$ErrorActionPreference = 'Stop'
if (-not $PythonExe) {
    $workspace = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
    $PythonExe = Join-Path $workspace 'work\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExe)) { throw 'Use the project training environment, or pass -PythonExe.' }
$runtime = Join-Path $PSScriptRoot '.runtime'
& $PythonExe -m pip install --no-index --no-deps --target $runtime (Join-Path $PSScriptRoot 'vendor\lectrace-1.1.30-py3-none-any.whl')
if ($LASTEXITCODE -ne 0) { throw 'Lectrace installation failed.' }
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $runtime
    $env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.matplotlib-cache'
    & $PythonExe (Join-Path $PSScriptRoot 'build_presentation.py')
    if ($LASTEXITCODE -ne 0) { throw 'Lectrace build failed.' }
} finally { $env:PYTHONPATH = $previousPythonPath }
