param([string]$PythonExe = '', [int]$Port = 8782)
$ErrorActionPreference = 'Stop'
if (-not $PythonExe) {
    $workspace = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
    $localPython = Join-Path $workspace 'work\.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) { $PythonExe = $localPython }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $PythonExe = 'python' }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { $PythonExe = 'py' }
    else { throw 'Python 3 is required. Pass -PythonExe with its full path.' }
}
& $PythonExe (Join-Path $PSScriptRoot 'view_presentation.py') --port $Port
