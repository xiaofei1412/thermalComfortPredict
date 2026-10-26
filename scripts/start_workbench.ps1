param([ValidateRange(1024, 65535)][int]$Port = 8765)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspaceRoot '.venv-cpu/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Missing .venv-cpu. Install the project using docs/experiments/baseline.md first.'
}
Push-Location -LiteralPath $workspaceRoot
try {
    & $pythonPath -m thermal_comfort.workbench --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Workbench stopped with an error.' }
} finally {
    Pop-Location
}
