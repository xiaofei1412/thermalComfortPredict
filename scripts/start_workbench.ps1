param(
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [string]$PythonPath
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$localUrl = "http://127.0.0.1:$Port"
try { $existing = Invoke-RestMethod -Uri "$localUrl/api/health" -TimeoutSec 2 } catch { $existing = $null }
if ($existing -and $existing.service -eq 'thermal-workbench') {
    if ([System.IO.Path]::GetFullPath($existing.workspace) -ne [System.IO.Path]::GetFullPath($workspaceRoot)) {
        throw "Port $Port belongs to another workspace. Use -Port or close that service yourself."
    }
    Write-Output "Workbench already running: $localUrl"
    return
}
if (-not $PythonPath) {
    foreach ($candidate in @('.venv-cpu/Scripts/python.exe', '.venv/Scripts/python.exe')) {
        $candidatePath = Join-Path $workspaceRoot $candidate
        if (Test-Path -LiteralPath $candidatePath) { $PythonPath = $candidatePath; break }
    }
}
if (-not $PythonPath -or -not (Test-Path -LiteralPath $PythonPath)) {
    throw 'No Python environment. Create .venv as documented in docs/workbench.md, or pass -PythonPath.'
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
Push-Location -LiteralPath $workspaceRoot
try {
    & $PythonPath -c 'import numpy, pandas; import thermal_comfort.workbench'
    if ($LASTEXITCODE -ne 0) { throw 'Missing workbench dependencies. Install the project with this Python as documented.' }
    & $PythonPath -m thermal_comfort.workbench --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Workbench stopped with an error.' }
} finally {
    Pop-Location
}
