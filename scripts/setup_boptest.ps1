param(
    [ValidateSet('up', 'down', 'status')]
    [string]$Action = 'up',
    [string]$Distribution = 'Ubuntu-24.04',
    [ValidatePattern('^[a-z0-9][a-z0-9-]{0,62}$')][string]$ProjectName = 'thermal-boptest',
    [ValidateRange(1024, 65535)][int]$Port = 8080,
    [switch]$NoCache
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $workspaceRoot 'outputs/boptest-v0.9.0'
$revision = '9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9'

if ($Action -eq 'up') {
    $storageRoot = Join-Path $workspaceRoot 'outputs/storage-build'
    New-Item -ItemType Directory -Path $storageRoot -Force | Out-Null
    $archives = @(
        @('minio', 'f19c534b9f457773dcd043d977433e1a71525c3b', 'ac3bc23e5d91981a89921800a931a59aad27146434f952e9a79076b919d8a661'),
        @('mc', '1e78af443bf0443eb177f26421fe081f08d83dc5', '703d2c20ed692c6d534f680276165fd11e006c273ec64a0c8e708b74d6176112')
    )
    foreach ($archive in $archives) {
        $destination = Join-Path $storageRoot ($archive[0] + '.tar.gz')
        if (-not (Test-Path -LiteralPath $destination)) {
            Invoke-WebRequest -Uri "https://codeload.github.com/minio/$($archive[0])/tar.gz/$($archive[1])" -OutFile $destination
        }
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower() -ne $archive[2]) {
            throw "Source archive checksum mismatch: $destination"
        }
    }
    if (-not (Test-Path -LiteralPath $sourceRoot)) {
        New-Item -ItemType Directory -Path (Join-Path $workspaceRoot 'outputs') -Force | Out-Null
        git clone --depth 1 --branch v0.9.0 --filter=blob:none --no-checkout https://github.com/ibpsa/project1-boptest.git $sourceRoot
        if ($LASTEXITCODE -ne 0) { throw 'Could not clone BOPTEST v0.9.0.' }
        git -C $sourceRoot sparse-checkout init --cone
        if ($LASTEXITCODE -ne 0) { throw 'Could not initialize sparse checkout.' }
        git -C $sourceRoot sparse-checkout set service data forecast kpis testcases/bestest_hydronic_heat_pump
        if ($LASTEXITCODE -ne 0) { throw 'Could not select the single-zone test case.' }
        git -C $sourceRoot checkout v0.9.0
        if ($LASTEXITCODE -ne 0) { throw 'Could not check out the pinned release.' }
    }
    $actualRevision = git -C $sourceRoot rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $actualRevision -ne $revision) {
        throw 'The BOPTEST source revision differs from the pinned release.'
    }
    $sourceChanges = @(git -C $sourceRoot status --porcelain)
    if ($LASTEXITCODE -ne 0 -or $sourceChanges.Count -ne 0) {
        throw 'BOPTEST source has local changes; preserve and inspect them before deployment.'
    }
    $fmuPath = Join-Path $sourceRoot 'testcases/bestest_hydronic_heat_pump/models/wrapped.fmu'
    if ((Get-FileHash -LiteralPath $fmuPath -Algorithm SHA256).Hash.ToLower() -ne '674b9500c1c89fdbec64f2c053af8ddcd06afebe63bd7c1b7fc708692a11863e') {
        throw 'The single-zone FMU checksum differs from the pinned model.'
    }
}

$composeFile = Join-Path $workspaceRoot 'infra/boptest.compose.yml'
$normalizedComposeFile = $composeFile.Replace('\', '/')
$linuxComposeFile = wsl -d $Distribution --exec wslpath -a -u $normalizedComposeFile
if ($LASTEXITCODE -ne 0) { throw 'Could not map the project path into WSL.' }
wsl -d $Distribution -u root -- systemctl start docker
if ($LASTEXITCODE -ne 0) { throw 'Docker Engine is required inside the selected WSL distribution.' }

$storageImage = if ($ProjectName -eq 'thermal-boptest') { 'thermal-boptest-storage:2025-04' } else { "thermal-boptest-storage:$ProjectName" }
$provisionImage = "thermal-boptest-provision:$ProjectName"
$composePrefix = @('-d', $Distribution, '-u', 'root', '--', 'env', "BOPTEST_PROJECT=$ProjectName",
    "BOPTEST_PORT=$Port", "BOPTEST_STORAGE_IMAGE=$storageImage", "BOPTEST_PROVISION_IMAGE=$provisionImage",
    'docker', 'compose', '-f', $linuxComposeFile)

switch ($Action) {
    'up' {
        $buildArgs = @('build')
        if ($NoCache) { $buildArgs += '--no-cache' }
        & wsl @composePrefix @buildArgs
        if ($LASTEXITCODE -ne 0) { throw 'BOPTEST auxiliary image build failed.' }
        & wsl @composePrefix up -d --no-build
        if ($LASTEXITCODE -ne 0) { throw 'BOPTEST deployment failed.' }
        $ready = $false
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            try {
                $version = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/version" -TimeoutSec 3
                $cases = @(Invoke-RestMethod -Uri "http://127.0.0.1:$Port/testcases" -TimeoutSec 3)
                if ($version.payload.version -eq '0.9.0' -and 'bestest_hydronic_heat_pump' -in $cases.testcaseid) {
                    $ready = $true
                    break
                }
            } catch { }
            Start-Sleep -Seconds 2
        }
        if (-not $ready) { throw 'BOPTEST API or test-case provisioning is not ready; inspect compose logs.' }
        Write-Output "BOPTEST 0.9.0 project $ProjectName is ready at http://127.0.0.1:$Port."
    }
    'down' {
        & wsl @composePrefix down
        if ($LASTEXITCODE -ne 0) { throw 'Could not stop the project containers.' }
    }
    'status' {
        & wsl @composePrefix ps -a
        if ($LASTEXITCODE -ne 0) { throw 'Could not read container status.' }
    }
}
