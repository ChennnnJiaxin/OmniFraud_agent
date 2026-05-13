$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($projectRoot -match "[^\x00-\x7F]") {
    $existingSubst = subst | Select-String -Pattern ([regex]::Escape($projectRoot)) | Select-Object -First 1
    if ($existingSubst) {
        $shortRoot = ($existingSubst.ToString() -split "\\:")[0] + ":"
    } else {
        $usedDrives = (Get-PSDrive -PSProvider FileSystem).Name
        $driveLetter = @("Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q") | Where-Object { $_ -notin $usedDrives } | Select-Object -First 1
        if (-not $driveLetter) {
            throw "No free drive letter is available for a short Neo4j path."
        }
        $shortRoot = "${driveLetter}:"
        subst $shortRoot $projectRoot
    }

    $shortScript = Join-Path $shortRoot "run_neo4j_enterprise_console.ps1"
    powershell -NoProfile -ExecutionPolicy Bypass -File $shortScript
    exit $LASTEXITCODE
}

$javaHome = Join-Path $projectRoot ".neo4j-local\jdk-17.0.18+8"
$neo4jScript = Join-Path $projectRoot ".neo4j-local\neo4j-enterprise-5.26.0\bin\neo4j.ps1"

if (!(Test-Path (Join-Path $javaHome "bin\java.exe"))) {
    throw "Java runtime not found at $javaHome"
}

if (!(Test-Path $neo4jScript)) {
    throw "Neo4j Enterprise script not found at $neo4jScript"
}

$env:JAVA_HOME = $javaHome
$env:PATH = "$javaHome\bin;$env:PATH"
$env:NEO4J_ACCEPT_LICENSE_AGREEMENT = "yes"

powershell -NoProfile -ExecutionPolicy Bypass -File $neo4jScript console
