$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$javaHome = Join-Path $projectRoot ".neo4j-local\jdk-17.0.18+8"
$neo4jScript = Join-Path $projectRoot ".neo4j-local\neo4j-community-5.26.0\bin\neo4j.ps1"

if (!(Test-Path (Join-Path $javaHome "bin\java.exe"))) {
    throw "Java runtime not found at $javaHome"
}

if (!(Test-Path $neo4jScript)) {
    throw "Neo4j script not found at $neo4jScript"
}

$env:JAVA_HOME = $javaHome
$env:PATH = "$javaHome\bin;$env:PATH"

powershell -NoProfile -ExecutionPolicy Bypass -File $neo4jScript console
