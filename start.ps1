<#
.SYNOPSIS
    Starts AI Research Paper Assistant: Docker services first, then the Streamlit app.

.DESCRIPTION
    - Runs from any directory (always targets this script's folder).
    - Starts the Neo4j service from docker-compose.yml and waits for it to become
      reachable. Set -SkipDocker to start only the app.
    - Creates .env from .env.example if it does not exist.
    - Installs dependencies with `uv sync` if the virtual environment is missing
      (falls back to `python -m venv` + pip if uv is not installed).
    - Warns about empty external API keys but still starts the server.
    - Writes the server PID to .streamlit-<port>.pid.
    - Refuses to touch a different application that already occupies the port.
    - All actions (including Docker output) are logged to logs\start.log with timestamps.
      Streamlit stdout/stderr go to logs\streamlit-<port>.log / .err.log.

.EXAMPLE
    .\start.ps1
    .\start.ps1 -OpenBrowser
    .\start.ps1 -Port 8502
    .\start.ps1 -SkipDocker
#>
[CmdletBinding()]
param(
    [int]$Port = 8501,
    [switch]$OpenBrowser,
    [switch]$SkipSync,
    [switch]$SkipDocker,
    [int]$DockerWaitSeconds = 90
)

$ErrorActionPreference = 'Stop'

$Root          = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PidFile       = Join-Path $Root ".streamlit-$Port.pid"
$LogDir        = Join-Path $Root 'logs'
$StartLog      = Join-Path $LogDir 'start.log'
$EnvFile       = Join-Path $Root '.env'
$EnvExample    = Join-Path $Root '.env.example'
$ComposeFile   = Join-Path $Root 'docker-compose.yml'
$VenvPython    = Join-Path $Root '.venv\Scripts\python.exe'
$VenvStreamlit = Join-Path $Root '.venv\Scripts\streamlit.exe'
$Url           = "http://localhost:$Port"

Set-Location $Root

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}
try { New-Item -ItemType File -Path $StartLog -Force | Out-Null } catch { }

function Write-Log {
    param([string]$Level, [string]$Message, [string]$Color = 'White')
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    Write-Host $line -ForegroundColor $Color
    try { Add-Content -Path $StartLog -Value $line -Encoding UTF8 -ErrorAction SilentlyContinue } catch { }
}
function Write-Info([string]$Message) { Write-Log 'INFO' $Message 'Cyan' }
function Write-Ok([string]$Message) { Write-Log 'OK' $Message 'Green' }
function Write-WarnLine([string]$Message) { Write-Log 'WARN' $Message 'Yellow' }
function Write-Err([string]$Message) { Write-Log 'ERROR' $Message 'Red' }
function Write-Docker([string]$Message) { Write-Log 'DOCKER' $Message 'DarkGray' }

# Native commands write progress to stderr. With ErrorActionPreference='Stop' PowerShell
# would treat that as a terminating error, so temporarily relax it around native calls.
function Invoke-Native {
    param([string]$File, [string[]]$Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $File @Arguments 2>&1
        $code = $LASTEXITCODE
        return [pscustomobject]@{ Code = $code; Output = @($output) }
    }
    finally { $ErrorActionPreference = $previous }
}

# A process counts as ours only if its command line points into this project
# AND runs app.py (Streamlit's listener is a child python.exe outside .venv).
function Test-IsAssistantProcess($ProcInfo) {
    if (-not $ProcInfo) { return $false }
    $cmd = [string]$ProcInfo.CommandLine
    if (-not $cmd) { return $false }
    $isInThisProject = $cmd -match [regex]::Escape($Root)
    $runsOurApp = $cmd -match '(?i)(^|[\s"/])app\.py($|[\s"])'
    return ($isInThisProject -and $runsOurApp)
}

function Get-PortOwners([int]$PortNumber) {
    $owners = @()
    $connections = @(Get-NetTCPConnection -LocalPort $PortNumber -State Listen -ErrorAction SilentlyContinue)
    foreach ($connection in $connections) {
        $info = Get-CimInstance Win32_Process -Filter "ProcessId=$($connection.OwningProcess)" -ErrorAction SilentlyContinue
        if ($info) { $owners += $info }
    }
    return $owners
}

function Test-PortListening([int]$PortNumber) {
    return (@(Get-NetTCPConnection -LocalPort $PortNumber -State Listen -ErrorAction SilentlyContinue).Count -gt 0)
}

function Test-DockerDaemon {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-WarnLine 'Docker CLI was not found in PATH. Docker services will be skipped.'
        return $false
    }
    $result = Invoke-Native -File 'docker' -Arguments @('info', '--format', '{{.ServerVersion}}')
    if ($result.Code -ne 0) {
        foreach ($line in $result.Output) { Write-Docker $line.ToString() }
        Write-WarnLine 'Docker daemon is not reachable. Docker services will be skipped.'
        return $false
    }
    Write-Info "Docker daemon available (server $(@($result.Output)[0]))."
    return $true
}

function Invoke-Compose {
    param([string[]]$Arguments)
    if ($script:UseComposeV2) { $result = Invoke-Native -File 'docker' -Arguments (@('compose') + $Arguments) }
    else { $result = Invoke-Native -File 'docker-compose' -Arguments $Arguments }
    foreach ($line in $result.Output) { if ($line) { Write-Docker $line.ToString() } }
    return $result.Code
}

function Start-DockerServices {
    if ($SkipDocker) {
        Write-WarnLine 'Docker startup skipped (-SkipDocker).'
        return $false
    }
    if (-not (Test-Path $ComposeFile)) {
        Write-Err "docker-compose.yml not found at $ComposeFile"
        return $false
    }
    if (-not (Test-DockerDaemon)) { return $false }

    $script:UseComposeV2 = ((Invoke-Native -File 'docker' -Arguments @('compose', 'version')).Code -eq 0)
    if (-not $script:UseComposeV2) {
        if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
            Write-Err 'Neither "docker compose" nor "docker-compose" is available.'
            return $false
        }
        Write-Info 'Using legacy docker-compose CLI.'
    }

    Write-Info 'Starting Docker service: neo4j (docker compose up -d neo4j)...'
    $code = Invoke-Compose -Arguments @('up', '-d', 'neo4j')
    if ($code -ne 0) {
        Write-Err "docker compose up failed (exit code $code). The app will start without the Neo4j graph."
        return $false
    }
    Write-Ok 'Docker compose started the neo4j service.'

    $neoPort = 7476
    if ($script:EnvValues.ContainsKey('NEO4J_HTTP_URL') -and $script:EnvValues['NEO4J_HTTP_URL'] -match ':(\d+)/?$') {
        $neoPort = [int]$Matches[1]
    }
    for ($i = 0; $i -lt $DockerWaitSeconds; $i++) {
        if (Test-PortListening -PortNumber $neoPort) {
            Write-Ok "Neo4j is reachable on port $neoPort."
            return $true
        }
        Start-Sleep -Seconds 1
    }
    Write-WarnLine "Neo4j did not become reachable on port $neoPort within ${DockerWaitSeconds}s. The app will start without the graph."
    return $false
}

$script:UseComposeV2 = $false
$script:EnvValues = @{}

try {
    Write-Log 'INFO' "=== Start requested :: port $Port ===" -Color 'White'

    # --- 1. Make sure an .env file exists --------------------------------------
    if (-not (Test-Path $EnvFile)) {
        if (Test-Path $EnvExample) {
            Copy-Item -Path $EnvExample -Destination $EnvFile
            Write-Info 'Created .env from .env.example - add your API keys there.'
        }
        else {
            New-Item -ItemType File -Path $EnvFile | Out-Null
            Write-Info 'Created an empty .env file.'
        }
    }

    # --- 2. Parse env values and warn about empty external keys -----------------
    foreach ($line in (Get-Content -Path $EnvFile -ErrorAction SilentlyContinue)) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $script:EnvValues[$Matches[1]] = $Matches[2]
        }
    }
    $required = @('OPENAI_API_KEY', 'TAVILY_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY')
    $emptyKeys = @($required | Where-Object {
        -not $script:EnvValues.ContainsKey($_) -or
        [string]::IsNullOrWhiteSpace($script:EnvValues[$_]) -or
        $script:EnvValues[$_] -match '^(your[-_]|<|sk-\.\.\.|tvly-\.\.\.)'
    })
    if ($emptyKeys.Count -gt 0) {
        Write-WarnLine "These keys are empty in .env: $($emptyKeys -join ', ')"
        Write-WarnLine 'The server will start, but chat/upload features need real keys.'
    }

    # --- 3. Bring up Docker services first --------------------------------------
    $dockerReady = Start-DockerServices

    # --- 4. Is the app already running / is the port taken by someone else? ------
    $owners = @(Get-PortOwners -PortNumber $Port)
    $ourOwners = @($owners | Where-Object { Test-IsAssistantProcess $_ })

    $trackedAlive = $false
    if (Test-Path $PidFile) {
        $raw = (Get-Content -Path $PidFile -Raw -ErrorAction SilentlyContinue).Trim()
        if ($raw -match '^\d+$') {
            $trackedInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$raw)" -ErrorAction SilentlyContinue
            if ($trackedInfo -and (Test-IsAssistantProcess $trackedInfo)) { $trackedAlive = $true }
        }
        if (-not $trackedAlive) {
            Write-WarnLine "Removing stale PID file $(Split-Path -Leaf $PidFile)."
            Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        }
    }

    if ($ourOwners.Count -gt 0 -or $trackedAlive) {
        Write-WarnLine "AI Research Paper Assistant is already running on port $Port."
        Write-Info "Open $Url  |  stop it with: .\stop.ps1 -Port $Port"
        if (-not $dockerReady) { Write-WarnLine 'Docker services were not started; use start.ps1 again after fixing Docker.' }
        return
    }
    if ($owners.Count -gt 0) {
        $other = $owners[0]
        Write-Err "Port $Port is occupied by another app (PID $($other.ProcessId): $($other.CommandLine))."
        Write-Err 'AI Research Paper Assistant was NOT started; the other app was NOT touched.'
        Write-Info 'Use a different port, e.g.: .\start.ps1 -Port 8502'
        exit 1
    }
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue }

    # --- 5. Install dependencies if needed --------------------------------------
    if (-not (Test-Path $VenvPython)) {
        if (-not $SkipSync -and (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Info 'No virtual environment found - running uv sync (first run takes a while)...'
            $result = Invoke-Native -File 'uv' -Arguments @('sync')
            foreach ($line in $result.Output) { Write-Docker $line.ToString() }
            if ($result.Code -ne 0) { throw "uv sync failed (exit code $($result.Code)). See logs\start.log" }
        }
        else {
            Write-Info 'No virtual environment found - creating one with python and installing requirements.txt...'
            $result = Invoke-Native -File 'python' -Arguments @('-m', 'venv', '.venv')
            foreach ($line in $result.Output) { Write-Docker $line.ToString() }
            if ($result.Code -ne 0) { throw 'Could not create a virtual environment. See logs\start.log' }
            $result = Invoke-Native -File $VenvPython -Arguments @('-m', 'pip', 'install', '-r', 'requirements.txt')
            foreach ($line in $result.Output) { Write-Docker $line.ToString() }
            if ($result.Code -ne 0) { throw "pip install failed (exit code $($result.Code)). See logs\start.log" }
        }
    }
    if (-not (Test-Path $VenvStreamlit)) {
        throw 'Streamlit was not found in .venv. Run: uv sync'
    }

    # --- 6. Start the server in the background ----------------------------------
    $stArgs = @(
        'run', 'app.py',
        "--server.port=$Port",
        '--server.address=127.0.0.1',
        '--server.headless=true',
        '--browser.gatherUsageStats=false'
    )
    Write-Info "Starting Streamlit on port $Port..."
    $proc = Start-Process -FilePath $VenvStreamlit `
        -ArgumentList $stArgs `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput (Join-Path $LogDir "streamlit-$Port.log") `
        -RedirectStandardError (Join-Path $LogDir "streamlit-$Port.err.log")
    Set-Content -Path $PidFile -Value $proc.Id -Encoding ASCII
    Write-Ok "Server PID $($proc.Id) saved to $(Split-Path -Leaf $PidFile)"

    # --- 7. Wait until it answers ------------------------------------------------
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        if ($proc.HasExited) {
            Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
            throw "Streamlit exited during startup (code $($proc.ExitCode)). See logs\streamlit-$Port.err.log"
        }
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        }
        catch { Start-Sleep -Milliseconds 500 }
    }

    if (-not $ready) {
        Write-WarnLine "Server did not become healthy in time. See logs\streamlit-$Port.err.log"
    }
    else {
        Write-Ok "AI Research Paper Assistant is ready at $Url"
    }
    if (-not $dockerReady -and -not $SkipDocker) {
        Write-WarnLine 'Docker services are not ready - graph features may be unavailable.'
    }

    if ($OpenBrowser) { Start-Process $Url }
    Write-Info "Stop everything with: .\stop.ps1 -Port $Port"
    Write-Info "Log file: $StartLog"
}
catch {
    Write-Err "Start failed: $($_.Exception.Message)"
    exit 1
}
