<#
.SYNOPSIS
    Stops AI Research Paper Assistant: the Streamlit server and its Docker services.

.DESCRIPTION
    - Reads the PID from .streamlit-<port>.pid and kills that process tree, but only
      if it is confirmed to be THIS project's app.py process. A fallback scan of the
      port also only stops this project's process. Other applications are never touched.
    - Stops the docker-compose services with `docker compose stop` (containers and
      named volumes are preserved). Use -RemoveContainers for `docker compose down`
      (containers removed, named data volumes still preserved).
    - Set -SkipDocker to stop only the Streamlit server.
    - All actions, including Docker output, are logged to logs\stop.log with timestamps.

.EXAMPLE
    .\stop.ps1
    .\stop.ps1 -Port 8502
    .\stop.ps1 -RemoveContainers
    .\stop.ps1 -SkipDocker
#>
[CmdletBinding()]
param(
    [int]$Port = 8501,
    [switch]$SkipDocker,
    [switch]$RemoveContainers,
    [int]$DockerWaitSeconds = 30
)

$ErrorActionPreference = 'Stop'

$Root      = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PidFile   = Join-Path $Root ".streamlit-$Port.pid"
$LogDir    = Join-Path $Root 'logs'
$StopLog   = Join-Path $LogDir 'stop.log'
$ComposeFile = Join-Path $Root 'docker-compose.yml'

Set-Location $Root

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}
try { New-Item -ItemType File -Path $StopLog -Force | Out-Null } catch { }

function Write-Log {
    param([string]$Level, [string]$Message, [string]$Color = 'White')
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    Write-Host $line -ForegroundColor $Color
    try { Add-Content -Path $StopLog -Value $line -Encoding UTF8 -ErrorAction SilentlyContinue } catch { }
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

function Test-PortListening([int]$PortNumber) {
    return (@(Get-NetTCPConnection -LocalPort $PortNumber -State Listen -ErrorAction SilentlyContinue).Count -gt 0)
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

function Stop-AssistantProcess($ProcInfo, [string]$Reason) {
    if (-not (Test-IsAssistantProcess $ProcInfo)) {
        Write-WarnLine "PID $($ProcInfo.ProcessId) is not this project's app.py - leaving it alone."
        return $false
    }
    Write-Info "Stopping PID $($ProcInfo.ProcessId) - $Reason"
    $killCode = 1
    try {
        $result = Invoke-Native -File 'taskkill' -Arguments @('/PID', [string]$ProcInfo.ProcessId, '/T', '/F')
        $killCode = $result.Code
        foreach ($line in $result.Output) { Write-Docker $line.ToString() }
    }
    catch {
        Write-Err "taskkill threw an exception: $($_.Exception.Message)"
    }
    if ($killCode -ne 0 -and (Get-CimInstance Win32_Process -Filter "ProcessId=$($ProcInfo.ProcessId)" -ErrorAction SilentlyContinue)) {
        Write-Err "PID $($ProcInfo.ProcessId) is still running after taskkill."
        return $false
    }
    $script:stopped += [int]$ProcInfo.ProcessId
    return $true
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

function Get-ComposeRunning {
    if ($script:UseComposeV2) { $result = Invoke-Native -File 'docker' -Arguments @('compose', 'ps', '--status', 'running', '--quiet') }
    else { $result = Invoke-Native -File 'docker-compose' -Arguments @('ps', '-q') }
    if ($result.Code -ne 0) { return @() }
    return @($result.Output | Where-Object { $_ })
}

function Stop-DockerServices {
    if ($SkipDocker) {
        Write-WarnLine 'Docker shutdown skipped (-SkipDocker).'
        return
    }
    if (-not (Test-Path $ComposeFile)) {
        Write-WarnLine "docker-compose.yml not found at $ComposeFile - skipping Docker shutdown."
        return
    }
    if (-not (Test-DockerDaemon)) { return }

    $script:UseComposeV2 = ((Invoke-Native -File 'docker' -Arguments @('compose', 'version')).Code -eq 0)
    if (-not $script:UseComposeV2) {
        if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
            Write-Err 'Neither "docker compose" nor "docker-compose" is available.'
            $script:hadError = $true
            return
        }
        Write-Info 'Using legacy docker-compose CLI.'
    }

    if ($RemoveContainers) {
        Write-Info 'Removing Docker containers (docker compose down; named volumes kept)...'
        $code = Invoke-Compose -Arguments @('down')
    }
    else {
        Write-Info 'Stopping Docker services (docker compose stop; containers kept)...'
        $code = Invoke-Compose -Arguments @('stop')
    }
    if ($code -ne 0) {
        Write-Err "Docker compose shutdown failed (exit code $code)."
        $script:hadError = $true
        return
    }
    Write-Ok 'Docker compose shutdown command succeeded.'

    for ($i = 0; $i -lt $DockerWaitSeconds; $i++) {
        $running = @(Get-ComposeRunning)
        if ($running.Count -eq 0) { break }
        Start-Sleep -Seconds 1
    }
    $running = @(Get-ComposeRunning)
    if ($running.Count -gt 0) {
        Write-WarnLine "Docker containers still running after ${DockerWaitSeconds}s: $($running -join ', ')"
        $script:hadError = $true
    }
    else {
        Write-Ok 'All docker-compose containers are stopped.'
    }
}

$script:UseComposeV2 = $false
$script:stopped = @()
$script:hadError = $false

try {
    Write-Log 'INFO' "=== Stop requested :: port $Port ===" -Color 'White'

    # --- 1. PID written by start.ps1 --------------------------------------------
    if (Test-Path $PidFile) {
        $raw = (Get-Content -Path $PidFile -Raw -ErrorAction SilentlyContinue).Trim()
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        if ($raw -match '^\d+$') {
            $procInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$raw)" -ErrorAction SilentlyContinue
            if ($procInfo) {
                if (-not (Stop-AssistantProcess -ProcInfo $procInfo -Reason "from .streamlit-$Port.pid")) { $script:hadError = $true }
            }
            else {
                Write-Info "PID $raw from the PID file was no longer running."
            }
        }
        else {
            Write-WarnLine "PID file $(Split-Path -Leaf $PidFile) was invalid and has been removed."
        }
    }

    # --- 2. Fallback: an AI Research Paper Assistant process on the port ----------
    $connections = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    foreach ($connection in $connections) {
        $ownerPid = [int]$connection.OwningProcess
        if ($script:stopped -contains $ownerPid) { continue }
        $procInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$ownerPid" -ErrorAction SilentlyContinue
        if (-not $procInfo) { continue }
        if (Test-IsAssistantProcess $procInfo) {
            if (-not (Stop-AssistantProcess -ProcInfo $procInfo -Reason "listening on port $Port")) { $script:hadError = $true }
        }
        else {
            Write-Info "Port $Port is in use by another app (PID $ownerPid); left it running."
        }
    }

    if ($script:stopped.Count -gt 0) {
        Write-Ok "Stopped AI Research Paper Assistant (PID(s): $($script:stopped -join ', '))."
    }
    else {
        Write-Info "No running AI Research Paper Assistant server on port $Port (nothing to stop)."
    }

    # --- 3. Verify the app port was released -------------------------------------
    if (Test-PortListening -PortNumber $Port) {
        $stillOurs = @(Get-PortOwners -PortNumber $Port | Where-Object { Test-IsAssistantProcess $_ })
        if ($stillOurs.Count -gt 0) {
            Write-WarnLine "An AI Research Paper Assistant process is still listening on port $Port."
            $script:hadError = $true
        }
    }

    # --- 4. Stop Docker services --------------------------------------------------
    Stop-DockerServices

    # --- 5. Summary ---------------------------------------------------------------
    if ($script:hadError) {
        Write-Err "Stop completed with errors. See $StopLog"
        exit 1
    }
    Write-Ok 'Stop completed successfully.'
    Write-Info "Log file: $StopLog"
}
catch {
    Write-Err "Stop failed: $($_.Exception.Message)"
    exit 1
}
