# HiveBench Studio - one-shot fresh-machine setup.
#
# Idempotent: safe to re-run; each step skips work that is already done.
# Assumes: this repo at <root>, the dsh fork cloned at <root>\..\hivebench-studio
# (pass -ForkPath to override), Node 22+/24 + git + GPU drivers installed.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File setup.ps1 [-ForkPath <dir>] [-SkipFork] [-SkipLlama]
#
# What it does:
#   1. Python venv + requirements
#   2. Editable install of this repo (flat packages incl. harness/)
#   3. Fork toolchain: corepack shims on the user PATH, pnpm install, full build
#   4. Node carrier for the dsh Python SDK (pnpm deploy + fixups script)
#   5. Editable install of the fork's deepseek-harness sdk + runtime-bin
#   6. llama.cpp Vulkan llama-server into tools\llama.cpp (unless -SkipLlama)
#   7. Prints next steps (providers.local.json, python -m harness)
#
# NOTE: keep this file pure ASCII (PowerShell 5.1 reads BOM-less files as
# ANSI, and multi-byte characters corrupt the parse).

param(
    [string]$ForkPath = (Join-Path (Split-Path $PSScriptRoot -Parent) "hivebench-studio"),
    [switch]$SkipFork,
    [switch]$SkipLlama
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# --- 1. venv + requirements -------------------------------------------------
Step "Python venv + requirements"
if (-not (Test-Path $VenvPy)) {
    python -m venv (Join-Path $Root ".venv")
}
& $VenvPy -m pip install --upgrade pip --quiet
& $VenvPy -m pip install -r (Join-Path $Root "requirements.txt") -r (Join-Path $Root "requirements-dev.txt") --quiet
& $VenvPy -m pip install -e $Root --quiet
Write-Host "venv ready: $VenvPy"

if (-not $SkipFork) {
    if (-not (Test-Path (Join-Path $ForkPath "package.json"))) {
        Step "cloning the dsh fork into $ForkPath"
        git clone https://github.com/deepseek-ai/deepseek-harness.git $ForkPath
        Push-Location $ForkPath
        git checkout -q -b strata-studio b150a551b8
        Pop-Location
        Write-Host "fork cloned and pinned at b150a551b8 (branch strata-studio)"
    }
    if (-not (Test-Path (Join-Path $ForkPath "package.json"))) {
        throw "dsh fork not available at $ForkPath"
    }

    # --- 3. fork toolchain ----------------------------------------------------
    Step "corepack shims (user PATH)"
    $bin = Join-Path $env:USERPROFILE ".local\bin"
    New-Item -ItemType Directory -Path $bin -Force | Out-Null
    corepack enable --install-directory $bin 2>$null
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$bin*") {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$bin", "User")
    }
    $env:PATH = "$bin;$env:PATH"

    Step "pnpm install + build (fork) - several minutes"
    Push-Location $ForkPath
    corepack pnpm install
    corepack pnpm run build
    Pop-Location

    # --- 4. node carrier for the Python SDK ------------------------------------
    Step "dsh SDK node runtime carrier"
    Push-Location $ForkPath
    $staging = Join-Path $ForkPath "python\sdk-runtime\src\deepseek_harness_runtime\runtime\node"
    $entry = Join-Path $staging "node_modules\@deepseek-ai\dsh-sdk-jsonrpc-demo\lib\packaged-bin.js"
    if (-not (Test-Path $entry)) {
        corepack pnpm --filter dsh-jsonrpc-agent-pkg deploy --legacy --prod `
            --config.node-linker=hoisted --config.auto-install-peers=false `
            --config.link-workspace-packages=true $staging
        node (Join-Path $ForkPath "scripts\stage-node-carrier.mjs")
    } else {
        Write-Host "carrier already staged"
    }
    Pop-Location

    # --- 5. SDK editable installs ----------------------------------------------
    Step "deepseek-harness SDK (editable)"
    & $VenvPy -m pip install -e (Join-Path $ForkPath "python\sdk-runtime") --quiet
    & $VenvPy -m pip install -e (Join-Path $ForkPath "python\sdk") --quiet
}

# --- 6. llama.cpp Vulkan ------------------------------------------------------
if (-not $SkipLlama) {
    Step "llama.cpp llama-server (Vulkan) into tools\llama.cpp"
    $server = Join-Path $Root "tools\llama.cpp\llama-server.exe"
    if (-not (Test-Path $server)) {
        $rel = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
        $asset = $rel | ForEach-Object { $_.assets } |
            Where-Object { $_.name -match "bin-win-vulkan-x64\.zip$" } | Select-Object -First 1
        if (-not $asset) { throw "no win-vulkan asset found in recent llama.cpp releases" }
        $zip = Join-Path $Root "tools\llama-vulkan.zip"
        New-Item -ItemType Directory -Path (Join-Path $Root "tools") -Force | Out-Null
        Invoke-WebRequest $asset.browser_download_url -OutFile $zip
        Expand-Archive $zip -DestinationPath (Join-Path $Root "tools\llama.cpp") -Force
        Remove-Item $zip
    } else {
        Write-Host "llama-server already present"
    }
}

# --- done ----------------------------------------------------------------------
Step "Setup complete - next steps"
Write-Host @"
  1. (optional) copy providers.example.json -> providers.local.json and add keys
  2. start the studio:   .\.venv\Scripts\python -m harness --llama-port 1235
     offline demo:       .\.venv\Scripts\python -m harness --mock
  3. open the console:   http://127.0.0.1:8765/server
  4. optional hardening: set HARNESS_TOKEN (the console will prompt for it)
"@
