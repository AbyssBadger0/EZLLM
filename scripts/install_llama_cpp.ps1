[CmdletBinding()]
param(
    [ValidateSet("auto", "cpu", "cuda12", "cuda13", "vulkan")]
    [string] $Backend = "auto",

    [string] $Tag = "latest",

    [string] $Destination,

    [switch] $DryRun,

    [switch] $Force
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-DefaultBackend {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -ne $nvidiaSmi) {
        try {
            $caps = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
            foreach ($cap in $caps) {
                $parsed = 0.0
                if ([double]::TryParse($cap.Trim(), [ref] $parsed) -and $parsed -ge 12.0) {
                    return "cuda13"
                }
            }
        }
        catch {
            return "cuda12"
        }

        return "cuda12"
    }
    return "cpu"
}

function Get-Release {
    param([string] $RequestedTag)

    $headers = @{ "User-Agent" = "EZLLM llama.cpp installer" }
    if ($RequestedTag -eq "latest") {
        return Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    }

    return Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$RequestedTag"
}

function Select-ReleaseAsset {
    param(
        [object] $Release,
        [string] $Pattern
    )

    $asset = $Release.assets | Where-Object { $_.name -match $Pattern } | Select-Object -First 1
    if ($null -eq $asset) {
        $available = ($Release.assets | ForEach-Object { "  - $($_.name)" }) -join [Environment]::NewLine
        throw "Could not find release asset matching '$Pattern' in $($Release.tag_name). Available assets:$([Environment]::NewLine)$available"
    }

    return $asset
}

function Get-AssetPatterns {
    param(
        [string] $ResolvedBackend,
        [string] $TagName
    )

    $tagPattern = [regex]::Escape($TagName)
    switch ($ResolvedBackend) {
        "cpu" {
            return @("^llama-$tagPattern-bin-win-cpu-x64\.zip$")
        }
        "cuda12" {
            return @(
                "^llama-$tagPattern-bin-win-cuda-12\.4-x64\.zip$",
                "^cudart-llama-bin-win-cuda-12\.4-x64\.zip$"
            )
        }
        "cuda13" {
            return @(
                "^llama-$tagPattern-bin-win-cuda-13\.1-x64\.zip$",
                "^cudart-llama-bin-win-cuda-13\.1-x64\.zip$"
            )
        }
        "vulkan" {
            return @("^llama-$tagPattern-bin-win-vulkan-x64\.zip$")
        }
        default {
            throw "Unsupported backend: $ResolvedBackend"
        }
    }
}

function Initialize-Destination {
    param(
        [string] $DestinationPath,
        [string] $VendorRoot,
        [bool] $AllowRemove
    )

    $destinationFull = [System.IO.Path]::GetFullPath($DestinationPath)
    $vendorFull = [System.IO.Path]::GetFullPath($VendorRoot)

    if ((Test-Path $destinationFull) -and ((Get-ChildItem -Force $destinationFull | Measure-Object).Count -gt 0)) {
        if (-not $AllowRemove) {
            throw "Destination is not empty: $destinationFull. Re-run with -Force to replace a vendor/llama destination."
        }

        if (-not $destinationFull.StartsWith($vendorFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove a non-vendor destination: $destinationFull"
        }

        Remove-Item -LiteralPath $destinationFull -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $destinationFull | Out-Null
    return $destinationFull
}

$repoRoot = Get-RepoRoot
$vendorRoot = Join-Path $repoRoot "vendor\llama"
$resolvedBackend = if ($Backend -eq "auto") { Get-DefaultBackend } else { $Backend }

if (-not $Destination) {
    $Destination = Join-Path $vendorRoot "win-x64-$resolvedBackend"
}

$release = Get-Release -RequestedTag $Tag
$patterns = Get-AssetPatterns -ResolvedBackend $resolvedBackend -TagName $release.tag_name
$assets = foreach ($pattern in $patterns) {
    Select-ReleaseAsset -Release $release -Pattern $pattern
}

Write-Host "llama.cpp release: $($release.tag_name)"
Write-Host "backend: $resolvedBackend"
Write-Host "destination: $Destination"
Write-Host "assets:"
$assets | ForEach-Object { Write-Host "  - $($_.name)" }

if ($DryRun) {
    Write-Host "Dry run only. No files downloaded."
    exit 0
}

$destinationFull = Initialize-Destination -DestinationPath $Destination -VendorRoot $vendorRoot -AllowRemove:$Force.IsPresent
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("ezllm-llama-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

try {
    foreach ($asset in $assets) {
        $zipPath = Join-Path $tempRoot $asset.name
        Write-Host "Downloading $($asset.name)..."
        Invoke-WebRequest -Headers @{ "User-Agent" = "EZLLM llama.cpp installer" } -Uri $asset.browser_download_url -OutFile $zipPath
        Write-Host "Extracting $($asset.name)..."
        Expand-Archive -LiteralPath $zipPath -DestinationPath $destinationFull -Force
    }
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$server = Get-ChildItem -Path $destinationFull -Recurse -File -Filter "llama-server.exe" | Select-Object -First 1
if ($null -eq $server) {
    throw "Download completed, but llama-server.exe was not found under $destinationFull."
}

Write-Host ""
Write-Host "Installed llama-server:"
Write-Host $server.FullName
Write-Host ""
Write-Host "Use this in EZLLM config:"
Write-Host "[llama]"
Write-Host "server_bin = '$($server.FullName)'"
Write-Host "model_path = 'C:\path\to\your-model.gguf'"
