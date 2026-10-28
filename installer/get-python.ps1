# Fetch the pinned portable Python for setup.cmd.
#
# Used only when no Python 3.11-3.13 is installed. Downloads python.org's
# NuGet build (a plain zip: no installer, no registry, no PATH change) into
# <Artifex folder>\python, and refuses it unless:
#   - the URL is https on api.nuget.org (the pinned URL in catalog.json),
#   - the size and SHA-512 match the pins in catalog.json (NuGet's published
#     packageHash, not a hash taken from the download), and
#   - python.exe carries a valid Authenticode signature from the Python
#     Software Foundation.
# A refused download is deleted; nothing half-written is left behind.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pin = (Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'catalog.json') | ConvertFrom-Json).bootstrap_python

$uri = [Uri]$pin.url
if ($uri.Scheme -ne 'https' -or $uri.Host -ne 'api.nuget.org') { throw "refused: $($pin.url) is not https on api.nuget.org" }
if ($pin.dest -notmatch '^[A-Za-z0-9_-]+$') { throw "refused: bad destination '$($pin.dest)'" }

$dest = Join-Path $root $pin.dest
$exe = Join-Path $dest 'tools\python.exe'
$zip = Join-Path $root 'python-download.zip'
$tmp = Join-Path $root 'python-unpack.tmp'

function Test-PythonSignature($path) {
    $sig = Get-AuthenticodeSignature -FilePath $path
    return ($sig.Status -eq 'Valid' -and $sig.SignerCertificate.Subject -match "O=`"?$([regex]::Escape($pin.signer))")
}

if ((Test-Path $exe) -and (Test-PythonSignature $exe)) {
    Write-Output "Portable Python $($pin.version) is already here."
    exit 0
}

Write-Output "Downloading Python $($pin.version) from python.org's NuGet package ($([math]::Round($pin.size / 1MB)) MB)..."
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
try {
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $pin.url -OutFile $zip -UseBasicParsing -MaximumRedirection 0
    $size = (Get-Item $zip).Length
    if ($size -ne $pin.size) { throw "got $size bytes, expected $($pin.size); refused" }
    $hash = (Get-FileHash -Algorithm SHA512 $zip).Hash.ToLower()
    if ($hash -ne $pin.sha512.ToLower()) { throw "SHA-512 does not match the pinned value; refused" }
    Write-Output "Checksum OK."

    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    Expand-Archive -Path $zip -DestinationPath $tmp
    if (-not (Test-PythonSignature (Join-Path $tmp 'tools\python.exe'))) {
        throw "python.exe is not signed by $($pin.signer); refused"
    }
    Write-Output "Signature OK ($($pin.signer))."
    if (Test-Path $dest) { throw "$dest already exists and isn't this Python; move it aside and run again (nothing was changed)" }
    Move-Item $tmp $dest
    Write-Output "Portable Python ready in $dest"
} catch {
    Write-Output "Could not set up Python: $($_.Exception.Message)"
    Write-Output "Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and run this again."
    exit 1
} finally {
    if (Test-Path $zip) { Remove-Item -Force $zip }
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
}
