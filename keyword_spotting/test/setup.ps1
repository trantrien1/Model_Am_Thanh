$ErrorActionPreference = 'Stop'
$packagePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'

if (-not (Test-Path -LiteralPath $packagePython)) {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    $selectedPython = $null
    $selectedArguments = @()
    if ($pythonLauncher) {
        foreach ($version in @('3.12', '3.11', '3.10')) {
            try {
                $probe = & $pythonLauncher.Source "-$version" -c 'import sys; print(sys.version_info[:2])' 2>$null
            } catch { continue }
            if ($LASTEXITCODE -eq 0) {
                $selectedPython = $pythonLauncher.Source
                $selectedArguments = @("-$version")
                break
            }
        }
    }
    if (-not $selectedPython) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            try {
                $probe = & $pythonCommand.Source -c 'import sys; sys.exit(0 if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12 else 1)' 2>$null
                if ($LASTEXITCODE -eq 0) { $selectedPython = $pythonCommand.Source }
            } catch { $selectedPython = $null }
        }
    }
    if (-not $selectedPython) {
        throw 'Install Python 3.10, 3.11 or 3.12, then run setup.ps1 again. See README.md.'
    }
    & $selectedPython @selectedArguments -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}

& $packagePython -c 'import sys; sys.exit(0 if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12 else 1)'
if ($LASTEXITCODE -ne 0) { throw 'The .venv environment must use Python 3.10, 3.11 or 3.12.' }
& $packagePython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip update failed.' }
& $packagePython -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'PyTorch CPU installation failed.' }
& $packagePython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $packagePython -B (Join-Path $PSScriptRoot 'inference.py') --inspect --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Model verification failed.' }
Write-Host 'Ready. Run infer.ps1 as shown in README.md.'
