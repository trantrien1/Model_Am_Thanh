param(
    [string[]]$Audio,
    [string]$Folder,
    [string]$Csv,
    [string]$Python,
    [switch]$Inspect,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
if (-not $Python) { $Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Run setup.ps1 first. If you already have a compatible Python environment, pass -Python with its python.exe path. See README.md.'
}
if ($Inspect -and ($Audio -or $Folder)) { throw 'Use -Inspect separately from -Audio or -Folder.' }
if ($Audio -and $Folder) { throw 'Use either -Audio or -Folder.' }
$inferenceArguments = @('-B', (Join-Path $PSScriptRoot 'inference.py'), '--device', 'cpu')
if ($Audio) { $inferenceArguments += '--audio'; $inferenceArguments += $Audio }
elseif ($Folder) { $inferenceArguments += @('--folder', $Folder) }
else { $inferenceArguments += '--inspect' }
if ($Json) { $inferenceArguments += '--json' }
if ($Csv) { $inferenceArguments += @('--csv', $Csv) }
& $Python @inferenceArguments
exit $LASTEXITCODE
