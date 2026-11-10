param(
    [string[]]$Audio,
    [string]$Folder,
    [string]$RunDir,
    [string]$Csv,
    [switch]$Inspect,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
$candidatePythonPaths = @(
    (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
    (Join-Path $PSScriptRoot '../.venv/Scripts/python.exe')
)
$inferencePython = $candidatePythonPaths | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $inferencePython) {
    throw 'Python environment not found. Install dependencies as described in INFERENCE.md or use python inference.py in your torch environment.'
}
$inferenceArguments = @((Join-Path $PSScriptRoot 'inference.py'), '--device', 'cpu')
if ($RunDir) { $inferenceArguments += @('--run-dir', $RunDir) }
if ($Audio) { $inferenceArguments += '--audio'; $inferenceArguments += $Audio }
elseif ($Folder) { $inferenceArguments += @('--folder', $Folder) }
else { $inferenceArguments += '--inspect' }
if ($Inspect -and ($Audio -or $Folder)) { throw 'Use -Inspect separately from -Audio or -Folder.' }
if ($Json) { $inferenceArguments += '--json' }
if ($Csv) { $inferenceArguments += @('--csv', $Csv) }
& $inferencePython @inferenceArguments
exit $LASTEXITCODE
