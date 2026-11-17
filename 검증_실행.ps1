param(
    [string]$PythonExecutable = 'python',
    [string]$Model,
    [string]$ReportDir
)

$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot '01_소스코드/verify_package.py'
$arguments = @('-B', $script)
if ($Model) { $arguments += @('--model', $Model) }
if ($ReportDir) { $arguments += @('--report-dir', $ReportDir) }

& $PythonExecutable @arguments
exit $LASTEXITCODE
