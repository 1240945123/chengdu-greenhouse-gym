param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid
)

$signature = @'
using System;
using System.Runtime.InteropServices;

public static class ExecutionState
{
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@

Add-Type -TypeDefinition $signature

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]0x00000001
$awayModeRequired = [uint32]0x00000040

try {
    while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
        [void][ExecutionState]::SetThreadExecutionState(
            $continuous -bor $systemRequired -bor $awayModeRequired
        )
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][ExecutionState]::SetThreadExecutionState($continuous)
}
