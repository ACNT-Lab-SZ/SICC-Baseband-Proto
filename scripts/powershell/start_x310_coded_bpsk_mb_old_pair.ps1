param(
    [double]$TxGain = 0,
    [double]$RxGain = 5,
    [int]$RxLeadSeconds = 3,
    [double]$MaxRunTimeSec = 60,
    [switch]$NoPlot
)

$matlab = $(if ($env:MATLAB_EXE) { $env:MATLAB_EXE } else { "matlab" })
$repo = $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path })

$plotFlag = '$true'
if ($NoPlot) {
    $plotFlag = '$false'
}

$rxCmd = "cd('$(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path })'); x310_coded_bpsk_mb_old_debug('rx', struct('txGain',$TxGain,'rxGain',$RxGain,'maxRunTimeSec',$MaxRunTimeSec,'showPlot',$plotFlag));"
$txCmd = "cd('$(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path })'); x310_coded_bpsk_mb_old_debug('tx', struct('txGain',$TxGain,'rxGain',$RxGain,'maxRunTimeSec',$MaxRunTimeSec,'showPlot',$plotFlag));"

Write-Host "Starting RX first..."
$rxProc = Start-Process -FilePath $matlab -ArgumentList '-batch', $rxCmd -PassThru

Start-Sleep -Seconds $RxLeadSeconds

Write-Host "Starting TX after RX lead time..."
$txProc = Start-Process -FilePath $matlab -ArgumentList '-batch', $txCmd -PassThru

Write-Host "RX PID: $($rxProc.Id)"
Write-Host "TX PID: $($txProc.Id)"
Write-Host "Press Ctrl+C in this PowerShell window if you want to stop waiting."

Wait-Process -Id $rxProc.Id, $txProc.Id

