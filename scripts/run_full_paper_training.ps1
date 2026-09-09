$ErrorActionPreference = "Continue"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"

$resultDirectory = Join-Path (Get-Location) "results\chengdu_agri_greenhouse_001\controller_benchmark\paper"
$logPath = Join-Path $resultDirectory "background_training.log"

Add-Content -Path $logPath -Value "[$(Get-Date -Format o)] Starting hidden resumable PPO/SAC paper training."
Add-Content -Path $logPath -Value "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2"

& "D:\software\anaconda3\python.exe" -m experiments.controllers.run_chengdu_benchmark `
    --profile paper `
    --algorithms ppo sac `
    --resume `
    --parallel-rl-seeds `
    --max-seed-workers 5 *>> $logPath

$exitCode = $LASTEXITCODE
Add-Content -Path $logPath -Value "[$(Get-Date -Format o)] Training process exited with code $exitCode."
exit $exitCode
