@echo off
setlocal

cd /d "%~dp0.."

rem Keep five SB3 seed workers from oversubscribing CPU math libraries.
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "OPENBLAS_NUM_THREADS=2"

set "RESULT_DIR=results\chengdu_agri_greenhouse_001\controller_benchmark\paper"
set "LOG_FILE=%RESULT_DIR%\background_training.log"

echo [%date% %time%] Starting resumable PPO/SAC paper training.>>"%LOG_FILE%"
echo OMP_NUM_THREADS=%OMP_NUM_THREADS% MKL_NUM_THREADS=%MKL_NUM_THREADS% OPENBLAS_NUM_THREADS=%OPENBLAS_NUM_THREADS% >>"%LOG_FILE%"

python -m experiments.controllers.run_chengdu_benchmark ^
  --profile paper ^
  --algorithms ppo sac ^
  --resume ^
  --parallel-rl-seeds ^
  --max-seed-workers 5 >>"%LOG_FILE%" 2>&1

set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] Training process exited with code %EXIT_CODE%.>>"%LOG_FILE%"
exit /b %EXIT_CODE%
