@echo off
set ANACONDA_PATH=E:\Programs\anaconda3
set ENV_NAME=tf_gpu
set LOG_DIR="logs\fit"
CALL %ANACONDA_PATH%\Scripts\activate.bat %ANACONDA_PATH%
CALL conda activate %ENV_NAME%
tensorboard --logdir=%LOG_DIR%
pause