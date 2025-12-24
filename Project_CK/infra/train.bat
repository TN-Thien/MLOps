@echo off
setlocal
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m training.src.train --epochs 3 --batch-size 16 --lr 1e-3 --backbone mobilenet_v2 --alias staging --primary-metric srocc