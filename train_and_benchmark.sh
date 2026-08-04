./clear_build.sh
python3 ml/train_can_ml.py
python3 conventional/can_conventional_ids.py 
python3 ml/run_can_ml_ids.py