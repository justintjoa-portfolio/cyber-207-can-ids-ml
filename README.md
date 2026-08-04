CYBER 207 ML based CAN IDS system 

Please first run pip install -r requirements.txt

To train the model (required to also form held-out data set for conventional and ml ids) run

python3 ml/train_can_ml.py

To run the conventional and ml ids, respectively, over the heldout data sets.

python3 conventional/can_conventional_ids.py 
python3 ml/run_can_ml_ids.py

To clear build:

./clear_build.sh

We also offer a script to expedite this. You can run it via: 
./train_and_benchmark.sh

