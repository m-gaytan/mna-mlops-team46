install:
    python -m pip install -r requirements.txt

data:
    python mlops/dataset.py

train:
    python mlops/modeling/train.py

predict:
    python mlops/modeling/predict.py
