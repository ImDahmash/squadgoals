# #SQuADGoals


## Building

```
./get_started.sh

./split_glove.sh

python preprocess.py

python train.py
```

## Results

My implementation achieve 53.47 F1 score, and 41.425 Exact Match on the SQuAD hidden test set.
It falls short of the original authors' scores, but still handily beats a heavily feature-engineered
logistic regression implmentation.

You can see the run [here on CodaLab](https://worksheets.codalab.org/bundles/0x8803cedd7f1f4f7981312d1b2c472cf2/).
