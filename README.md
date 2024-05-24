# Group PLN

## NLP Project on NER using BERT and RNN models

### Contributors:

-   Benjamin Storm Larsen (bsla@itu.dk)
-   Yuliia Storm Larsen (yuls@itu.dk)
-   Hakon Eriksen (hake@itu.dk)

The project used python version 3.12.2
Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

The repository contains two models, so we have created two seperate bash scripts for running the models and evaluating the results. The bash scripts are as follows:

```bash
bash run_bert.sh
```

```bash
bash run_rnn.sh
```

However for the sake of simplicity, we have also created a single bash script that runs both the models. To run it use the following command:

```bash
bash run.sh
```

The baseline should me runned manually, as it is not included in the run.sh script. To run the baseline use the following command:

```bash
cd baseline
python baseline.py
```
