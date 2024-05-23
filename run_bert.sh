
# Train BERT model
python3 bert.py --dataset data/CrossNER/conll2003/train.txt -e 10 -lr 0.00005


# Evaluate BERT model on dataset
python3 bert_eval.py