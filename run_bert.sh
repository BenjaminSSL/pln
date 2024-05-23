
# Train BERT model
python3 bert.py --dataset data/CrossNER/conll2003/train_testing.txt -e 1 -lr 0.00005


# Evaluate BERT model on dataset
python3 bert_eval.py