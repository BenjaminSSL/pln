### CONFIG ###

# Which dataset to train the model on, has to be in CoNLL format
DATASET="data/CrossNER/ai/test_all.txt"
# Number of epochs to train the model
EPOCHS=1
# Learning rate for the optimizer
LEARNING_RATE=0.1


### TRAINING ###

python3 rnn.py -d $DATASET -e $EPOCHS -lr $LEARNING_RATE

### EVALUATION ###

python3 rnn_eval.py