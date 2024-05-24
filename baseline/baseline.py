import torch
from torch import nn
import pickle


class Vocab():
    def __init__(self, pad_unk):
        """
        A convenience class that can help store a vocabulary
        and retrieve indices for inputs.
        """
        self.pad_unk = pad_unk
        self.word2idx = {self.pad_unk: 0}
        self.idx2word = [self.pad_unk]

    def __len__(self):
        return len(self.idx2word)

    def getIdx(self, word, add=False):
        if word not in self.word2idx:
            if add:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
            else:
                return self.word2idx[self.pad_unk]
        return self.word2idx[word]

    def getWord(self, idx):
        return self.idx2word[idx]


DIM_EMBEDDING = 100
RNN_HIDDEN = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 5
PAD = "PAD"
torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


train_data = pickle.load(open("ewt.train.pkl", "rb"))
test_data = pickle.load(open("ewt.test.pkl", "rb"))
dev_data = pickle.load(open("ewt.dev.pkl", "rb"))

train_data = train_data[:]
test_data = test_data[:]
dev_data = dev_data[:]

max_len = max([len(x[0]) for x in train_data])


class RNNTagger(nn.Module):
    def __init__(self, nwords, ntags):
        super(RNNTagger, self).__init__()
        # Create word embeddings
        self.word_embedding = nn.Embedding(
            nwords, DIM_EMBEDDING)
        # Create input dropout parameter
        self.word_dropout = nn.Dropout(.2)
        # Create LSTM parameters
        self.rnn = nn.RNN(DIM_EMBEDDING, RNN_HIDDEN, num_layers=1,
                          batch_first=True, bidirectional=False)
        # Create output dropout parameter
        self.rnn_output_dropout = nn.Dropout(.3)
        # Create final matrix multiply parameters
        self.hidden_to_tag = nn.Linear(RNN_HIDDEN, ntags)

    def forward(self, input):
        # Look up word vectors
        word_vectors = self.word_embedding(input)
        # Apply dropout
        dropped_word_vectors = self.word_dropout(word_vectors)
        rnn_out, _ = self.rnn(dropped_word_vectors, None)
        # Apply dropout
        rnn_out_dropped = self.rnn_output_dropout(rnn_out)
        # Matrix multiply to get scores for each tag
        output_scores = self.hidden_to_tag(rnn_out_dropped)

        # Calculate loss and predictions
        return output_scores


vocab_tokens = Vocab(PAD)
vocab_tags = Vocab(PAD)
# id_to_token = [PAD]

for tokens, tags in train_data:
    for token in tokens:
        vocab_tokens.getIdx(token, True)
    for tag in tags:
        vocab_tags.getIdx(tag, True)

NWORDS = len(vocab_tokens.idx2word)
NTAGS = len(vocab_tags.idx2word)


def data2feats(data, vocab_tokens, vocab_tags):
    """
    Converts the data into tensors. Uses the two vocabularies to convert the tokens and tags into indices.
    """
    feats = torch.zeros((len(data), max_len), dtype=torch.long)
    labels = torch.zeros((len(data), max_len), dtype=torch.long)
    for sentPos, sent in enumerate(data):
        for wordPos, word in enumerate(sent[0][:max_len]):
            wordIdx = vocab_tokens.getIdx(word)
            feats[sentPos][wordPos] = wordIdx
        for labelPos, label in enumerate(sent[1][:max_len]):
            labelIdx = vocab_tags.getIdx(label)
            labels[sentPos][labelPos] = labelIdx
    return feats, labels


train_feats, train_labels = data2feats(train_data, vocab_tokens, vocab_tags)

num_batches = int(len(train_feats)/BATCH_SIZE)
train_feats_batches = train_feats[:BATCH_SIZE *
                                  num_batches].view(num_batches, BATCH_SIZE, max_len)
train_labels_batches = train_labels[:BATCH_SIZE *
                                    num_batches].view(num_batches, BATCH_SIZE, max_len)

model = RNNTagger(NWORDS, NTAGS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')


print("Epoch\tLoss\tAccuracy")

for epoch in range(EPOCHS):
    model.train()  # Set the model to training mode
    model.zero_grad()

    loss = 0
    match = 0
    total = 0
    for batchIdx in range(0, num_batches):
        output_scores = model.forward(
            train_feats_batches[batchIdx])  # Forward pass
        output_scores = output_scores.view(
            BATCH_SIZE*max_len, -1)  # Reshape to 2D tensor
        flat_labels = train_labels_batches[batchIdx].view(
            BATCH_SIZE*max_len)  # Flatten the labels
        batch_loss = loss_function(
            output_scores, flat_labels)  # Calculate the loss

        predicted_labels = torch.argmax(output_scores, 1)
        predicted_labels = predicted_labels.view(BATCH_SIZE, max_len)

        # Backward pass
        batch_loss.backward()
        optimizer.step()
        model.zero_grad()
        loss += batch_loss.item()

        # Update the number of correct tags and total tags
        for gold_sent, pred_sent in zip(train_labels_batches[batchIdx], predicted_labels):
            for gold_label, pred_label in zip(gold_sent, pred_sent):
                if gold_label != 0:
                    total += 1
                    if gold_label == pred_label:
                        match += 1
        print('{0: <8}{1: <10}{2}'.format(epoch, '{:.2f}'.format(
            loss/num_batches), '{:.4f}'.format(match / total)))

sents_lens = []
for index, sents in enumerate(test_data):
    sents_lens.append(len(sents[0]))

BATCH_SIZE = 1
test_feats, test_labels = data2feats(test_data, vocab_tokens, vocab_tags)
num_batches_dev = int(len(test_feats)/BATCH_SIZE)

dev_feats_batches = test_feats[:BATCH_SIZE *
                               num_batches_dev].view(num_batches_dev, BATCH_SIZE, max_len)
dev_labels_batches = test_labels[:BATCH_SIZE *
                                 num_batches_dev].view(num_batches_dev, BATCH_SIZE, max_len)

sentences = []

for index, sents in enumerate(dev_feats_batches):
    model.eval()
    sents_len = sents_lens[index]
    output_scores = model.forward(sents)
    predicted_tags = torch.argmax(output_scores, 2)
    sents = [vocab_tokens.getWord(idx) for idx in sents.tolist()[0]]
    predicted_tags = [vocab_tags.getWord(idx)
                      for idx in predicted_tags.tolist()[0]]

    sents = sents[:sents_len]

    predicted_tags = predicted_tags[:sents_len]
    sentences.append((sents, predicted_tags))


def data_to_iob2(data, path):
    """
    Converts the data into IOB2 format.
    """
    with open(path, "w+") as f:

        for tokens, tags in data:
            idx = 1

            for token, tag in zip(tokens, tags):

                f.write(str(idx)+"\t" + token + "\t" + tag + "\n")
                idx += 1
            f.write("\n")


data_to_iob2(sentences, 'baseline_pred_test_epochs_{}.iob2'.format(EPOCHS))
