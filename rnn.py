import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from util.vocab import Vocab
from util.data import load_data
from argparse import ArgumentParser


def prepare_sequence(seq, vocab: Vocab):
    idxs = [vocab.getIdx(w) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def reverse_sequence(seq, vocab: Vocab):
    tokens = [vocab.gettoken(w) for w in seq]
    return tokens


def make_vocabs(train_data, unk_pad="<UNK>"):
    token_vocab = Vocab(unk_pad)
    tag_vocab = Vocab(unk_pad)
    for tokens, tags in train_data:
        for token in tokens:
            token_vocab.getIdx(token, True)
        for tag in tags:
            tag_vocab.getIdx(tag, True)
    return token_vocab, tag_vocab


class RNN(nn.Module):
    def __init__(self, params, n_tokens, n_tags):
        super(RNN, self).__init__()

        self.token_embeddings = nn.Embedding(n_tokens, params["embedding_dim"])

        self.lstm = nn.LSTM(params["embedding_dim"], params["hidden_dim"])

        self.hidden2tag = nn.Linear(params['hidden_dim'], n_tags)

    def forward(self, x,):
        embeds = self.token_embeddings(x)

        lstm_out, _ = self.lstm(embeds.view(len(x), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(x), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def train(train_data, epochs, learning_rate):
    print(f'Training on {len(train_data)} examples with {
          epochs} epochs and learning rate {learning_rate}...')

    params = {'hidden_dim': 128, 'embedding_dim': 128,
              'num_layers': 1, 'dropout': 0.2}
    token_vocab, tag_vocab = make_vocabs(train_data)

    model = RNN(params, len(token_vocab), len(tag_vocab))

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for sentence, tags in train_data:
            model.zero_grad()

            sentence_in = prepare_sequence(sentence, token_vocab)

            targets = prepare_sequence(tags, tag_vocab)

            tags_scores = model(sentence_in)

            loss = loss_function(tags_scores, targets)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "output/model.pth")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", default="./data/CrossNER/conll2003/train.txt", help="Path to the data file in CoNLL format")
    parser.add_argument("-e", "--epochs", default=10, type=int,
                        help="Number of epochs to train the model")
    parser.add_argument("-lr", "--learning_rate", default=0.001,
                        type=float, help="Learning rate for the optimizer")

    args = parser.parse_args()

    train_dataset = load_data(args.dataset)

    train(train_dataset, args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
    # args

    # with torch.no_grad():   reuters_test = load_data("./data/CrossNER/conll2003/dev.txt")

    #     for sentence, tags in reuters_test:
    #         sentence_in = prepare_sequence(sentence, token_vocab)
    #         targets = prepare_sequence(tags, tag_vocab)

    #         tag_scores = model(sentence_in)
    #         _, indices = torch.max(tag_scores, 1)

    #         indices = indices.numpy()
    #         targets = targets.numpy()

    #         acc += sum([1 for i in range(len(indices))
    #                    if indices[i] == targets[i]]) / len(indices)

    #     acc /= len(reuters_test)
    #     print(acc)
