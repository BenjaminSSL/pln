from argparse import ArgumentParser
import json
import os
import torch
from rnn import RNN, prepare_sequence
from util.map_labels import map_list
from util.metric import compute_accuracy, compute_metrics, compute_span_f1, compute_tag_accuracy
from util.data import load_data
import pickle


def predict(model, eval_dataset, tokens_vocab, tag_vocab):
    labels, pred = [], []

    for sentence in eval_dataset:

        tokens = sentence["tokens"]

        tags = sentence["tags"]
        labels.append(tags)

        predictions = model(prepare_sequence(tokens, tokens_vocab))

        _, predictions = torch.max(predictions, 1)

        predictions = predictions.tolist()

        predictions = [tag_vocab.getWord(idx) for idx in predictions]

        pred.append(predictions)

    print(labels[0], pred[0])
    return labels, pred


def main():
    parser = ArgumentParser()

    parser.add_argument("-d", "--dataset")
    parser.add_argument("-s", "--save", default=False, action="store_true")

    args = parser.parse_args()

    dataset = args.dataset

    eval_dataset = load_data(
        dataset, as_dict=True)

    token_vocab = pickle.load(open("output/rnn/token_vocab.pkl", "rb"))
    tag_vocab = pickle.load(open("output/rnn/tag_vocab.pkl", "rb"))

    model = RNN(len(token_vocab), len(tag_vocab))

    model.load_state_dict(torch.load("output/rnn/model.pth"))

    folder = dataset.split("/")[-2]

    if folder != "conll2003":
        for value in eval_dataset:
            value["tags"] = map_list(value["tags"], folder)

    labels, predictions = predict(model, eval_dataset, token_vocab, tag_vocab)

    precision, recall, f1 = compute_metrics(labels, predictions)

    accuracy = compute_accuracy(labels, predictions)

    tag_accuracy = compute_tag_accuracy(labels, predictions)

    compute_span_f1(labels, predictions)

    span_f1 = compute_span_f1(labels, predictions)

    if args.save:
        metrics = {
            "accuracy": accuracy,
            "tag_accuracy": tag_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "span_f1": span_f1,
        }

        if not os.path.exists("output/metrics/rnn/"):
            os.makedirs("output/metrics/rnn/")
        # Save metrics
        with open("output/metrics/rnn/{}.json".format(folder), "w+") as f:
            json.dump(metrics, f)

        # sentence = "Peter Blackburn work in Property Capital".split()
        # sentence_in = prepare_sequence(sentence, token_vocab)

        # print(sentence_in)

        # tags_scores = model(sentence_in)

        # _, predicted_tags = torch.max(tags_scores, 1)

        # print([tag_vocab.getWord(idx) for idx in predicted_tags])


if __name__ == "__main__":
    main()
