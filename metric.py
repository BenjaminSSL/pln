import json


def load_metric(path):
    with open(path, "r") as f:
        return json.load(f)


def main():

    categories = ["conll2003", "ai", "literature",
                  "music", "politics", "science"]

    rnn_metrics = {}

    for category in categories:
        rnn_metrics[category] = load_metric(
            './output/metrics/rnn/{}.json'.format(category))

    accuracies = get_accuracy(rnn_metrics)

    print(accuracies)

    metrics = get_metrics(rnn_metrics)

    for category in metrics:
        print("{}: p: {} r: {} f1: {} f1_span: {}".format(
            category,
            metrics[category]['precision'],
            metrics[category]['recall'],
            metrics[category]['f1'],
            metrics[category]['span_f1']
        ))

    bert_metrics = {}

    for category in categories:
        bert_metrics[category] = load_metric(
            './output/metrics/bert/{}.json'.format(category))

    accuracies = get_accuracy(bert_metrics)

    print(accuracies)

    metrics = get_metrics(bert_metrics)

    for category in metrics:
        print("{}: p: {} r: {} f1: {} f1_span: {}".format(
            category,
            metrics[category]['precision'],
            metrics[category]['recall'],
            metrics[category]['f1'],
            metrics[category]['span_f1']
        ))


def get_accuracy(metrics):

    accuracies = {}

    for category in metrics:
        accuracies[category] = metrics[category]["accuracy"]

    return accuracies


def get_metrics(metrics):
    metricss = {}

    for category in metrics:
        metricss[category] = {
            "precision": metrics[category]["precision"],
            "recall": metrics[category]["recall"],
            "f1": metrics[category]["f1"],
            "span_f1": metrics[category]["span_f1"]
        }

    return metricss


if __name__ == "__main__":
    main()
