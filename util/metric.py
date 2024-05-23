

def compute_metrics(true_labels, pred_labels):
    tp = 0
    fp = 0
    fn = 0

    for true, pred in zip(true_labels, pred_labels):
        for t, p in zip(true, pred):
            if t == "O" and p == "O":
                continue
            if t == p:
                tp += 1
            else:
                if t != "O" and p != "O":
                    fn += 1
                else:
                    fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_accuracy(true_labels, pred_labels):
    total = 0
    correct = 0

    for true, pred in zip(true_labels, pred_labels):
        for t, p in zip(true, pred):
            if t == p:
                correct += 1
            if t != "O":
                total += 1

    return correct / total


def compute_tag_accuracy(true_labels, pred_labels):

    with open("temp.txt", "w+") as f:
        for t, p in zip(true_labels, pred_labels):
            f.write(f"{t} - {p}\n")

    tags = {}

    for trues, preds in zip(true_labels, pred_labels):
        for true, pred in zip(trues, preds):
            if true == pred:
                if true not in tags:
                    tags[true] = {"correct": 0, "total": 0}
                tags[true]["correct"] += 1
                tags[true]["total"] += 1
            else:
                if true not in tags:
                    tags[true] = {"correct": 0, "total": 0}
                tags[true]["total"] += 1

    tag_accuracies = {}

    for tag in tags:
        tag_accuracies[tag] = tags[tag]["correct"] / tags[tag]["total"]

    return tag_accuracies
