from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def filter_empty_tags(true_labels, pred_labels):
    true_labels_all = []
    pred_labels_all = []

    for trues, preds in zip(true_labels, pred_labels):
        tl = []
        pl = []
        for t, p in zip(trues, preds):
            if t != "O" and p != "O":
                tl.append(t)
                pl.append(p)
        true_labels_all.append(tl)
        pred_labels_all.append(pl)

    return true_labels_all, pred_labels_all

    # true_labels = [tag for tags in true_labels for tag in tags]
    # pred_labels = [tag for tags in pred_labels for tag in tags]

    # true_labels_filtered = [t for t, p in zip(
    #     true_labels, pred_labels) if t != "O" and p != "O"]
    # pred_labels_filtered = [p for t, p in zip(
    #     true_labels, pred_labels) if t != "O" and p != "O"]

    # return true_labels_filtered, pred_labels_filtered


def compute_metrics(true_labels, pred_labels):
    true_labels, pred_labels = filter_empty_tags(true_labels, pred_labels)

    true_labels = [tag for tags in true_labels for tag in tags]
    pred_labels = [tag for tags in pred_labels for tag in tags]

    return precision_score(true_labels, pred_labels, average="weighted"), recall_score(true_labels, pred_labels, average="weighted"), f1_score(true_labels, pred_labels, average="weighted")


def toSpans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg+1, len(tags)):
                if tags[end][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])

    return spans


def compute_span_f1(true_labels, pred_labels):

    tp = 0
    fp = 0
    fn = 0
    for true, pred in zip(true_labels, pred_labels):
        true_spans = toSpans(true)
        pred_spans = toSpans(pred)
        overlap = len(true_spans.intersection(pred_spans))
        tp += overlap
        fp += len(pred_spans) - overlap
        fn += len(true_spans) - overlap

    precision = 0.0 if tp+fp == 0 else tp/(tp+fp)
    recall = 0.0 if tp+fn == 0 else tp/(tp+fn)

    f1 = 0.0 if precision+recall == 0.0 else 2 * (precision * recall) / \
        (precision + recall)

    return f1


def compute_accuracy(true_labels, pred_labels):

    true_labels_filtered, pred_labels_filtered = filter_empty_tags(
        true_labels, pred_labels)

    true_labels_filtered = [
        tag for tags in true_labels_filtered for tag in tags]
    pred_labels_filtered = [
        tag for tags in pred_labels_filtered for tag in tags]
    return accuracy_score(true_labels_filtered, pred_labels_filtered)


def compute_tag_accuracy(true_labels, pred_labels):

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
