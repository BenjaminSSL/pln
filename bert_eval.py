import json
import pickle
from util.data import load_data
from transformers import BertTokenizerFast
from transformers import AutoModelForTokenClassification
from transformers import pipeline
import datasets
from bert import CustomTokenizerAligner
from multiprocessing import Process, Queue

from util.map_labels import map_list
from util.metric import compute_accuracy, compute_metrics, compute_tag_accuracy

# https://github.com/rohan-paul/LLM-FineTuning-Large-Language-Models/blob/main/Other-Language_Models_BERT_related/YT_Fine_tuning_BERT_NER_v1.ipynb


TOKENIZER_PATH = "output/tokenizer"
MODEL_PATH = "output/ner_model"


def predict(pipe, eval_dataset, queue):
    labels, pred = [], []
    label_names = ["O", "B-location", "I-location", "B-organisation",
                   "I-organisation", "B-person", "I-person", "B-misc", "I-misc"]
    length = len(eval_dataset)
    for index, sentence in enumerate(eval_dataset):
        if index % 1 == 0:
            print(f"{index}/{length}")
        tokens = sentence["tokens"]

        tags = sentence["tags"]

        tags = [label_names[tag] for tag in tags]

        labels.append(tags)

        predictions = pipe(tokens)

        label_predictions = []

        for prediction in predictions:

            if len(prediction) > 0:
                label_predictions.append(prediction[0].get("entity"))
            else:
                label_predictions.append("O")

        pred.append(label_predictions)

    queue.put({"labels": labels, "pred": pred})


def main():

    eval_dataset = load_data(
        "./data/CrossNER/conll2003/train_testing.txt", as_dict=True)

    for value in eval_dataset:
        value["tags"] = map_list(value["tags"], "ai")

    label_names = ["O", "B-location", "I-location", "B-organisation",
                   "I-organisation", "B-person", "I-person", "B-misc", "I-misc"]

    id2label = {
        str(i): label for i, label in enumerate(label_names)
    }
    label2id = {
        label: str(i) for i, label in enumerate(label_names)
    }

    features = datasets.Features({"tokens": datasets.Sequence(datasets.Value(
        "string")), "tags": datasets.Sequence(datasets.ClassLabel(names=label_names))})

    config = json.load(open(MODEL_PATH+"/config.json"))
    config["id2label"] = id2label
    config["label2id"] = label2id

    json.dump(config, open(MODEL_PATH+"/config.json", "w"))
    tokenizer_fine_tuned = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained(
        "output/ner_model")

    cmt = CustomTokenizerAligner(tokenizer_fine_tuned)

    eval_dataset_loader = datasets.Dataset.from_list(
        eval_dataset, features=features)
    eval_dataset_loader = eval_dataset_loader.map(
        cmt.tokenize_and_align_labels, batched=True)

    dataset_list = []

    length = len(eval_dataset)

    split = 1

    for i in range(split):
        start = i * (length // split)

        end = (i + 1) * (length // split) if i != split - 1 else length
        eval_dataset_loader = datasets.Dataset.from_list(
            eval_dataset[start:end], features=features)
        eval_dataset_loader = eval_dataset_loader.map(
            cmt.tokenize_and_align_labels, batched=True)
        dataset_list.append(eval_dataset_loader)

    pipe = pipeline("ner", model=model_fine_tuned,
                    tokenizer=tokenizer_fine_tuned)

    queue = Queue()
    proccesses = []
    results = []

    for i in range(0, split):

        print(f"Starting process {i}")

        p = Process(target=predict, args=(
            pipe, dataset_list[i], queue))
        proccesses.append(p)
        p.start()
    print(proccesses)

    for _ in range(0, split):
        results.append(queue.get())

    print("All processes started")

    for p in proccesses:
        p.join()
    print("All processes joined")

    labels = []
    predictions = []

    for result in results:
        labels.extend(result["labels"])
        predictions.extend(result["pred"])
    print("All processes done")
    pickle.dump(labels, open("output/labels.list", "wb"))
    pickle.dump(predictions, open("output/predictions.list", "wb"))

    precision, recall, f1 = compute_metrics(labels, predictions)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    accuracy = compute_accuracy(labels, predictions)

    print(f"Accuracy: {accuracy}")

    tag_accuracy = compute_tag_accuracy(labels, predictions)

    print(f"Tag Accuracy: {tag_accuracy}")


if __name__ == "__main__":
    main()
