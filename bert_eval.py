import json
from util.data import load_data
from transformers import BertTokenizerFast
from transformers import AutoModelForTokenClassification
from transformers import pipeline
import datasets

from util.map_labels import map_list

# https://github.com/rohan-paul/LLM-FineTuning-Large-Language-Models/blob/main/Other-Language_Models_BERT_related/YT_Fine_tuning_BERT_NER_v1.ipynb


def tokenize_and_align_labels(data):

    tokenizer_fine_tuned = BertTokenizerFast.from_pretrained(
        "output/tokenizer")
    tokenized_inputs = tokenizer_fine_tuned(
        data["tokens"],  is_split_into_words=True)

    labels = []
    for i, label in enumerate(data[f"tags"]):
        # Map tokens to their respective word.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.

            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


testset = load_data("./data/CrossNER/ai/benjamin.txt", as_dict=True)

for value in testset:
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


config = json.load(open("output/ner_model/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("output/ner_model/config.json", "w"))
tokenizer_fine_tuned = BertTokenizerFast.from_pretrained("output/tokenizer")
model_fine_tuned = AutoModelForTokenClassification.from_pretrained(
    "output/ner_model")

eval_dataset_loader = datasets.Dataset.from_list(
    testset, features=features)
eval_dataset_loader = eval_dataset_loader.map(
    tokenize_and_align_labels, batched=True)


pipe = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer_fine_tuned)

acc = 0


def calculate_start_end_index(tokens, tags):

    tags_locations = []
    marker = 0
    for index, tag in enumerate(tags):

        start = marker
        end = start + len(tokens[index])
        if tag != "O":
            tags_locations.append({"start": start, "end": end, "entity": tag})

        # Update the marker and add 1 for the space
        marker = end + 1

    return tags_locations


total = 0
for data in eval_dataset_loader:
    labels = data["labels"]

    for label in labels:
        if label != -100 and label != 0:
            total += 1

print("total", total)


for sentence in eval_dataset_loader:

    tokens = sentence["tokens"]
    tags = sentence["tags"]
    labels = sentence["labels"]
    print(" ".join(tokens))
    prediction = pipe(" ".join(tokens))
    print(prediction)

    for pred in prediction:

        if labels[pred["index"]] == -100:
            continue

        print(pred["entity"], label_names[labels[pred["index"]]])

        if pred["entity"] == label_names[labels[pred["index"]]]:

            acc += 1

print(acc / total)
