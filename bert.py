from argparse import ArgumentParser
import os
from util.data import load_data
import datasets
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer


class CustomTokenizerAligner():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_and_align_labels(self, data):

        tokenized_inputs = self.tokenizer(
            data["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(data["tags"]):
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


class BertTagger():

    def __init__(self, labels):

        self.labels = labels
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "prajjwal1/bert-tiny")
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self.model = AutoModelForTokenClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=len(self.labels), id2label=self.id2label, label2id=self.label2id
                                                                     )

    # SORUCE: https://huggingface.co/docs/transformers/en/tasks/token_classification (with modifications)

    def prepare_labels(self, data):

        cmt = CustomTokenizerAligner(self.tokenizer)

        return cmt.tokenize_and_align_labels(data)

    def train(self, dataloader, devset_loader, epochs, learning_rate):

        training_args = TrainingArguments(
            output_dir="./output",
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            evaluation_strategy="steps",
            save_steps=500,
            save_strategy="steps",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataloader,
            eval_dataset=devset_loader,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,


        )

        trainer.train()

        if not os.path.exists("output/bert/"):
            os.makedirs("output/bert/")

        self.model.save_pretrained("output/bert/ner_model_tiny")
        self.tokenizer.save_pretrained("output/bert/tokenizer_tiny")


def train(dataset, devset, epochs, learning_rate):

    # In future, extract this from the actual dataset
    labels = ["O", "B-location", "I-location", "B-organisation",
              "I-organisation", "B-person", "I-person", "B-misc", "I-misc"]

    features = datasets.Features({"tokens": datasets.Sequence(datasets.Value(
        "string")), "tags": datasets.Sequence(datasets.ClassLabel(names=labels))})

    model = BertTagger(labels=labels)

    dataset_loader = datasets.Dataset.from_list(
        dataset, features=features)
    dataset_loader = dataset_loader.map(
        model.prepare_labels, batched=True)

    devset_loader = datasets.Dataset.from_list(
        devset, features=features)

    devset_loader = devset_loader.map(
        model.prepare_labels, batched=True)

    model.train(dataset_loader, devset_loader, epochs, learning_rate)

    print("Model trained")


def main():
    parser = ArgumentParser()

    parser.add_argument("-d", "--dataset", default="./data/CrossNER/conll2003/train.txt",
                        help="Path to the data file in CoNLL format")
    parser.add_argument("-e", "--epochs", default=10, type=int,
                        help="Number of epochs to train the model")
    parser.add_argument("-lr", "--learning_rate", default=0.001,
                        type=float, help="Learning rate for the optimizer")

    args = parser.parse_args()

    train_dataset = load_data(args.dataset, as_dict=True)

    dev_dataset = load_data("./data/CrossNER/conll2003/dev.txt", as_dict=True)

    train(train_dataset, dev_dataset, args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
