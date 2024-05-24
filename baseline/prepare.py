import pickle


class IOB2:

    def __init__(self, path):
        self.path = path

    def load(self):
        data = []
        with open(self.path, "r", encoding="latin-1") as file:
            words = []
            tags = []
            for line in file:
                line = line.strip()
                if line.startswith("#"):
                    continue
                parts = line.split("\t")

                if parts == ['']:
                    data.append((words, tags))
                    words, tags = [], []
                else:
                    word = parts[1]
                    tag = parts[2]
                    words.append(word)
                    tags.append(tag)

        return data


def main():
    print("Extracting data from IOB2 files")
    train = IOB2("../data/ewt/en_ewt-ud-train.iob2").load()
    dev = IOB2("../data/ewt/en_ewt-ud-dev.iob2").load()
    test = IOB2("../data/ewt/en_ewt-ud-test-masked.iob2").load()

    with open("./ewt.train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open("./ewt.dev.pkl", "wb") as f:
        pickle.dump(dev, f)
    with open("./ewt.test.pkl", "wb") as f:
        pickle.dump(test, f)

    print("Data extracted and saved to pickles")


if __name__ == "__main__":
    main()
