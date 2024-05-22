def load_data(path, as_dict=False):
    '''
    Loads the data from the file at the specified path.
    The data is returned in the following format:
    data = [([words], [tags]), ([words], [tags]), ...]
    Where each tuple represents a sentence.
    '''
    data = []
    with open(path, "r", encoding="latin-1") as f:
        tokens = []
        tags = []
        for line in f:
            line = line.strip()
            parts = line.split("\t")

            if parts == [''] and len(tokens) > 0:
                if as_dict:
                    data.append({"tokens": tokens, "tags": tags})
                else:
                    data.append((tokens, tags))
                tokens, tags = [], []

            elif len(parts) == 2:
                token = parts[0]
                tag = parts[1]
                tokens.append(token)
                tags.append(tag)
    return data
