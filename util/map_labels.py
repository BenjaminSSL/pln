ai_to_reuters = {
    "researcher": "person",
    "country": "location",
    "person": "person",
    "location": "location",
    "organisation": "organisation",
    "misc": "misc"
}
politics_to_reuters = {
    "politician": "person",
    "country": "location",
    "political party": "organisation",
    "person": "person",
    "location": "location",
    "organisation": "organisation",
    "misc": "misc"

}
natural_science_to_reuters = {
    "scientist": "person",
    "university": "organisation",
    "country": "location",
    "person": "person",
    "location": "location",
    "organisation": "organisation",
    "misc": "misc"
}
music_to_reuters = {
    "band": "organisation",
    "musical artist": "person",
    "country": "location",
    "person": "person",
    "location": "location",
    "organisation": "organisation",
    "misc": "misc"
}
literature_to_reuters = {
    "writer": "person",
    "country": "location",
    "person": "person",
    "location": "location",
    "organisation": "organisation",
    "misc": "misc"
}


def map_token(token, map_type):
    found = False

    if map_type == "ai":
        found = True
        if token in ai_to_reuters:
            return ai_to_reuters[token]
    elif map_type == "politics":
        found = True
        if token in politics_to_reuters:
            return politics_to_reuters[token]
    elif map_type == "science":
        found = True
        if token in natural_science_to_reuters:
            return natural_science_to_reuters[token]
    elif map_type == "music":
        found = True
        if token in music_to_reuters:

            return music_to_reuters[token]
    elif map_type == "literature":
        found = True
        if token in literature_to_reuters:
            return literature_to_reuters[token]

    if not found:
        raise ValueError("Token {} not found in map".format(token))
    return "misc"


def map_list(tokens, map_type):
    mapped = []

    for token in tokens:

        if token == "O":
            mapped.append("O")
            continue
        prefix = token.split("-")[0]
        label = token.split("-")[1]
        token = map_token(label, map_type)

        mapped.append("{}-{}".format(prefix, token))

    return mapped
