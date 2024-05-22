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

    if map_type == "ai":
        if token in ai_to_reuters:
            return ai_to_reuters[token]
    elif map_type == "politics":
        if token in politics_to_reuters:
            return politics_to_reuters[token]
    elif map_type == "natural_science":
        if token in natural_science_to_reuters:
            return natural_science_to_reuters[token]
    elif map_type == "music":
        if token in music_to_reuters:
            return music_to_reuters[token]
    elif map_type == "literature":
        if token in literature_to_reuters:
            return literature_to_reuters[token]
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

        mapped.append(f"{prefix}-{token}")
    print(mapped)
    return mapped
