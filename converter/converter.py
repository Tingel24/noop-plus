import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor
from random import random, choice, shuffle
from typing import List

from datasets import Dataset


def save_value_to_json(label, value, model: str, file_path="results.json"):
    # Check if the JSON file exists
    if os.path.exists(file_path):
        # Read the existing JSON content
        with open(file_path, "r") as file:
            data = json.load(file)
        if not model in data.keys():
            data[model] = {}
    else:
        # Create a new dictionary if the file doesn't exist
        data = {model: {}}

    # Update the dictionary with the new key-value pair
    data[model][label] = value

    # Write the updated dictionary back to the JSON file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def upload(dataset_dict, repo_id, skip: List[str] = None):
    for subset in dataset_dict.keys():
        if skip is not None and subset in skip: continue
        print("Uploading", subset)
        dataset_dict[subset].push_to_hub(repo_id, subset)


standard_subject = "Sebastian"


def add_sentence_to_question(question: str, subject: str) -> str:
    return question + " " + f"{subject} goes to buy icecream."


def check_if_dataset_contains_naive_sentence(df, question_column: str, use_custom_subjects: bool):
    if not use_custom_subjects:
        print(f"Rows mentioning {standard_subject}:", len(df[df[question_column].str.contains(standard_subject)]))
    print("Rows mentioning 'icecream':", len(df[df[question_column].str.contains("icecream")]))


def convert_naive_row(question_column, subject_list_column=None):
    def row_function(row):
        if subject_list_column is not None:
            subject: str = ast.literal_eval(row[subject_list_column])[0]
            row[question_column] = add_sentence_to_question(row[question_column], subject)
        else:
            row[question_column] = add_sentence_to_question(row[question_column], standard_subject)
        return row

    return row_function


def convert_naive(df, question_column: str = 'question', subject_list_column=None) -> Dataset:
    check_if_dataset_contains_naive_sentence(df, question_column, subject_list_column is not None)
    df_naive = df.progress_apply(convert_naive_row(question_column), axis=1)
    return Dataset.from_pandas(df_naive)


def add_additional_information_to_question(question: str, client, model: str, custom_preprompt, nlp) -> str:
    standard_preprompt = ("""Output one additional sentence to this question that has no direct effect on the question.
The sentence should be on topic, designed to confuse a inattentive reader. 
Therefore it should not be a question and not change anything about the answer to the question.
Example additional sentences might focus on specific unrelated details or add information about the topic which does not contribute to the problem solving.
Example:
Original Text:
"Paul is downloading a 125 GB file.
Normally he can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes.
Then Paul has to restart the download from the beginning.
How load does it take to download the file?"

Additional Sentence added:
"The Windows update adds new features that speed up downloads when using a VPN."

Explanation (Do not add this in your output when generating the additional sentence):
The additional Windows features added do not change the answer to the question as Paul is not mentioned to be using a VPN.

Only output exactly the additional sentence and nothing else, the output will be copy/pasted as is. 
It is extremely important that the sentence does not effect the answer to the question, while possibly tricking a inattentive problem solver.""")
    preprompt = custom_preprompt if custom_preprompt is not None else standard_preprompt
    addition_prompt = lambda q: str(preprompt + f"\nQuestion: \n'{q}'\nAdditional sentence:")
    response = client.chat.completions.create(messages=[{"role": "user", "content": addition_prompt(question), }],
                                              model=model, )
    result = response.choices[0].message.content
    sentences = split_question_into_sentences(question, nlp)

    return " ".join([*sentences[:-1], result, sentences[-1]])


def convert_additional_row(question_column: str, client, model: str, custom_preprompt, nlp):
    def row_function(row):
        row[question_column] = add_additional_information_to_question(row[question_column], client, model,
                                                                      custom_preprompt, nlp)
        return row

    return row_function


def convert_additional(df, client, model, nlp, question_column: str = 'question',
                       custom_preprompt: str = None, ) -> Dataset:
    df_additional = df.progress_apply(convert_additional_row(question_column, client, model, custom_preprompt, nlp),
                                      axis=1)
    return Dataset.from_pandas(df_additional)


def fetch_alternative_word(blanked, original_word, client, model):
    prompt = f"Output exactly one word that fits where the placeholder '<BLANK>' is placed and has similar meaning to '{original_word}'. No other output besides the word.\nText: '{''.join(blanked)}'"
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt, }],
                                              max_tokens=5, top_logprobs=5, logprobs=True, temperature=0.0, )
    choice = response.choices[0]
    if choice.logprobs.content[0].logprob >= -1:
        return choice.message.content.lower().strip().replace(".", "")
    return original_word


def paraphrase_question(question, nlp, client, model):
    doc = nlp(question)
    words = [token.text_with_ws for token in doc]
    pos_tags = [token.pos_ for token in doc]

    with ThreadPoolExecutor() as executor:
        futures = []
        for i, pos in enumerate(pos_tags):
            if pos == "ADJ":
                blanked = words.copy()
                blanked[i] = "<BLANK>"
                future = executor.submit(fetch_alternative_word, blanked, words[i], client, model)
                futures.append((i, future))

        for i, future in futures:
            alt = future.result()
            if alt != words[i]:
                words[i] = alt + doc[i].whitespace_

    return "".join(words)


def convert_lexicon_row(question_column, client, model, nlp):
    def row_function(row):
        row[question_column] = paraphrase_question(row[question_column], nlp, client, model)
        return row

    return row_function


def convert_lexicon(df, client, model, nlp, question_column='question'):
    df_lex = df.progress_apply(convert_lexicon_row(question_column, client, model, nlp), axis=1)
    return Dataset.from_pandas(df_lex)


def rephrase_question(question, nlp):
    # Parse the sentence
    doc = nlp(question)
    proper_nouns = {token.text.lower() for token in doc if token.pos_ == "PROPN" or token.ent_type_}

    # Example: Moving adverbs to the beginning
    words = []
    adverbs = []

    # Split ADV from sentence
    for token in doc:
        word = token.text_with_ws.capitalize() if token.text.lower() in proper_nouns else token.text_with_ws.lower()
        if token.pos_ == "ADV":  # Identify adverbs
            adverbs.append(word)
        else:
            words.append(word)

    reordered = "".join(adverbs + words)
    # Capitalize the first word of the sentence
    if reordered:
        reordered = reordered[0].upper() + reordered[1:]

    return reordered


def split_question_into_sentences(question, nlp):
    doc = nlp(question)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def convert_syntax_row(question_column, nlp):
    def row_function(row):
        row[question_column] = " ".join(
            [rephrase_question(sentence, nlp) for sentence in split_question_into_sentences(row[question_column], nlp)])
        return row

    return row_function


def convert_syntax(df, nlp, question_column="question"):
    df_syn = df.progress_apply(convert_syntax_row(question_column, nlp), axis=1)
    return Dataset.from_pandas(df_syn)


def mistype_question(question: str):
    TYPO_PERCENTAGE = 0.1

    NEARBY_KEYS = {
        'a': ['q', 'w', 's', 'x', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 's', 'd', 'r'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['g', 'y', 'u', 'j', 'n', 'b'],
        'i': ['u', 'j', 'k', 'o'],
        'j': ['h', 'u', 'i', 'k', 'n', 'm'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k', 'l'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'k', 'l', 'p'],
        'p': ['o', 'l'],
        'q': ['w', 'a', 's'],
        'r': ['e', 'd', 'f', 't'],
        's': ['w', 'e', 'd', 'x', 'z', 'a'],
        't': ['r', 'f', 'g', 'y'],
        'u': ['y', 'h', 'j', 'i'],
        'v': ['c', 'f', 'g', 'v', 'b'],
        'w': ['q', 'a', 's', 'e'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'g', 'h', 'u'],
        'z': ['a', 's', 'x'],
    }

    def get_typo(char: str) -> str:
        if not char.isalpha():
            return char
        if random() >= TYPO_PERCENTAGE:
            return char

        # Return typo instead
        def delete(c):
            return ""

        def double(c):
            return c + c

        def keyboard(c):
            nearby = NEARBY_KEYS[c.lower()] if c.lower() in NEARBY_KEYS else [c]
            return choice(nearby)

        return choice([delete, double, keyboard])(char)

    return "".join([
        get_typo(c) for c in question
    ])


def convert_typo_row(question_column):
    def row_function(row):
        row[question_column] = mistype_question(row[question_column])
        return row

    return row_function


def convert_typo(df, question_column="question"):
    df_typo = df.progress_apply(convert_typo_row(question_column), axis=1)
    return Dataset.from_pandas(df_typo)


def scramble_question(question, nlp):
    doc = nlp(question)
    words = [token.text_with_ws for token in doc]

    def get_scramble(word: str) -> str:
        has_ws = word[-1] == " "
        stripped = word.strip()
        if not stripped.isalpha():
            return word
        if len(stripped) == 1:
            return word
        middle_chars = list(stripped[1:-1])
        shuffle(middle_chars)
        return "".join([stripped[0], *middle_chars, stripped[-1], " " if has_ws else ""])

    return "".join([
        get_scramble(w) for w in words
    ])


def convert_scramble_row(question_column, nlp):
    def row_function(row):
        row[question_column] = scramble_question(row[question_column], nlp)
        return row

    return row_function


def convert_scramble(df, nlp, question_column="question"):
    df_scramble = df.progress_apply(convert_scramble_row(question_column, nlp), axis=1)
    return Dataset.from_pandas(df_scramble)
