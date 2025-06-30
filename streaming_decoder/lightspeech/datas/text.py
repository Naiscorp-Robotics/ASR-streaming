import re
from typing import List, Dict
from importlib_resources import files


DELIMITER = "▁"
VOWELS = "aăâeêioôơuưy"
TONE_CHARS = "àằầèềìòồờùừỳáắấéếíóốớúứýảẳẩẻểỉỏổởủửỷạặậẹệịọộợụựỵãẵẫẽễĩõỗỡũữỹ"
TONE_MARKS = ["1_", "2_", "3_", "4_", "5_"]
SPECIAL_SUBWORDS = [
    "uôc",
    "uych",
    "uyn",
    "uynh",
    "uyp",
    "uyt",
    "uyên",
    "uyêt",
    "i",
    "in",
    "iêt",
    "iêu",
    "iêng",
]


def build_vocab() -> List[str]:
    vocab = files("lightspeech.corpus").joinpath("vocab.txt")
    vocab = vocab.read_text().split("\n")
    return vocab


def build_lexicon() -> Dict[str, List[str]]:
    lexicon = files("lightspeech.corpus").joinpath("lexicon.txt")
    lexicon = lexicon.read_text().split("\n")
    lexicon = [line.split("\t", 1) for line in lexicon]
    lexicon = {item[0]: item[1].split(" ") for item in lexicon}
    return lexicon


def refactor_tone_mark(word: str) -> str:

    pattern = "|".join(list(TONE_CHARS))
    tones = re.findall(pattern, word)

    # remove tone mark in vowel if any
    for t in set(tones):
        vowel = VOWELS[TONE_CHARS.index(t) % len(VOWELS)]
        word = word.replace(t, vowel)

    # extract the first tone mark in word
    tone_mark = ""
    if len(tones) != 0:
        tone_mark = TONE_CHARS.index(tones[0]) // len(VOWELS)
        tone_mark = TONE_MARKS[tone_mark]

    return f"{word}{tone_mark}"


def tokenize(
    sentence: str, vocab: List[str], lexicon: Dict[str, List[str]]
) -> List[str]:

    # remove invalid characters
    sentence = re.sub(r"[^\w\s<>]", "", sentence)
    sentence = re.sub(r"\s+", "|", sentence)
    sentence = sentence.lower().strip("|")

    # add DELIMITER to split OOV into list of characters when tokenizing
    words = sentence.split("|")
    for word in set(words):
        if word not in lexicon:
            new_word = "<<" + DELIMITER.join(list(word)) + ">>"
            sentence = re.sub(rf"\b{word}\b", new_word, sentence)

    # add DELIMITER to fix word splits prefixed with "qu" or "gi"
    special_words = re.findall(r"\bgi\w*\b|\bqu\w+\b", sentence)
    for word in set(special_words):
        _word = re.sub("|".join(TONE_MARKS), "", refactor_tone_mark(word))
        if _word[1:] in SPECIAL_SUBWORDS:
            new_word = word[0] + DELIMITER + word[1:]
            sentence = re.sub(rf"\b{word}\b", new_word, sentence)

    # split sentence into list of subwords
    patterns = "|".join(map(re.escape, sorted(vocab, reverse=True)))
    tokens = re.findall(patterns, sentence)

    return tokens
