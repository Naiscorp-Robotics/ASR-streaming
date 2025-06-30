import re
from typing import List

from lightspeech.layers.ngram import WittenBellInterpolated, Sym


class OOVRecognizer:
    def __init__(
        self,
        oov_filepath: str,
        vocabulary: List[str],
        max_order: int = 5,
        max_edit_distance: int = 5,
        start_oov: str = "<<",
        end_oov: str = ">>",
    ):
        self.vocabulary = vocabulary
        self.vocab_size = len(self.vocabulary)
        self.lm_ngram = WittenBellInterpolated(max_order)
        self.spell_checker = SymSpell(max_edit_distance)

        self.max_order = max_order
        self.max_edit_distance = max_edit_distance
        self.start_oov = start_oov
        self.end_oov = end_oov

        oov_words, oov_soundlikes = self.parse_oov_data(oov_filepath)

        # Training OOV Language Model Ngram
        oov_characters = ([start_oov] + list(w) + [end_oov] for w in oov_words)
        oov_ngrams = (everygrams(c, max_len=max_order) for c in oov_characters)
        self.lm_ngram.fit(oov_ngrams, self.vocabulary)

        # Update vocabulary of OOV Spelling Correction
        for oov_word in oov_words:
            self.spell_checker.create_dictionary_entry(oov_word, 1)

        # Initialize OOV Sound-Like Capture
        self.oov_soundlikes = oov_soundlikes

    def __call__(
        self,
        context: List[str],
        state: Optional[str] = None,
        offset: Optional[int] = 1,
    ):
        context = context[-self.max_order + 1 :]
        context = [self.vocabulary[idx - offset] for idx in context]

        if context[-1] == self.start_oov:
            state = "on"
        if (context[-1] == self.end_oov) or (state is None):
            state = "off"

        if state == "on":
            output = self.predict(tuple(context))
        else:
            avg_prob = 1.0 / self.vocab_size
            output = torch.empty(self.vocab_size).fill_(avg_prob)

        return output, state

    @lru_cache(maxsize=16)
    def predict(self, context: Tuple[str]) -> List[float]:
        # filter tokens which the probability is greater than zero
        for offset in range(self.max_order - 1):
            context_ = context[offset:]
            valid_tokens = self.lm_ngram.context_counts(context_).keys()
            if len(valid_tokens) > 0:
                break

        # calculate the probability of tokens, default score is zero
        output = torch.zeros(self.vocab_size)
        for token in valid_tokens:
            idx = self.vocabulary.index(token)
            output[idx] = self.lm_ngram.score(token, context)

        # normalize output to ensure sum of probability is one
        output += (1.0 - output.sum()) / self.vocab_size

        return output

    def parse_oov_data(
        self,
        oov_filepath: str,
    ) -> Tuple[List[str], Dict[str, str]]:

        with open(oov_filepath) as f:
            oov_datas = f.read().split("\n")
            oov_datas = oov_datas[:-1]

        oov_words, oov_soundlikes = [], []
        for data in oov_datas:
            columns = data.split("|")
            oov_word = columns[0].strip()
            oov_words.append(oov_word)

            if len(columns) == 2:
                soundlike = columns[1].split(",")
                soundlike = [sound.strip() for sound in soundlike]
                oov_soundlike = [(sound, oov_word) for sound in soundlike]
                oov_soundlikes.extend(oov_soundlike)

        oov_words = sorted(set(oov_words))
        oov_soundlikes = sorted(set(oov_soundlikes), reverse=True)

        return oov_words, oov_soundlikes

    def correct_spelling(self, sentence: str) -> str:
        # extract OOV words within oov tag and remove Vietnamese words
        oov_text = re.sub(r"\w+ ", " ", sentence.strip() + " ")
        oov_text = oov_text.strip()
        if not oov_text:
            return sentence

        # split sentence into a number of oov segments then
        # select the highest probability suggestion to replace original OOV
        oov_segments = re.split(" {2,}", oov_text)
        for segment in oov_segments:
            segment_ = re.sub(f"{self.start_oov}|{self.end_oov}", "", segment)
            suggestion = self.spell_checker.lookup_compound(
                segment_, self.max_edit_distance
            )
            suggestion = suggestion[0].term

            # insert oov tag into oov words
            oov_words = suggestion.split()
            oov_words = [self.start_oov + w + self.end_oov for w in oov_words]
            suggestion = " ".join(oov_words)
            sentence = sentence.replace(segment, suggestion)

        return sentence

    def capture_soundlike(self, sentence) -> str:
        for soundlike, oov in self.oov_soundlikes:
            oov = self.start_oov + oov + self.end_oov
            # sentence = sentence.replace(soundlike, oov)  # TODO: use regex
            sentence = re.sub(rf"\b{soundlike}\b", oov, sentence)
        return sentence
