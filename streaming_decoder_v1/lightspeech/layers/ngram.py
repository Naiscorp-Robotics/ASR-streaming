from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from functools import singledispatch
from itertools import chain, islice
from operator import methodcaller


@singledispatch
def _dispatched_lookup(words, vocab):
    msg = f"Unsupported type for looking up in vocabulary: {type(words)}"
    raise TypeError(msg)


@_dispatched_lookup.register(Iterable)
def _(words, vocab):
    """Look up a sequence of words in the vocabulary.
    Returns an iterator over looked up words.
    """
    return tuple(_dispatched_lookup(w, vocab) for w in words)


@_dispatched_lookup.register(str)
def _string_lookup(word, vocab):
    """Looks up one word in the vocabulary."""
    return word if word in vocab else vocab.unk_label


def _raise_unorderable_types(ordering, a, b):
    raise TypeError(
        "unorderable types: %s() %s %s()"
        % (type(a).__name__, ordering, type(b).__name__)
    )


def _count_values_gt_zero(distribution):
    """
    Count values that are greater than zero in a distribution.
    Assumes distribution is either a mapping with counts as values or
    an instance of `ConditionalFreqDist`.
    """
    as_count = (
        methodcaller("N")
        if isinstance(distribution, ConditionalFreqDist)
        else lambda count: count
    )
    # We explicitly check that values are > 0 to guard against negative counts.
    return sum(1 for sample in distribution.values() if as_count(sample) > 0)


def pad_sequence(
    sequence,
    length,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """Returns a padded sequence of items before ngram extraction."""
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (length - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (length - 1))
    return sequence


def everygrams(
    sequence,
    min_len=1,
    max_len=-1,
    pad_left=False,
    pad_right=False,
    left_pad_symbol="<<",
    right_pad_symbol=">>",
):
    """Returns all possible ngrams generated from a sequence of items."""

    # Get max_len for padding.
    if max_len == -1:
        try:
            max_len = len(sequence)
        except TypeError:
            sequence = list(sequence)
            max_len = len(sequence)

    # Pad if indicated using max_len.
    sequence = pad_sequence(
        sequence,
        max_len,
        pad_left,
        pad_right,
        left_pad_symbol,
        right_pad_symbol,
    )

    # Sliding window to store grams.
    history = list(islice(sequence, max_len))

    # Yield ngrams from sequence.
    while history:
        for ngram_len in range(min_len, len(history) + 1):
            yield tuple(history[:ngram_len])

        # Append element to history if sequence has more items.
        try:
            history.append(next(sequence))
        except StopIteration:
            pass

        del history[0]


class FreqDist(Counter):
    """
    A frequency distribution for the outcomes of an experiment.  A
    frequency distribution records the number of times each outcome of
    an experiment has occurred.  For example, a frequency distribution
    could be used to record the frequency of each word type in a
    document.  Formally, a frequency distribution can be defined as a
    function mapping from each sample to the number of times that
    sample occurred as an outcome.
    Frequency distributions are generally constructed by running a
    number of experiments, and incrementing the count for a sample
    every time it is an outcome of an experiment.
    """

    def __init__(self, samples=None):
        """
        Construct a new frequency distribution.  If ``samples`` is
        given, then the frequency distribution will be initialized
        with the count of each object in ``samples``; otherwise, it
        will be initialized to be empty.
        In particular, ``FreqDist()`` returns an empty frequency
        distribution; and ``FreqDist(samples)`` first creates an empty
        frequency distribution, and then calls ``update`` with the
        list ``samples``.
        :param samples: The samples to initialize the frequency
            distribution with.
        :type samples: Sequence
        """
        Counter.__init__(self, samples)

        # Cached number of samples in this FreqDist
        self._N = None

    def N(self):
        """
        Return the total number of sample outcomes that have been
        recorded by this FreqDist.  For the number of unique
        sample values (or bins) with counts greater than zero, use
        ``FreqDist.B()``.
        :rtype: int
        """
        if self._N is None:
            # Not already cached, or cache has been invalidated
            self._N = sum(self.values())
        return self._N

    def __setitem__(self, key, val):
        """
        Override ``Counter.__setitem__()`` to invalidate the cached N
        """
        self._N = None
        super().__setitem__(key, val)

    def __delitem__(self, key):
        """
        Override ``Counter.__delitem__()`` to invalidate the cached N
        """
        self._N = None
        super().__delitem__(key)

    def update(self, *args, **kwargs):
        """
        Override ``Counter.update()`` to invalidate the cached N
        """
        self._N = None
        super().update(*args, **kwargs)

    def setdefault(self, key, val):
        """
        Override ``Counter.setdefault()`` to invalidate the cached N
        """
        self._N = None
        super().setdefault(key, val)

    def B(self):
        """
        Return the total number of sample values (or "bins") that
        have counts greater than zero.  For the total
        number of sample outcomes recorded, use ``FreqDist.N()``.
        (FreqDist.B() is the same as len(FreqDist).)
        :rtype: int
        """
        return len(self)

    def hapaxes(self):
        """
        Return a list of all samples that occur once (hapax legomena)
        :rtype: list
        """
        return [item for item in self if self[item] == 1]

    def Nr(self, r, bins=None):
        return self.r_Nr(bins)[r]

    def r_Nr(self, bins=None):
        """
        Return the dictionary mapping r to Nr,
        the number of samples with frequency r, where Nr > 0.
        """

        _r_Nr = defaultdict(int)
        for count in self.values():
            _r_Nr[count] += 1

        # Special case for Nr[0]:
        _r_Nr[0] = bins - self.B() if bins is not None else 0

        return _r_Nr

    def _cumulative_frequencies(self, samples):
        """
        Return the cumulative frequencies of the specified samples.
        If no samples are specified, all counts are returned, starting
        with the largest.
        :param samples: the samples whose frequencies should be returned.
        :type samples: any
        :rtype: list(float)
        """
        cf = 0.0
        for sample in samples:
            cf += self[sample]
            yield cf

    def freq(self, sample):
        """
        Return the frequency of a given sample.  The frequency of a
        sample is defined as the count of that sample divided by the
        total number of sample outcomes that have been recorded by
        this FreqDist.  The count of a sample is defined as the
        number of times that sample outcome was recorded by this
        FreqDist.  Frequencies are always real numbers in the range
        [0, 1].
        :param sample: the sample whose frequency
               should be returned.
        :type sample: any
        :rtype: float
        """
        n = self.N()
        if n == 0:
            return 0
        return self[sample] / n

    def max(self):
        """
        Return the sample with the greatest number of outcomes in this
        frequency distribution.  If two or more samples have the same
        number of outcomes, return one of them; which sample is
        returned is undefined.  If no outcomes have occurred in this
        frequency distribution, return None.
        :return: The sample with the maximum number of outcomes in this
                frequency distribution.
        :rtype: any or None
        """
        if len(self) == 0:
            msg = "FreqDist must have at least 1 sample before max is defined."
            raise ValueError(msg)
        return self.most_common(1)[0][0]

    def copy(self):
        """
        Create a copy of this frequency distribution.
        :rtype: FreqDist
        """
        return self.__class__(self)

    def __add__(self, other):
        """
        Add counts from two counters.
        >>> FreqDist('abbb') + FreqDist('bcc')
        FreqDist({'b': 4, 'c': 2, 'a': 1})
        """
        return self.__class__(super().__add__(other))

    def __sub__(self, other):
        """
        Subtract count, but keep only results with positive counts.
        >>> FreqDist('abbbc') - FreqDist('bccd')
        FreqDist({'b': 2, 'a': 1})
        """
        return self.__class__(super().__sub__(other))

    def __or__(self, other):
        """
        Union is the maximum of value in either of the input counters.
        >>> FreqDist('abbb') | FreqDist('bcc')
        FreqDist({'b': 3, 'c': 2, 'a': 1})
        """
        return self.__class__(super().__or__(other))

    def __and__(self, other):
        """
        Intersection is the minimum of corresponding counts.
        >>> FreqDist('abbb') & FreqDist('bcc')
        FreqDist({'b': 1})
        """
        return self.__class__(super().__and__(other))

    def __le__(self, other):
        """
        Returns True if this frequency distribution is a subset of the other
        and for no key the value exceeds the value of the same key from
        the other frequency distribution.
        The <= operator forms partial order and satisfying the axioms
        reflexivity, antisymmetry and transitivity.
        """
        if not isinstance(other, FreqDist):
            _raise_unorderable_types("<=", self, other)
        return set(self).issubset(other) and all(
            self[key] <= other[key] for key in self
        )

    def __ge__(self, other):
        if not isinstance(other, FreqDist):
            _raise_unorderable_types(">=", self, other)
        return set(self).issuperset(other) and all(
            self[key] >= other[key] for key in other
        )

    def __lt__(self, other):
        return self <= other and not self == other

    def __gt__(self, other):
        return self >= other and not self == other

    def __repr__(self):
        """Return a string representation of this FreqDist."""
        return self.pformat()

    def pprint(self, maxlen=10, stream=None):
        """Print a string representation of this FreqDist to 'stream'."""
        print(self.pformat(maxlen=maxlen), file=stream)

    def pformat(self, max_length=10):
        """Return a string representation of this FreqDist."""
        items = self.most_common(max_length)
        items = ["{!r}: {!r}".format(*item) for item in items]
        if len(self) > max_length:
            items.append("...")
        return "FreqDist({{{0}}})".format(", ".join(items))

    def __str__(self):
        """Return a string representation of this FreqDist."""
        return f"<FreqDist with {len(self)} samples and {self.N()} outcomes>"

    def __iter__(self):
        """
        Return an iterator which yields tokens ordered by frequency.
        :rtype: iterator
        """
        for token, _ in self.most_common(self.B()):
            yield


class ConditionalFreqDist(defaultdict):
    """
    A collection of frequency distributions for a single experiment
    run under different conditions.  Conditional frequency
    distributions are used to record the number of times each sample
    occurred, given the condition under which the experiment was run.
    For example, a conditional frequency distribution could be used to
    record the frequency of each word (type) in a document, given its
    length.  Formally, a conditional frequency distribution can be
    defined as a function that maps from each condition to the
    FreqDist for the experiment under that condition.
    Conditional frequency distributions are typically constructed by
    repeatedly running an experiment under a variety of conditions,
    and incrementing the sample outcome counts for the appropriate conditions.
    """

    def __init__(self, cond_samples=None):
        """
        Construct a new empty conditional frequency distribution.  In
        particular, the count for every sample, under every condition,
        is zero.
        :param cond_samples: The samples to initialize the conditional
            frequency distribution with
        :type cond_samples: Sequence of (condition, sample) tuples
        """
        defaultdict.__init__(self, FreqDist)

        if cond_samples:
            for (cond, sample) in cond_samples:
                self[cond][sample] += 1

    def __reduce__(self):
        kv_pairs = ((cond, self[cond]) for cond in self.conditions())
        return (self.__class__, (), None, None, kv_pairs)

    def conditions(self):
        """
        Return a list of the conditions that have been accessed for
        this ``ConditionalFreqDist``.  Use the indexing operator to
        access the frequency distribution for a given condition.
        Note that the frequency distributions for some conditions
        may contain zero sample outcomes.
        :rtype: list
        """
        return list(self.keys())

    def N(self):
        """
        Return the total number of sample outcomes that have been
        recorded by this ``ConditionalFreqDist``.
        :rtype: int
        """
        return sum(fdist.N() for fdist in self.values())

    def __add__(self, other):
        """
        Add counts from two ConditionalFreqDists.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = ConditionalFreqDist()
        for cond in self.conditions():
            newfreqdist = self[cond] + other[cond]
            if newfreqdist:
                result[cond] = newfreqdist
        for cond in other.conditions():
            if cond not in self.conditions():
                for elem, count in other[cond].items():
                    if count > 0:
                        result[cond][elem] = count
        return result

    def __sub__(self, other):
        """
        Subtract count, but keep only results with positive counts.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = ConditionalFreqDist()
        for cond in self.conditions():
            newfreqdist = self[cond] - other[cond]
            if newfreqdist:
                result[cond] = newfreqdist
        for cond in other.conditions():
            if cond not in self.conditions():
                for elem, count in other[cond].items():
                    if count < 0:
                        result[cond][elem] = 0 - count
        return result

    def __or__(self, other):
        """
        Union is the maximum of value in either of the input counters.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = ConditionalFreqDist()
        for cond in self.conditions():
            newfreqdist = self[cond] | other[cond]
            if newfreqdist:
                result[cond] = newfreqdist
        for cond in other.conditions():
            if cond not in self.conditions():
                for elem, count in other[cond].items():
                    if count > 0:
                        result[cond][elem] = count
        return result

    def __and__(self, other):
        """
        Intersection is the minimum of corresponding counts.
        """
        if not isinstance(other, ConditionalFreqDist):
            return NotImplemented
        result = ConditionalFreqDist()
        for cond in self.conditions():
            newfreqdist = self[cond] & other[cond]
            if newfreqdist:
                result[cond] = newfreqdist
        return result

    def __le__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            _raise_unorderable_types("<=", self, other)
        return set(self.conditions()).issubset(other.conditions()) and all(
            self[c] <= other[c] for c in self.conditions()
        )

    def __lt__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            _raise_unorderable_types("<", self, other)
        return self <= other and self != other

    def __ge__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            _raise_unorderable_types(">=", self, other)
        return other <= self

    def __gt__(self, other):
        if not isinstance(other, ConditionalFreqDist):
            _raise_unorderable_types(">", self, other)
        return other < self

    def __repr__(self):
        """
        Return a string representation of this ``ConditionalFreqDist``.
        :rtype: str
        """
        return "<ConditionalFreqDist with %d conditions>" % len(self)


class NgramCounter:
    def __init__(self, ngram_text=None):
        """
        If `ngram_text` is specified, counts ngrams from it,
        otherwise waits for `update` method to be called explicitly.
        """

        self._counts = defaultdict(ConditionalFreqDist)
        self._counts[1] = self.unigrams = FreqDist()

        if ngram_text:
            self.update(ngram_text)

    def update(self, ngram_text):
        """
        Updates ngram counts from `ngram_text`.
        Expects `ngram_text` to be a sequence of sentences (sequences).
        Each sentence consists of ngrams as tuples of strings.
        """

        for sent in ngram_text:
            for ngram in sent:
                if not isinstance(ngram, tuple):
                    msg = f"Ngram <{ngram}> isn't a tuple, but {type(ngram)}"
                    raise TypeError(msg)

                ngram_order = len(ngram)
                if ngram_order == 1:
                    self.unigrams[ngram[0]] += 1
                    continue

                context, word = ngram[:-1], ngram[-1]
                self[ngram_order][context][word] += 1

    def N(self):
        """
        Returns grand total number of ngrams stored.
        This includes ngrams from all orders, so some duplication is expected.
        """
        return sum(val.N() for val in self._counts.values())

    def __getitem__(self, item):
        """User-friendly access to ngram counts."""
        if isinstance(item, int):
            return self._counts[item]
        elif isinstance(item, str):
            return self._counts.__getitem__(1)[item]
        elif isinstance(item, Sequence):
            return self._counts.__getitem__(len(item) + 1)[tuple(item)]

    def __str__(self):
        return "<{} with {} ngram orders and {} ngrams>".format(
            self.__class__.__name__, len(self._counts), self.N()
        )

    def __len__(self):
        return self._counts.__len__()

    def __contains__(self, item):
        return item in self._counts


class Vocabulary:
    def __init__(self, counts=None, cutoff=1, unk_label="<UNK>"):
        if cutoff < 1:
            msg = f"Cutoff value cannot be less than 1. Got: {cutoff}"
            raise ValueError(msg)

        self._cutoff = cutoff
        self.unk_label = unk_label

        self.counts = Counter()
        self.update(counts if counts is not None else "")

    @property
    def cutoff(self):
        """
        Items with count below this value
        are not considered part of vocabulary.
        """
        return self._cutoff

    def update(self, *counter_args, **counter_kwargs):
        """
        Update vocabulary counts.
        Wraps `collections.Counter.update` method.
        """
        self.counts.update(*counter_args, **counter_kwargs)
        self._len = sum(1 for __ in self)

    def lookup(self, words):
        """
        Look up one or more words in the vocabulary.
        If passed one word as a string will return that word or self.unk_label,
        otherwise will assume it was passed a sequence of words, will try to
        look each of them up and return an iterator over the looked up words.
        """
        return _dispatched_lookup(words, self)

    def __getitem__(self, item):
        return self._cutoff if item == self.unk_label else self.counts[item]

    def __contains__(self, item):
        """Only consider items with counts greater or equal to cutoff
        as being in the vocabulary."""
        return self[item] >= self._cutoff

    def __iter__(self):
        """Build on membership check define how to iterate over vocabulary."""
        return chain(
            (item for item in self.counts if item in self),
            [self.unk_label] if self.counts else [],
        )

    def __len__(self):
        """Computing size of vocabulary reflects the cutoff."""
        return self._len

    def __eq__(self, other):
        return (
            self.unk_label == other.unk_label
            and self._cutoff == other.cutoff
            and self.counts == other.counts
        )

    def __str__(self):
        return "<{} with cutoff={} unk_label='{}' and {} items>".format(
            self.__class__.__name__, self._cutoff, self.unk_label, len(self)
        )


class KneserNeySmoothing:
    def __init__(self, counts, order, discount=0.1):
        self.order = order
        self.discount = discount
        self.counts = counts

    def unigram_score(self, word):
        word_continuation_count, total_count = self._continuation_counts(word)
        return word_continuation_count / total_count

    def alpha_gamma(self, word, context):
        prefix_counts = self.counts[context]
        word_continuation_count, total_count = (
            (prefix_counts[word], prefix_counts.N())
            if len(context) + 1 == self.order
            else self._continuation_counts(word, context)
        )
        alpha = max(word_continuation_count - self.discount, 0.0)
        gamma = self.discount * _count_values_gt_zero(prefix_counts)
        alpha = alpha / total_count
        gamma = gamma / total_count
        return alpha, gamma

    def _continuation_counts(self, word, context=tuple()):
        """
        Count continuations that end with context and word.
        Continuations track unique ngram "types",
        regardless of how many instances were observed for each "type".
        This is different than raw ngram counts
        which track number of instances.
        """
        higher_order_ngrams_with_context = (
            counts
            for prefix_ngram, counts in self.counts[len(context) + 2].items()
            if prefix_ngram[1:] == context
        )
        higher_order_ngrams_with_word_count, total = 0, 0
        for counts in higher_order_ngrams_with_context:
            higher_order_ngrams_with_word_count += int(counts[word] > 0)
            total += _count_values_gt_zero(counts)
        return higher_order_ngrams_with_word_count, total


class NgramLanguageModel:
    def __init__(self, order, discount=0.1):
        self.order = order
        self.discount = discount

        self.vocab = Vocabulary()
        self.counts = NgramCounter()

    def fit(self, sequences, vocabulary):
        """Trains the model on a text."""
        self.vocab.update(vocabulary)

        sequences = (everygrams(seq, max_len=self.order) for seq in sequences)
        self.counts.update(self.vocab.lookup(seq) for seq in sequences)

        estimator = KneserNeySmoothing(self.counts, self.order, self.discount)
        setattr(self, "estimator", estimator)

    def score(self, word, context=None):
        """
        Masks out of vocab (OOV) words and computes their model score.
        For model-specific logic of calculating scores,
        see the `unmasked_score` method.
        """
        word = self.vocab.lookup(word)
        context = self.vocab.lookup(context) if context else None
        return self.unmasked_score(word, context)

    def unmasked_score(self, word, context=None):
        if not context:
            # The base recursion case: no context, we only have a unigram.
            return self.estimator.unigram_score(word)
        if not self.counts[context]:
            # It can also happen that we have no data for this context.
            # In that case we defer to the lower-order ngram.
            # This is the same as setting alpha to 0 and gamma to 1.
            alpha, gamma = 0, 1
        else:
            alpha, gamma = self.estimator.alpha_gamma(word, context)
        return alpha + gamma * self.unmasked_score(word, context[1:])
