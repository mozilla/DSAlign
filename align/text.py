from __future__ import absolute_import, division, print_function

import codecs
from six.moves import range
from collections import Counter
from utils import enweight


class Alphabet(object):
    def __init__(self, config_file):
        self._config_file = config_file
        self._label_to_str = []
        self._str_to_label = {}
        self._size = 0
        with codecs.open(config_file, 'r', 'utf-8') as fin:
            for line in fin:
                if line[0:2] == '\\#':
                    line = '#\n'
                elif line[0] == '#':
                    continue
                self._label_to_str += line[:-1]  # remove the line ending
                self._str_to_label[line[:-1]] = self._size
                self._size += 1

    def string_from_label(self, label):
        return self._label_to_str[label]

    def has_label(self, string):
        return string in self._str_to_label

    def label_from_string(self, string):
        try:
            return self._str_to_label[string]
        except KeyError as e:
            raise KeyError(
                '''ERROR: Your transcripts contain characters which do not occur in data/alphabet.txt! Use util/check_characters.py to see what characters are in your {train,dev,test}.csv transcripts, and then add all these to data/alphabet.txt.'''
            ).with_traceback(e.__traceback__)

    def decode(self, labels):
        res = ''
        for label in labels:
            res += self.string_from_label(label)
        return res

    def size(self):
        return self._size

    def config_file(self):
        return self._config_file


class TextCleaner(object):
    def __init__(self, original_text, alphabet, to_lower=True, normalize_space=True, dashes_to_ws=True):
        self.original_text = original_text
        prepared_text = original_text.lower() if to_lower else original_text
        cleaned = []
        self.positions = []
        ws = False
        for position, c in enumerate(prepared_text):
            if dashes_to_ws and c == '-' and not alphabet.has_label('-'):
                c = ' '
            if normalize_space and c.isspace():
                if ws:
                    continue
                else:
                    ws = True
                    c = ' '
            if not alphabet.has_label(c):
                continue
            if not c.isspace():
                ws = False
            cleaned.append(c)
            self.positions.append(position)
        self.clean_text = ''.join(cleaned)

    def get_original_offset(self, clean_offset):
        if clean_offset == len(self.positions):
            return self.positions[-1]+1
        return self.positions[clean_offset]


class TextRange(object):
    def __init__(self, document, start, end):
        self.document = document
        self.start = start
        self.end = end

    @staticmethod
    def token_at(text, position):
        start = len(text)
        end = 0
        for step in [-1, 1]:
            pos = position
            while 0 <= pos < len(text) and not text[pos].isspace():
                if pos < start:
                    start = pos
                if pos > end:
                    end = pos
                pos += step
        return TextRange(text, start, end + 1) if start <= end else TextRange(text, position, position)

    def neighbour_token(self, direction):
        return TextRange.token_at(self.document, self.start - 2 if direction < 0 else self.end + 1)

    def next_token(self):
        return self.neighbour_token(1)

    def prev_token(self):
        return self.neighbour_token(-1)

    def get_text(self):
        return self.document[self.start:self.end]

    def __add__(self, other):
        if not self.document == other.document:
            raise Exception("Unable to add token from other string")
        return TextRange(self.document, min(self.start, other.start), max(self.end, other.end))

    def __eq__(self, other):
        return self.document == other.document and self.start == other.start and self.end == other.end

    def __len__(self):
        return self.end-self.start


def ngrams(s, size):
    """
    Lists all appearances of all N-grams of a string from left to right.
    :param s: String to decompose
    :param size: N-gram size
    :return: Produces strings representing all N-grams
    """
    window = len(s) - size
    if window < 1 or size < 1:
        if window == 0:
            yield s
        raise StopIteration
    for i in range(0, window + 1):
        yield s[i:i + size]


def weighted_ngrams(s, size, direction=0):
    """
    Lists all appearances of all N-grams of a string from left to right together with a positional weight value.
    The positional weight progresses quadratically.
    :param s: String to decompose
    :param size: N-gram size
    :param direction: Order of assigning positional weights to N-grams:
        direction < 0: Weight of first N-gram is 1.0 and of last one 0.0
        direction > 0: Weight of first N-gram is 0.0 and of last one 1.0
        direction == 0: Weight of center N-gram(s) near or equal 0, weight of first and last N-gram 1.0
    :return: Produces (string, float) tuples representing the N-gram along with its assigned positional weight value
    """
    return enweight(ngrams(s, size), direction=direction)


def similarity(a, b, direction=0, min_ngram_size=1, max_ngram_size=3, size_factor=1, position_factor=1):
    """
    Computes similarity value of two strings ranging from 0.0 (completely different) to 1.0 (completely equal).
    Counts intersection of weighted N-gram sets of both strings.
    :param a: String to compare
    :param b: String to compare
    :param direction: Order of equality importance:
        direction < 0: Left ends of strings more important to be similar
        direction > 0: Right ends of strings more important to be similar
        direction == 0: Left and right ends more important to be similar than center parts
    :param min_ngram_size: Minimum N-gram size to take into account
    :param max_ngram_size: Maximum N-gram size to take into account
    :param size_factor: Importance factor of the N-gram size (compared to the positional importance).
    :param position_factor: Importance factor of the N-gram position (compared to the size importance)
    :return: Number between 0.0 (completely different) and 1.0 (completely equal)
    """
    if len(a) < len(b):
        a, b = b, a
    ca, cb = Counter(), Counter()
    for s, c in [(a, ca), (b, cb)]:
        for size in range(min_ngram_size, max_ngram_size + 1):
            for ng, position_weight in weighted_ngrams(s, size, direction=direction):
                c[ng] += size * size_factor + position_weight * position_weight * position_factor
    score = 0
    for key in set(ca.keys()) & set(cb.keys()):
        score += min(ca[key], cb[key])
    return score / sum(ca.values())


# The following code is from: http://hetland.org/coding/python/levenshtein.py

# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>

def levenshtein(a, b):
    """
    Calculates the Levenshtein distance between a and b.
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
