from __future__ import absolute_import, division, print_function

import math
import codecs
import logging
from nltk import ngrams

from six.moves import range

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
        try:
            return self.positions[clean_offset]
        except:
            print(len(self.positions), clean_offset)


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
            raise Exception("Unable to join with token from other string")
        return TextRange(self.document, min(self.start, other.start), max(self.end, other.end))

    def __eq__(self, other):
        return self.document == other.document and self.start == other.start and self.end == other.end

    def __len__(self):
        return self.end-self.start


def greedy_minimum_search(a, b, compute, result_a=None, result_b=None):
    if a > b:
        a, b = b, a
    if a == b:
        return result_a or result_b or compute(a)
    result_a = result_a or compute(a)
    result_b = result_b or compute(b)
    if b == a+1:
        return result_a if result_a[0] < result_b[0] else result_b
    c = (a+b) // 2
    if result_a[0] < result_b[0]:
        return greedy_minimum_search(a, c, compute, result_a=result_a)
    else:
        return greedy_minimum_search(c, b, compute, result_b=result_b)


class FuzzySearch(object):
    def __init__(self, text):
        self.text = text
        self.ngrams = {}
        for i, ngram in enumerate(ngrams(' ' + text + ' ', 3)):
            if ngram in self.ngrams:
                ngram_bucket = self.ngrams[ngram]
            else:
                ngram_bucket = self.ngrams[ngram] = []
            ngram_bucket.append(i)

    @staticmethod
    def _find_best_neighbour_token(original_text, center_token):
        tokens = [center_token.prev_token(), center_token, center_token.next_token()]
        tokens = map(lambda t: (t, levenshtein(t.get_text(), original_text)), tokens)
        tokens = sorted(tokens, key=lambda item: item[1])
        return tokens[0][0]

    def _find_best_in_interval(self, look_for, start, stop, snap_token=True, stretch_factor=1/3):
        found_best = False

        best_distance, best_interval = \
            greedy_minimum_search(start,
                                  stop,
                                  lambda p: (levenshtein(self.text[p:p + len(look_for)], look_for), (p, p + len(look_for))))
        best_start, best_end = best_interval

        stretch_radius = int(len(look_for) * stretch_factor)

        best_distance, best_interval = \
            greedy_minimum_search(best_start-stretch_radius,
                                  best_start+stretch_radius,
                                  lambda p: (levenshtein(self.text[p:best_end], look_for), (p, best_end)))
        best_start, best_end = best_interval

        best_distance, best_interval = \
            greedy_minimum_search(best_end-stretch_radius,
                                  best_end+stretch_radius,
                                  lambda p: (levenshtein(self.text[best_start:p], look_for), (best_start, p)))
        best_start, best_end = best_interval

        if snap_token:
            first_original_token = TextRange.token_at(self.text, best_start)
            if len(first_original_token) == 0:
                first_original_token = first_original_token.next_token()
            first_original_token = FuzzySearch._find_best_neighbour_token(
                TextRange.token_at(look_for, 0).get_text(),
                first_original_token)

            last_original_token = TextRange.token_at(self.text, best_end-1)
            if len(last_original_token) == 0:
                last_original_token = last_original_token.prev_token()
            last_original_token = FuzzySearch._find_best_neighbour_token(
                TextRange.token_at(look_for, len(look_for)-1).get_text(),
                last_original_token)

            best_interval = first_original_token+last_original_token

        best_distance = levenshtein(best_interval.get_text(), look_for)
        return best_distance, best_interval

    def find_best(self,
                  look_for,
                  start=0,
                  stop=-1,
                  max_candidates=10,
                  candidate_threshold=0.8,
                  snap_token=True,
                  stretch_factor=1/3):
        stop = len(self.text) if stop < 0 else stop
        window_size = len(look_for)
        windows = {}
        for i, ngram in enumerate(ngrams(' ' + look_for + ' ', 3)):
            if ngram in self.ngrams:
                ngram_bucket = self.ngrams[ngram]
                for occurrence in ngram_bucket:
                    if occurrence < start or occurrence > stop:
                        continue
                    window = occurrence // window_size
                    windows[window] = (windows[window] + 1) if window in windows else 1
        candidate_windows = sorted(windows.keys(), key=lambda w: windows[w], reverse=True)
        best_interval = None
        best_distance = -1
        last_window_grams = 0.1
        for window in candidate_windows[:max_candidates]:
            if windows[window] / last_window_grams < candidate_threshold:
                break
            last_window_grams = windows[window]
            interval_start = max(start,              int((window-0.5)*window_size))
            interval_stop  = min(stop-len(look_for), int((window+0.5)*window_size))
            interval_distance, interval = self._find_best_in_interval(look_for,
                                                                      interval_start,
                                                                      interval_stop,
                                                                      snap_token=snap_token,
                                                                      stretch_factor=stretch_factor)
            if not best_interval or interval_distance < best_distance:
                best_interval = interval
                best_distance = interval_distance
        return best_interval, best_distance


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
