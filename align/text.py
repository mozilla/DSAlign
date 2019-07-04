from __future__ import absolute_import, division, print_function

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

    def filter(self, string):
        new_string = ''
        for c in string:
            if self.has_label(c):
                new_string += c
        return new_string

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


class TextCleaner:
    def __init__(self, original_text, alphabet, to_lower=True, normalize_space=True):
        self.original_text = original_text
        clean_text = original_text
        if to_lower:
            clean_text = clean_text.lower()
        if normalize_space:
            clean_text = ' '.join(clean_text.split())
        self.clean_text = alphabet.filter(clean_text)

    def get_original_offset(self, clean_offset):
        return clean_offset


class LevenshteinSearch:
    def __init__(self, text):
        self.text = text
        self.ngrams = {}
        for i, ngram in enumerate(ngrams(' ' + text + ' ', 3)):
            if ngram in self.ngrams:
                ngram_bucket = self.ngrams[ngram]
            else:
                ngram_bucket = self.ngrams[ngram] = []
            ngram_bucket.append(i)

    def find_best(self, look_for, start=0, stop=-1, threshold=0):
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
        found_best = False
        best_distance = -1
        best_offset = -1
        best_len = -1
        for window in candidate_windows[0:4]:
            for offset in range(int((window-0.5)*window_size), int((window+0.5)*window_size)):
                distance = levenshtein(self.text[offset:offset + len(look_for)], look_for)
                if not found_best or distance < best_distance:
                    found_best = True
                    best_distance = distance
                    best_offset = offset
                    best_len = len(look_for)
        return best_distance, best_offset, best_len


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
