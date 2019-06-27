from __future__ import absolute_import, division, print_function

import codecs
import logging

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


def minimal_distance(search_in, search_for, start=0, threshold=0):
    best_distance = 1000000000
    best_offset = -1
    best_len = -1
    window = 10
    rough_acceptable_distance = int(1.5 * window)
    acceptable_distance = int(len(search_for) * threshold)
    stop = len(search_in)-len(search_for)
    for rough_offset in range(start, stop, window):
        rough_distance = levenshtein(search_in[rough_offset:rough_offset+len(search_for)], search_for)
        if rough_distance < rough_acceptable_distance:
            for offset in range(rough_offset-window, rough_offset+window, 1):
                distance = levenshtein(search_in[offset:offset+len(search_for)], search_for)
                if distance < best_distance:
                    best_distance = distance
                    best_offset = offset
                    best_len = len(search_for)
            if best_distance <= acceptable_distance:
                return best_distance, best_offset, best_len
    return -1, 0, 0
