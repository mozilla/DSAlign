from nltk import ngrams
from text import levenshtein, TextRange
from utils import circulate


class FuzzySearch(object):
    def __init__(self,
                 text,
                 max_candidates=10,
                 candidate_threshold=0.8,
                 snap_token=True,
                 stretch_factor=1/3,
                 missed_penalty=1.0):
        self.text = text
        self.max_candidates = max_candidates
        self.candidate_threshold = candidate_threshold
        self.snap_token = snap_token
        self.stretch_factor = stretch_factor
        self.missed_penalty = missed_penalty
        self.word_distances = {}
        self.ngrams = {}
        for i, ngram in enumerate(ngrams(' ' + text + ' ', 3)):
            if ngram in self.ngrams:
                ngram_bucket = self.ngrams[ngram]
            else:
                ngram_bucket = self.ngrams[ngram] = []
            ngram_bucket.append(i)

    def get_missed_distance(self, missed):
        return self.word_distance('', ' '.join(missed)) * self.missed_penalty

    def find_best_token_range(self, look_for, look_in, look_in_center=None):
        if len(look_for) == 0:
            return 0, (0, 0)

        if len(look_in) == 0:
            return self.get_missed_distance(look_for), (0, 0)

        dc, i, j = self.find_similar_words(look_for,
                                           look_in,
                                           look_in_center=look_in_center)

        look_for_left = look_for[:i]
        look_in_left = look_in[:j]
        dl, il = self.find_best_token_range(look_for_left,
                                            look_in_left,
                                            look_in_center=len(look_in_left) - len(look_for_left) // 2)
        ls, le = il
        ml = look_in_left[le+1:]
        dml = self.get_missed_distance(ml)

        look_for_right = look_for[i+1:]
        look_in_right = look_in[j+1:]
        dr, ir = self.find_best_token_range(look_for_right,
                                            look_in_right,
                                            look_in_center=len(look_for_right) // 2)
        rs, re = ir
        mr = look_in_right[:rs]
        dmr = self.get_missed_distance(mr)

        start = ls
        end = j + 1 + re

        return (dl + dml + dc + dmr + dr) / (end - start), (start, end)

    def find_best_in_interval(self, look_for, start, stop):
        token_look_for = look_for.split()
        look_in_range = TextRange.token_at(self.text, start) + TextRange.token_at(self.text, stop)
        tokens_look_in = look_in_range.get_text().split()
        distance, token_range = self.find_best_token_range(token_look_for, tokens_look_in)
        token_start, token_end = token_range
        text_start = look_in_range.start + len(''.join(tokens_look_in[:token_start])) + token_start
        text_end = text_start + len(' '.join(tokens_look_in[token_start:token_end]))
        return distance, TextRange(self.text, text_start, text_end)

    def find_best(self, look_for, start=0, stop=-1):
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
        for window in candidate_windows[:self.max_candidates]:
            if windows[window] / last_window_grams < self.candidate_threshold:
                break
            last_window_grams = windows[window]
            interval_start = max(start,              int((window-0.5)*window_size))
            interval_stop  = min(stop-len(look_for), int((window+0.5)*window_size))
            interval_distance, interval = self.find_best_in_interval(look_for, start=interval_start, stop=interval_stop)
            if not best_interval or interval_distance < best_distance:
                best_interval = interval
                best_distance = interval_distance
        return best_interval, best_distance

    def word_distance(self, a, b):
        key = (a, b)
        if key in self.word_distances:
            return self.word_distances[key]
        avg_len = max(len(a), 1)
        s = self.word_distances[key] = (levenshtein(a, b) / avg_len)
        return s

    def find_similar_words(self,
                           look_for,
                           look_in,
                           look_for_center=None,
                           look_in_center=None,
                           distance_threshold=0.15,
                           len_threshold=5):
        found_len = False
        for i, wa in circulate(look_for, center=look_for_center):
            if len(wa) > len_threshold:
                found_len = True
                for j, wb in circulate(look_in, center=look_in_center):
                    d = self.word_distance(wa, wb)
                    if d < distance_threshold:
                        print('Accepted with distance %.2f: "%s" - "%s"' % (d, wa, wb))
                        return d, i, j
        if found_len:
            distance_threshold *= 1.1
            print('Trying to find word with distance threshold %.2f' % distance_threshold)
        else:
            len_threshold -= 1
            print('Trying to find word with len threshold %d' % len_threshold)
        return self.find_similar_words(look_for,
                                       look_in,
                                       distance_threshold=distance_threshold,
                                       len_threshold=len_threshold)
