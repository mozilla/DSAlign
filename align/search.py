from collections import Counter
from text import TextRange, ngrams


class FuzzySearch(object):
    def __init__(self,
                 text,
                 max_candidates=10,
                 candidate_threshold=0.92,
                 snap_to_word=True,
                 snap_radius=0,
                 match_score=100,
                 mismatch_score=-100,
                 gap_score=-100,
                 char_similarities=None):
        self.text = text
        self.max_candidates = max_candidates
        self.candidate_threshold = candidate_threshold
        self.snap_to_word = snap_to_word
        self.snap_radius = snap_radius
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_score = gap_score
        self.char_similarities = char_similarities
        self.ngrams = {}
        for i, ngram in enumerate(ngrams(' ' + text + ' ', 3)):
            if ngram in self.ngrams:
                ngram_bucket = self.ngrams[ngram]
            else:
                ngram_bucket = self.ngrams[ngram] = []
            ngram_bucket.append(i)

    @staticmethod
    def char_pair(a, b):
        if a > b:
            a, b = b, a
        return '' + a + b

    def char_similarity(self, a, b):
        key = FuzzySearch.char_pair(a, b)
        if self.char_similarities and key in self.char_similarities:
            return self.char_similarities[key]
        return self.match_score if a == b else self.mismatch_score

    def sw_align(self, a, start, end):
        b = self.text[start:end]
        n, m = len(a), len(b)
        # building scoring matrix
        f = [[0]] * (n + 1)
        for i in range(0, n + 1):
            f[i] = [0] * (m + 1)
        for i in range(1, n + 1):
            f[i][0] = self.gap_score * i
        for j in range(1, m + 1):
            f[0][j] = self.gap_score * j
        max_score = 0
        start_i, start_j = 0, 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = f[i - 1][j - 1] + self.char_similarity(a[i - 1], b[j - 1])
                insert = f[i][j - 1] + self.gap_score
                delete = f[i - 1][j] + self.gap_score
                score = max(0, match, insert, delete)
                f[i][j] = score
                if score > max_score:
                    max_score = score
                    start_i, start_j = i, j
        # backtracking
        substitutions = Counter()
        i, j = start_i, start_j
        while (j > 0 or i > 0) and f[i][j] != 0:
            if i > 0 and j > 0 and f[i][j] == (f[i - 1][j - 1] + self.char_similarity(a[i - 1], b[j - 1])):
                substitutions[FuzzySearch.char_pair(a[i - 1], b[j - 1])] += 1
                i, j = i - 1, j - 1
            elif i > 0 and f[i][j] == (f[i - 1][j] + self.gap_score):
                i -= 1
            elif j > 0 and f[i][j] == (f[i][j - 1] + self.gap_score):
                j -= 1
            else:
                raise Exception('Smith–Waterman failure')
        return start + j - 1, start + start_j, f[start_i][start_j], substitutions

    def snap(self, start, end):
        start_token = TextRange.token_at(self.text, start)
        if len(start_token) == 0:
            start_token = TextRange.token_at(self.text, start + 1)
        end_token = TextRange.token_at(self.text, max(0, end - 1))
        snap_range = start_token + end_token
        return snap_range.start, snap_range.end

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
        best_substitutions = None
        best_score = -10000000000
        last_window_grams = 0.1
        for window in candidate_windows[:self.max_candidates]:
            ngram_factor = (windows[window] / last_window_grams)
            if ngram_factor < self.candidate_threshold:
                break
            last_window_grams = windows[window]
            interval_start = max(start, int((window - 1) * window_size))
            interval_end = min(stop,  int((window + 2) * window_size))
            interval_start, interval_end, score, substitutions = self.sw_align(look_for, interval_start, interval_end)
            if self.snap_to_word:
                interval_start, interval_end = self.snap(interval_start, interval_end)
            if score > best_score:
                best_interval = TextRange(self.text, interval_start, interval_end)
                best_score = score
                best_substitutions = substitutions
        return best_interval, best_score, best_substitutions
