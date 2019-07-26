from collections import Counter
from nltk import ngrams
from text import levenshtein, TextRange
from utils import circulate, by_len


class FuzzySearch(object):
    def __init__(self,
                 text,
                 max_candidates=10,
                 candidate_threshold=0.8,
                 snap_token=True,
                 stretch_factor=1/3,
                 match_score=2,
                 mismatch_score=-2,
                 delete_score=-1,
                 insert_score=-1,
                 similarities=None):
        self.text = text
        self.max_candidates = max_candidates
        self.candidate_threshold = candidate_threshold
        self.snap_token = snap_token
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.delete_score = delete_score
        self.insert_score = insert_score
        self.similarities = similarities
        self.ngrams = {}
        for i, ngram in enumerate(ngrams(' ' + text + ' ', 3)):
            if ngram in self.ngrams:
                ngram_bucket = self.ngrams[ngram]
            else:
                ngram_bucket = self.ngrams[ngram] = []
            ngram_bucket.append(i)

    @staticmethod
    def similarity_key(a, b):
        if a > b:
            a, b = b, a
        return '' + a + b

    def similarity(self, a, b):
        key = FuzzySearch.similarity_key(a, b)
        if self.similarities and key in self.similarities:
            return self.similarities[key]
        return self.match_score if a == b else self.mismatch_score

    def nwmatch(self, a, b):
        n, m = len(a), len(b)
        f = [[0] * m] * n
        for i in range(n):
            f[i][0] = self.insert_score * i  # CHECK: correct order of delete and insert scores?
        for j in range(m):
            f[0][j] = self.delete_score * j
        max_score = 0
        start_i, start_j = 0, 0
        for i in range(1, n):
            for j in range(1, m):
                match = f[i - 1][j - 1] + self.similarity(a[i], b[j])
                delete = f[i - 1][j] + self.delete_score
                insert = f[i][j - 1] + self.insert_score
                score = max(0, match, insert, delete)
                f[i][j] = score
                if score > max_score:
                    max_score = score
                    start_i, start_j = i, j

        substitutions = Counter()
        score = 0
        match_start, match_len = -1, 0
        i, j = start_i, start_j
        align_a, align_b = '', ''
        while (j > 0 or i > 0) and f[i][j] != 0:
            if i > 0 and j > 0 and f[i][j] == (f[i - 1][j - 1] + self.similarity(a[i], b[j])):
                align_a = a[i] + align_a
                align_b = b[j] + align_b
                score += self.similarity(a[i], b[j])
                substitutions[FuzzySearch.similarity_key(a[i], b[j])] += 1
                i, j = i - 1, j - 1
            elif i > 0 and f[i][j] == (f[i - 1][j] + self.delete_score):
                print('D')
                align_a = a[i] + align_a
                align_b = '-' + align_b
                score += self.delete_score
                i -= 1
            elif j > 0 and f[i][j] == (f[i][j - 1] + self.insert_score):
                print('I')
                align_a = '-' + align_a
                align_b = b[j] + align_b
                score += self.insert_score
                j -= 1
            else:
                print('Warum?', i, j, self.similarity(a[i], b[j]), f[i-1][j-1], f[i-1][j], f[i][j-1], f[i][j])

        print(align_a)
        print(align_b)

        return match_start, match_len, score, substitutions

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
        best_score = -10000000000
        last_window_grams = 0.1
        for window in candidate_windows[:self.max_candidates]:
            if windows[window] / last_window_grams < self.candidate_threshold:
                break
            last_window_grams = windows[window]
            interval_start = max(start, int((window-0.5)*window_size))
            interval_end   = min(stop,  int((window+1.5)*window_size))
            interval_text = self.text[interval_start:interval_end]
            match_start, match_len, score, substitutions = self.nwmatch(look_for, interval_text)
            match_start += interval_start
            interval = TextRange(self.text, match_start, match_start + match_len)
            if score > best_score:
                print('new best')
                best_interval = interval
                best_score = score
        return best_interval, best_score
