## Text distance metrics

This section lists all available text distance metrics along with their IDs for
command-line use.

### Weighted N-grams (wng)

The weighted N-gram score is computed as the sum of the number of weighted shared N-grams
between the two texts.
It ensures that:
- Shared N-gram instances near interval bounds (dependent on situation) get rated higher than
the ones near the center or opposite end
- Large shared N-gram instances are weighted higher than short ones

`--align-min-ngram-size <SIZE>` sets the start (minimum) N-gram size

`--align-max-ngram-size <SIZE>` sets the final (maximum) N-gram size

`--align-ngram-size-factor <FACTOR>` sets a weight factor for the size preference

`--align-ngram-position-factor <FACTOR>` sets a weight factor for the position preference

### Jaro-Winkler (jaro_winkler)

Jaro-Winkler is an edit distance metric described
[here](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance).

### Editex (editex)

Editex is a phonetic text distance algorithm described
[here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.2138&rep=rep1&type=pdf).

### Levenshtein (levenshtein)

Levenshtein is an edit distance metric described
[here](https://en.wikipedia.org/wiki/Levenshtein_distance).

### MRA (mra)

The "Match rating approach" is a phonetic text distance algorithm described
[here](https://en.wikipedia.org/wiki/Match_rating_approach).

### Hamming (hamming)

The Hamming distance is an edit distance metric described
[here](https://en.wikipedia.org/wiki/Hamming_distance).

### Word error rate (wer)

This is the same as Levenshtein - just on word level.

Not available for gap alignment.

### Character error rate (cer)

This is the same as Levenshtein but using a different implementation.

Not available for gap alignment.

### Smith-Waterman score (sws)

This is the final Smith-Waterman score coming from the rough alignment
step (but before gap alignment!).
It is described
[here](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm).

Not available for gap alignment.

### Transcript length (tlen)

The character length of the STT transcript.

Not available for gap alignment.

### Matched text length (mlen)

The character length of the matched text of the original transcript (cleaned).

Not available for gap alignment.
