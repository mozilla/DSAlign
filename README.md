# DSAlign
DeepSpeech based forced alignment tool

## Installation

It is recommended to use this tool from within a virtual environment.
There is a script for creating one with all requirements in the git-ignored dir `venv`:

```bash
$ bin/createenv.sh
$ ls venv
bin  include  lib  lib64  pyvenv.cfg  share
```

`bin/align.sh` will automatically use it.

## Prerequisites

Internally DSAlign uses the [DeepSpeech](https://github.com/mozilla/DeepSpeech/) STT engine.
For it to be able to function, it requires a couple of files that are specific to 
the language of the speech data you want to align.
If you want to align English, there is already a helper script that will download and prepare
all required data:

```bash
$ bin/getmodel.sh 
[...]
$ ls models/en/
alphabet.txt  lm.binary  output_graph.pb  output_graph.pbmm  output_graph.tflite  trie
```

### Example data

There is also a script for downloading and preparing some public domain speech and transcript data.

```bash
$ bin/gettestdata.sh
$ ls data
test1  test2
```

## Using the tool

```bash
$ bin/align.sh --help
[...]
```

### Alignment using example data

```bash
$ bin/align.sh --output-max-cer 15 --loglevel 10 data/test1/audio.wav data/test1/transcript.txt data/test1/aligned.json
```

## The algorithm

### Step 1 - Splitting audio

A voice activity detector (at the moment this is `webrtcvad`) is used
to split the provided audio data into voice fragments.
These fragments are essentially streams of continuous speech without any longer pauses 
(e.g. sentences).

`--audio-vad-aggressiveness <AGGRESSIVENESS>` can be used to influence the length of the resulting fragments.

### Step 2 - Transcription of voice fragments through STT

After VAD splitting the resulting fragments are transcribed into textual phrases.
This transcription is done through [DeepSpeech](https://github.com/mozilla/DeepSpeech/) STT.

As this can take a longer time, all resulting phrases 
are - together with their timestamps - saved into a cache file 
(the `result` parameter path with suffix `.cache`).
Consecutive calls will look for that file and - if found - 
load it and skip the transcription phase.

`--stt-model-dir <DIR>` points DeepSpeech to the language specific model data directory.
It defaults to `models/en`. Use `bin/getmodel.sh` for preparing it. 

### Step 3 - Preparation of original text

STT transcripts are typically provided in a normalized textual form with
- no casing,
- no punctuation and
- normalized whitespace (single spaces only).

So for being able to align STT transcripts with the original text it is necessary
to internally convert the original text into the same form.

This happens in two steps:
1. Normalization of whitespace, lower-casing all text and 
replacing some characters with spaces (e.g. dashes)
2. Removal of all characters that are not in the languages's alphabet
(see DeepSpeech model data)

Be aware: *This conversion happens on text basis and will not remove unspoken content
like markup/markdown tags or artifacts. This should be done beforehand.
Reducing the difference of spoken and original text will improve alignment quality and speed.*

In the very unlikely situation that you have to change the default behavior (of step 1),
there are some switches:

`--text-keep-dashes` will prevent substitution of dashes with spaces.

`--text-keep-ws` will keep whitespace untouched.

`--text-keep-casing` will keep character casing as provided.

### Step 4a - Finding candidate windows in original text

Finding the best match of a given phrase within the original transcript is essentially about
finding the character sequence within the original text that has the lowest Levenshtein distance
to the phrase.

As best Levenshtein distance algorithms are still of quadratic complexity,
computing them for each possible target sequence to find the lowest one is not feasible. 

So this tool follows a two-phase approach where the first goal is to get a list of so called 
candidate windows. Candidate windows are areas within the original text with higher
probability of containing the sequence with the lowest Levenshtein distance to the search pattern.

For determining them the original text is virtually split into a sequence of disjunct windows
of the length of the search pattern. Then the blocks are ordered descending by the number of
3-grams they share with the search pattern. Candidate windows are then taken from the beginning
of this ordered list.

`--align-max-candidates <CANDIDATES>` sets the maximum number of candidate windows
taken from the beginning of the list for further alignment.

`--align-candidate-threshold <THRESHOLD>` multiplied with the number of 3-grams of the predecessor
window it gives the minimum number of 3-grams the next candidate window has to have to also be
considered a candidate.

### Step 4b - Aligning phrases within candidate windows

For each candidate window the best possible alignment is searched 
by computing the Levenshtein distance for each candidate in a radius of half a window 
around the candidate window:

1. Binary search best (Levenshtein distance) sequence with the same length as the search pattern 
2. Binary search best (Levenshtein distance) stretched or shrunken sequence 
around sequence location of step 1 
3. Snap first and last token of search pattern to lowest distance tokens around character offsets
in original text

`--align-no-snap-to-token` deactivates token snapping (step 3)

`--align-stretch-fraction <FRACTION>` sets the fraction of the search pattern length the final alignment
can get shrunken or expanded

### Step 5 - Selection, filtering and output

Finally the best alignment of all candidate windows is selected as the winner.
It has to survive a series of filters for getting into the result file:

`--output-min-length <LENGTH>` only accepts samples having original transcripts of the
provided minimum character length
                              
`--output-max-length <LENGTH>` only accepts samples having original transcripts of the
provided maximum character length 

`--output-min-wer <WER>` only accepts samples whose STT transcripts have the provided minimum
word error rate when compared to the best matching original transcript sequence

`--output-max-wer <WER>` only accepts samples whose STT transcripts have the provided maximum
word error rate when compared to the best matching original transcript sequence

`--output-min-cer <CER>` only accepts samples whose STT transcripts have the provided minimum
character error rate when compared to best matching original transcript sequence

`--output-max-cer <CER>` only accepts samples whose STT transcripts have the provided maximum
character error rate when compared to best matching original transcript sequence

All result samples are written to a JSON result file of the form:
```JSON
[
  {
    "time-start": 714630,
    "time-length": 8100,
    "text-start": 13522,
    "text-length": 150,
    "cer": 0.31654676258992803,
    "wer": 0.39285714285714285
  },
  ...
]
```

Each object array-entry represents a matched audio fragment with the following attributes:
- `time-start`: Time offset of the audio fragment in milliseconds from the beginning of the aligned audio file
- `time-length`: Duration of the audio fragment in milliseconds
- `text-start`: Character offset of the fragment's associated original text within the aligned text document
- `text-length`: Character length of the fragment's associated original text within the aligned text document
- `cer`: Character error rate of the STT transcribed audio fragment compared to the associated original text
- `wer`: Word error rate of the STT transcribed audio fragment compared to the associated original text

Error rates are provided as fractional values (typically between 0.0 = 0% and 1.0 = 100% where numbers >1.0 are theoretically possible).

## General options

`--play` will play each aligned sample using the `play` command of the SoX audio toolkit

`--text-context <CONTEXT-SIZE>` will add additional `CONTEXT-SIZE` characters around original
transcripts when logged
