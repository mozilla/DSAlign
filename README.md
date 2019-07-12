# DSAlign
DeepSpeech based forced alignment tool

## Installation

It is recommended to use this tool from within a virtual environment.
We got a script for creating one in git-ignored dir `venv` with all requirements:

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

### Step 1 - Using STT to get phrases

A voice activity detector (at the moment this is `webrtcvad`) is used
to split the provided audio data into voice fragments.
These fragments are essentially streams of continuous speech without any longer pauses 
(e.g. sentences).
You can use `--audio-vad-aggressiveness` to influence the length of the resulting fragments.

Each fragment is immediately transcribed (through DeepSpeech using the user-provided model data) into a phrase.

As this can take a longer time, all resulting phrases 
are - together with their timestamps - saved into a cache file 
(the `result` parameter path with suffix `.cache`). Consecutive calls will look for that file 
and - if found - load it and skip the transcription phase. 

### Step 2 - Finding phrase alignment candidate windows in original text

[tbd]

### Step 3 - Aligning phrases within candidate windows

[tbd]

### Step 4 - Selection, filtering and output

[tbd]