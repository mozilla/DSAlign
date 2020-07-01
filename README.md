# DSAlign
DeepSpeech based forced alignment tool

## Installation

It is recommended to use this tool from within a virtual environment.
After cloning and changing to the root of the project,
there is a script for creating one with all requirements in the git-ignored dir `venv`:

```shell script
$ bin/createenv.sh
$ ls venv
bin  include  lib  lib64  pyvenv.cfg  share
```

`bin/align.sh` will automatically use it.

Internally DSAlign uses the [DeepSpeech](https://github.com/mozilla/DeepSpeech/) STT engine.
For it to be able to function, it requires a couple of files that are specific to 
the language of the speech data you want to align.
If you want to align English, there is already a helper script that will download and prepare
all required data:

```shell script
$ bin/getmodel.sh 
[...]
$ ls models/en/
alphabet.txt  lm.binary  output_graph.pb  output_graph.pbmm  output_graph.tflite  trie
```

## Overview and documentation

A typical application of the aligner is done in three phases: 

 1. __Preparing__ the data. Albeit most of this has to be done individually,
    there are some [tools for data preparation, statistics and maintenance](doc/tools.md).
    All involved file formats are described [here](doc/files.md).
 2. __Aligning__ the data using [the alignment tool and it algorithm](doc/algo.md).
 3. __Exporting__ aligned data using [the data-set exporter](doc/export.md).

## Quickstart example

### Example data

There is a script for downloading and preparing some public domain speech and transcript data.
It requires `ffmpeg` for some sample conversion.

```shell script
$ bin/gettestdata.sh
$ ls data
test1  test2
```

### Alignment using example data

Now the aligner can be called either "manually" (specifying all involved files directly):

```shell script
$ bin/align.sh --audio data/test1/audio.wav --script data/test1/transcript.txt --aligned data/test1/aligned.json --tlog data/test1/transcript.log
```

Or "automatically" by specifying a so-called catalog file that bundles all involved paths:

```shell script
$ bin/align.sh --catalog data/test1.catalog
```
