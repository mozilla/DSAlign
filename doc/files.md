## File formats

### Catalog files (.catalog)

Catalog files (suffix `.catalog`) are used for organizing bigger data file collections and
defining relations among them. It is basically a JSON array of hash-tables where each entry stands
for a single audio file and its associated original transcript.

So a typical catalog looks like this (`data/all.catalog` from this project):

```javascript
[
  {
    "audio": "test1/joined.mp3",
    "tlog": "test1/joined.tlog",
    "script": "test1/transcript.txt",
    "aligned": "test1/joined.aligned"
  },
  {
    "audio": "test2/joined.mp3",
    "tlog": "test2/joined.tlog",
    "script": "test2/transcript.script",
    "aligned": "test2/joined.aligned"
  }
]
```

- `audio` is a path to an audio file (of a format that `pydub` supports)
- `tlog` is the (supposed) path to the STT generated transcription log of the audio file
- `script` is the path to the original transcript of the audio file
(as `.txt` or `.script` file)
- `aligned` is the (supposed) path to a `.aligned` file

Be aware: __All relative file paths are treated as relative to the catalog file's directory__.

The tools `bin/align.sh`, `bin/statistics.sh` and `bin/export.sh` all support parameter
`--catalog`:

The __alignment tool__ `bin/align.sh` requires either `tlog` to point to an existing
file or (if not) `audio` to point to an existing audio file for being able to transcribe
it and store it at the path indicated by `tlog`. Furthermore it requires `script` to
point to an  existing script. It will write its alignment results to the path in `aligned`.

The __export tool__ `bin/export.sh` requires `audio` and `aligned` to point to existing files.

The __statistics tool__ `bin/statistics.sh` requires only `aligned` to point to existing files.

Advantages of having a catalog file:

- Simplified tool usage with only one parameter for defining all involved files (`--catalog`).
- A directory with many files has to be scanned just one time at catalog generation.
- Different file types can live at different and custom locations in the system.
This is important in case of read-only access rights to the original data.
It can also be used for avoiding to taint the original directory tree.
- Accumulated statistics
- Better progress indication (as the total number of files is available up front)
- Reduced tool startup overhead
- Allows for meta-data aware set-splitting on export - e.g. if some speakers are speaking
in several files.

So especially in case of many files to process it is highly recommended to __first create
a catalog file__ with all paths present (even the ones not pointing to existing files yet).


### Script files (.script|.txt)

The alignment tool requires an original script or (human transcript) of the provided audio.
These scripts can be represented in two basic forms:
- plain text files (`.txt`) or
- script files (`.script`)

In case of plain text files the content is considered a continuous stream of text without
any assigned meta data. The only exception is option `--text-meaningful-newlines` which
tells the aligner to consider newlines as separators between utterances
in conjunction with option `--align-phrase-snap-factor`.

If the original data source features utterance meta data, one should consider converting it
to the `.script` JSON file format which looks like this
(except of `data/test2/transcript.script`): 

```javascript
[
  // ...
  {
    "speaker": "Phebe",
    "text": "Good shepherd, tell this youth what 'tis to love."
  },
  {
    "speaker": "Silvius",
    "text": "It is to be all made of sighs and tears; And so am I for Phebe."
  },
  // ...
]
```

_This and the following sub-sections are all using the same real world examples and excerpts_

It is basically again an array of hash-tables, where each hash-table represents an utterance with the
only mandatory field `text` for its textual representation.

All other fields are considered meta data 
(with the key called "meta data type" and the value "meta data instance").

### Transcription log files (.tlog)

The alignment tool relies on timed STT transcripts of the provided audio.
These transcripts are either provided by some external processing 
(even using a different STT system than DeepSpeech) or will get generated
as part of the alignment process.

They are called transcription logs (`.tlog`) and are looking like this
(except of `data/test2/joined.tlog`):

```javascript
[
  // ...
  {
    "start": 7491960,
    "end": 7493040,
    "transcript": "good shepherd"
  },
  {
    "start": 7493040,
    "end": 7495110,
    "transcript": "tell this youth what tis to love"
  },
  {
    "start": 7495380,
    "end": 7498020,
    "transcript": "it is to be made of soles and tears"
  },
  {
    "start": 7498470,
    "end": 7500150,
    "transcript": "and so a may for phoebe"
  },
  // ...
]
```

The fields of each entry:
- `start`: time offset of the audio fragment in milliseconds from the beginning of the
aligned audio file (mandatory)
- `end`: time offset of the audio fragment's end in milliseconds from the beginning of the
aligned audio file (mandatory) 
- `transcript`: STT transcript of the utterance (mandatory)

### Aligned files (.aligned)

The result of aligning an audio file with an original transcript is written to an
`.aligned` JSON file consisting of an array of hash-tables of the following form:

```javascript
[
  // ...
  {
    "start": 7491960,
    "end": 7493040,
    "transcript": "good shepherd",
    "text-start": 98302,
    "text-end": 98316,
    "meta": {
      "speaker": [
        "Phebe"
      ]
    },
    "aligned-raw": "Good shepherd,",
    "aligned": "good shepherd",
    "wng": 99.99999999999997,
    "jaro_winkler": 100.0,
    "levenshtein": 100.0,
    "mra": 100.0,
    "cer": 0.0
  },
  {
    "start": 7493040,
    "end": 7495110,
    "transcript": "tell this youth what tis to love",
    "text-start": 98317,
    "text-end": 98351,
    "meta": {
      "speaker": [
        "Phebe"
      ]
    },
    "aligned-raw": "tell this youth what 'tis to love.",
    "aligned": "tell this youth what 'tis to love",
    "wng": 92.71730687405957,
    "jaro_winkler": 100.0,
    "levenshtein": 96.96969696969697,
    "mra": 100.0,
    "cer": 3.0303030303030303
  },
  {
    "start": 7495380,
    "end": 7498020,
    "transcript": "it is to be made of soles and tears",
    "text-start": 98352,
    "text-end": 98392,
    "meta": {
      "speaker": [
        "Silvius"
      ]
    },
    "aligned-raw": "It is to be all made of sighs and tears;",
    "aligned": "it is to be all made of sighs and tears",
    "wng": 77.93921929148159,
    "jaro_winkler": 100.0,
    "levenshtein": 82.05128205128204,
    "mra": 100.0,
    "cer": 17.94871794871795
  },
  {
    "start": 7498470,
    "end": 7500150,
    "transcript": "and so a may for phoebe",
    "text-start": 98393,
    "text-end": 98415,
    "meta": {
      "speaker": [
        "Silvius"
      ]
    },
    "aligned-raw": "And so am I for Phebe.",
    "aligned": "and so am i for phebe",
    "wng": 66.82687893873339,
    "jaro_winkler": 98.47964113181504,
    "levenshtein": 82.6086956521739,
    "mra": 100.0,
    "cer": 19.047619047619047
  },
  // ...
]
```

Each object array-entry represents an aligned audio fragment with the following attributes:
- `start`: time offset of the audio fragment in milliseconds from the beginning of the
aligned audio file
- `end`: time offset of the audio fragment's end in milliseconds from the beginning of the
aligned audio file
- `transcript`: STT transcript used for aligning
- `text-start`: character offset of the fragment's associated original text within the
aligned text document
- `text-end`: character offset of the end of the fragment's associated original text within the
aligned text document
- `meta`: meta data hash-table with
  - _key_: meta data type
  - _value_: array of meta data instances coalesced from the `.script` entries that
  this entry intersects with
- `aligned-raw`: __raw__ original text fragment that got aligned with the audio fragment
and its STT transcript
- `aligned`: __clean__ original text fragment that got aligned with the audio fragment
and its STT transcript
- `<metric>` For each `--output-<metric>` parameter the alignment tool adds an entry with the
computed value (in this case `wng`, `jaro_winkler`, `levenshtein`, `mra`, `cer`)
