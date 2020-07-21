## Export

After files got successfully aligned, one would possibly want to export the aligned utterances
as machine learning training samples.

This is where the export tool `bin/export.sh` comes in.

### Step 1 - Reading the input

The exporter takes either a single audio file (`--audio <AUDIO>`) 
plus a corresponding `.aligned` file (`--aligned <ALIGNED>`) or a series
of such pairs from a `.catalog` file (`--catalog <CATALOG>`) as input.

All of the following computations will be done on the joined list of all aligned
utterances of all input pairs.

Option `--ignore-missing` will not fail on missing file references in the catalog
and instead just ignore the affected catalog entry.

### Step 2 - (Pre-) Filtering

The parameter `--filter <EXPR>` allows to specify a Python expression that has access
to all data fields of an aligned utterance (as can be seen in `.aligned` file entries).

This expression is now applied to each aligned utterance and in case it returns `True`,
the utterance will get excluded from all the following steps. 
This is useful for excluding utterances that would not work as input for the planned
training or other kind of application.

### Step 3 - Computing quality

As with filtering, the parameter `--criteria <EXPR>` allows for specifying a Python 
expression that has access to all data fields of an aligned utterance.

The expression is applied to each aligned utterance and its numerical return 
value is assigned to each utterance as `quality`.

### Step 4 - De-biasing

This step is to (optionally) exclude utterances that would otherwise bias the data
(risk of overfitting).

For each `--debias <META DATA TYPE>` parameter the following procedure is applied:
1. Take the meta data type (e.g. "name") and read its instances (e.g. "Alice" or "Bob")
from each utternace and group all utterances accordingly
(e.g. a group with 2 utterances of "Alice" and a group with 15 utterances of "Bob"...)
2. Compute the standard deviation (`sigma`) of the instance-counts of the groups
3. For each group: If the instance-count exceeds `sigma` times `--debias-sigma-factor <FACTOR>`:
    - Drop the number of exceeding utterances in order of their `quality` (lowest first)
    
### Step 5 - Partitioning

Training sets are often partitioned into several quality levels.

For each `--partition <QUALITY:PARTITION>` parameter (ordered descending by `QUALITY`):
If the utterance's `quality` value is greater or equal `QUALITY`, assign it to `PARTITION`.

Remaining utterances are assigned to partition `other`.

### Step 6 - Splitting

Training sets (actually their partitions) are typically split into sets `train`, `dev` 
and `test` ([explanation](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets)).

This can get automated through parameter `--split` which will let the exporter split each
partition (or the entire set) accordingly.

Parameter `--split-field` allows for specifying a meta data type that should be considered 
atomic (e.g. "speaker" would result in all utterances of a speaker 
instance - like "Alice" - to end up in one sub-set only). This atomic behavior will also hold
true across partitions.

Option `--split-drop-multiple` allows for dropping all samples with multiple `--split-field` assignments - e.g. a 
sample with more than one "speaker".

In contrast option `--split-drop-unknown` allows for dropping all samples with no `--split-field assignment`.

With option `--assign-{train|dev|test} <VALUES>` one can pre-assign values (of the comma-separated list)
to the specified set.

Option `--split-seed <SEED>` sets an integer random seed for the split operation.

### Step 7 - Output

For each partition/sub-set combination the following is done:
 - Construction of a `name` (e.g. `good-dev` will represent the validation set of partition `good`).
 - All samples are lazy-loaded and potentially re-sampled to match parameters: 
   - `--channels <N>`: Number of audio channels - 1 for mono (default), 2 for stereo
   - `--rate <RATE>`: Sample rate - default: 16000
   - `--width <WIDTH>`: Sample width in bytes - default: 2 (16 bit)
   
   `--workers <WORKERS>` can be used to specify how many parallel processes should be used for loading and re-sampling.
   
   `--tmp-dir <DIR>` overrides system default temporary directory that is used for converting samples.
   
   `--skip-damaged` allows for just skipping export of samples that cannot be loaded.
   
 - If option `--target-dir <DIR>` is provided, all output will be written to the provided target directory.
   This can be done in two different ways:
   
     1. With the additional option `--sdb` each set will be written to a so called Sample-DB
        that can be used by DeepSpeech. It will be written as `<name>.sdb` into the target directory.
        SDB export can be controlled with the following additional options:
        - `--sdb-bucket-size <SIZE>`: SDB bucket size (using units like "1GB") for external sorting of the samples
        - `--sdb-workers <WORKERS>`: Number of parallel workers for preparing and compressing SDB entries
        - `--sdb-buffered-samples <SAMPLES>`: Number of samples per bucket buffer during last phase of external sorting
        - `--sdb-audio-type <TYPE>`: Internal audio type for storing SDB samples - `wav` or `opus` (default)
     2. Without option `--sdb` all samples are written as WAV-files into sub-directory `<name>`
        of the target directory and a list of samples to a `<name>.csv` file next to it with columns 
        `wav_filename`, `wav_filesize`, `transcript`.
        
   If not omitted through option `--no-meta`, a CSV file called `<name>.meta` is written to the target directory.
   For each written sample it provides the following columns: 
   `sample`, `split_entity`, `catalog_index`, `source_audio_file`, `aligned_file`, `alignment_index`.
   
   Throughout this process option `--force` allows to overwrite any existing files.
 - If instead option `--target-tar <TAR-FILE>` is provided, the same file structure as with `--target-dir <DIR>`
   is directly written to the specified tar-file.
   This output variant does not support writing SDBs.
 
### Additional functionality

Option `--plan <PLAN>` can be used to cache all computational steps before actual output writing.
Will be loaded if existing or generated otherwise.
This allows for writing several output formats using the same sample set distribution and without having to load
alignment files and re-calculate quality metrics, de-biasing, partitioning or splitting.

Using `--dry-run` one can avoid any writing and get a preview on set-splits and so forth
(`--dry-run-fast` won't even load any sample).
