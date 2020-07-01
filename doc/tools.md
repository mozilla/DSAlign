## Tools

### Statistics tool

The statistics tool `bin/statistics.sh` can be used for displaying aggregated statistics of
all passed alignment files. Alignment files can be specified directly through the 
`--aligned <ALIGNED-FILE>` multi-option and indirectly through the `--catalog <CATALOG-FILE>` multi-option.

Example call:

```shell script
DSAlign$ bin/statistics.sh --catalog data/all.catalog 
Reading catalog
 2 of 2 : 100.00% (elapsed: 00:00:00, speed: 94.27 it/s, ETA: 00:00:00)
Total number of files: 2

Total number of utterances: 5,949

Total aligned utterance character length: 202,191

Total utterance duration: 3:53:28.410000 (3 hours)

Overall number of instances of meta type "speaker": 27

100 most frequent "speaker" instances:
Rosalind                     678
Touchstone                   401
Orlando                      310
Jaques                       303
Celia                        281
Oliver                       125
Phebe                        108
Duke Senior                   87
Silvius                       86
Adam                          81
Corin                         68
Duke Frederick                53
Le Beau                       52
First Lord                    49
Charles                       33
Amiens                        27
Audrey                        27
Second Page                   22
Hymen                         19
Jaques De Boys                16
Second Lord                   12
William                       12
Forester                       8
First Page                     7
Sir Oliver Martext             4
Dennis                         3
A Lord                         1
```

### Catalog tool

The catalog tool allows for maintenance of catalog files.
It takes multiple catalog files (supporting wildcards) and allows for applying several checks and tweaks before
potentially exporting them to a new combined catalog file.

Options:

 - `--output <CATALOG>`: Writes all items of all passed catalogs into to the specified new catalog.
 - `--make-relative`: Makes all paths entries of all items relative to the parent directory of the 
   new catalog (see `--output`).
 - `--order-by <ENTRY>`: Entry that should be used for sorting items in new catalog (see `--output`).
 - `--check <ENTRIES>`: Checks file existence of all passed (comma separated) entries of each catalog 
   item (e.g. `--check aligned,audio` will check if `aligned` and `audio` file paths of each catalog item exist). 
   `--check all` will check all entries of each item.
 - `--on-miss fail|drop|remove|ignore`: What to do if a checked (`--check`) file is not existing. 
   - `fail`: tool will exit with an error status (default)
   - `drop`: the catalog item with all its entries will be removed (see `--output`)
   - `remove`: the missing entry within the catalog item will be removed (see `--output`)
   - `ignore`: just logs the missing entry
   
Example usage:
```shell script
$ cat a.catalog 
[
  {
    "entry1": "is/not/existing/x",
    "entry2": "is/existing/x"
  }
]
$ cat b.catalog 
[
  {
    "entry1": "is/not/existing/y",
    "entry2": "is/existing/y"
  }
]
$ bin/catalog_tool.sh --check all --on-miss remove --output c.catalog --make-relative a.catalog b.catalog 
Loading catalog "a.catalog"
Catalog "a.catalog" - Missing file for "entry1" ("is/not/existing/x") - removing entry from item
Loading catalog "b.catalog"
Catalog "b.catalog" - Missing file for "entry1" ("is/not/existing/y") - removing entry from item
Writing catalog "c.catalog"
$ cat c.catalog 
[
  {
    "entry2": "is/existing/x"
  },
  {
    "entry2": "is/existing/y"
  }
]
```

### Meta data annotation tool

The meta data annotation tool allows for assigning meta data fields to all items of script files or transcription logs.
It takes only two parameters: The file and a series of `<key>=<value>` assignments.

Example usage:
```shell script
$ cat a.tlog 
[
  {
    "start": 330.0,
    "end": 2820.0,
    "transcript": "some text without a meaning"
  },
  {
    "start": 3456.0,
    "end": 5123.0,
    "transcript": "some other text without a meaning"
  }
]
$ bin/meta.sh a.tlog speaker=alice year=2020
$ cat a.tlog 
[
  {
    "start": 330.0,
    "end": 2820.0,
    "transcript": "some text without a meaning",
    "speaker": "alice",
    "year": "2020"
  },
  {
    "start": 3456.0,
    "end": 5123.0,
    "transcript": "some other text without a meaning",
    "speaker": "alice",
    "year": "2020"
  }
]
```