#!/usr/bin/env python
'''
Builds Sample Databases (.sdb files)
Use "python3 sdb_tool.py -h" for help
'''
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse

from utils import parse_file_size, log_progress
from audio import change_audio_types, AUDIO_TYPE_WAV, AUDIO_TYPE_OPUS
from sample_collections import samples_from_files, DirectSDBWriter, SortingSDBWriter

AUDIO_TYPE_LOOKUP = {
    'wav': AUDIO_TYPE_WAV,
    'opus': AUDIO_TYPE_OPUS
}


def progress(it=None, desc='Processing', total=None):
    print(desc, file=sys.stderr, flush=True)
    return it if CLI_ARGS.no_progress else log_progress(it, interval=CLI_ARGS.progress_interval, total=total)


def add_samples(sdb_writer):
    samples = samples_from_files(CLI_ARGS.sources)
    for sample in progress(change_audio_types(samples, audio_type=sdb_writer.audio_type, processes=CLI_ARGS.workers),
                           total=len(samples),
                           desc='Writing "{}"...'.format(CLI_ARGS.target)):
        sdb_writer.add(sample)


def build_sdb():
    audio_type = AUDIO_TYPE_LOOKUP[CLI_ARGS.audio_type]
    if CLI_ARGS.sort:
        with SortingSDBWriter(CLI_ARGS.target,
                              tmp_sdb_filename=CLI_ARGS.sort_tmp_file,
                              cache_size=parse_file_size(CLI_ARGS.sort_cache_size),
                              audio_type=audio_type) as sdb_writer:
            add_samples(sdb_writer)
    else:
        with DirectSDBWriter(CLI_ARGS.target, audio_type=audio_type) as sdb_writer:
            add_samples(sdb_writer)


def handle_args():
    parser = argparse.ArgumentParser(description='Tool for building Sample Databases (SDB files) '
                                                 'from DeepSpeech CSV files and other SDB files')
    parser.add_argument('--workers', type=int, default=None, help='Number of encoding SDB workers')
    parser.add_argument('--audio-type', default='opus', choices=AUDIO_TYPE_LOOKUP.keys(),
                        help='Audio representation inside target SDB')
    parser.add_argument('--sort', action='store_true', help='Force sample sorting by durations '
                                                            '(assumes SDB sources unsorted)')
    parser.add_argument('--sort-tmp-file', default=None, help='Overrides default tmp_file (target + ".tmp") '
                                                              'for sorting through --sort option')
    parser.add_argument('--sort-cache-size', default='1GB', help='Cache (bucket) size for binary audio data '
                                                                 'for sorting through --sort option')
    parser.add_argument('--no-progress', action="store_true", help='Prevents showing progress indication')
    parser.add_argument('--progress-interval', type=float, default=1.0, help='Progress indication interval in seconds')
    parser.add_argument('sources', nargs='+', help='Source CSV and/or SDB files')
    parser.add_argument('target', help='SDB file to create')
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    build_sdb()
