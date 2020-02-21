import os
import sys
import json
import argparse
from os import path
from pickle import load, dump
from collections import Counter
from datetime import timedelta
from utils import log_progress


def fail(message, code=1):
    print(message)
    exit(code)


class AlignmentStatistics:
    def __init__(self):
        self.top = 100
        self.stat_ids = ['wng', 'sws', 'wer', 'cer', 'jaro_winkler', 'editex', 'levenshtein', 'mra', 'hamming']
        self.stats = {}
        self.stats_duration = {}
        for stat_id in self.stat_ids:
            self.stats[stat_id] = Counter()
            self.stats_duration[stat_id] = Counter()

        self.total_files = 0
        self.total_utterances = 0
        self.total_duration = 0
        self.total_length = 0

        self.durations = Counter()
        self.lengths = Counter()

        self.meta_counters = {}

    @staticmethod
    def progress(lst, desc='Processing', total=None):
        return lst

    def load_aligned(self, aligned_path):
        self.total_files += 1
        with open(aligned_path, 'r') as aligned_file:
            utterances = json.loads(aligned_file.read())
        for utterance in utterances:
            self.total_utterances += 1
            duration = utterance['end'] - utterance['start']
            self.durations[int(duration / 1000)] += 1
            self.total_duration += duration
            length = utterance['text-end'] - utterance['text-start']
            self.lengths[length] += 1
            self.total_length += length
            for stat_id in self.stat_ids:
                if stat_id in utterance:
                    self.stats[stat_id][int(utterance[stat_id])] += 1
                    self.stats_duration[stat_id][int(utterance[stat_id])] += duration
            if 'meta' in utterance:
                for meta_type, instances in utterance['meta'].items():
                    if meta_type not in self.meta_counters:
                        self.meta_counters[meta_type] = Counter()
                    for instance in instances:
                        self.meta_counters[meta_type][instance] += 1

    def load_catalog(self, catalog_path, ignore_missing=True):
        catalog = path.abspath(catalog_path)
        catalog_dir = path.dirname(catalog)
        with open(catalog, 'r') as catalog_file:
            catalog_entries = json.load(catalog_file)
        for entry in AlignmentStatistics.progress(catalog_entries, desc='Reading catalog'):
            aligned_path = entry['aligned']
            if not path.isabs(aligned_path):
                aligned_path = path.join(catalog_dir, aligned_path)
            if path.isfile(aligned_path):
                self.load_aligned(aligned_path)
            else:
                if ignore_missing:
                    continue
                else:
                    fail('Problem loading catalog "{}": Missing referenced alignment file "{}"'
                         .format(catalog_path, aligned_path))

    def print_stats(self):
        print('Total number of files: {:,}'.format(self.total_files))
        print('')
        print('Total number of utterances: {:,}'.format(self.total_utterances))
        print('')
        print('Total aligned utterance character length: {:,}'.format(self.total_length))
        print('')
        print('Total utterance duration: {} ({:,} hours)'.format(
            timedelta(milliseconds=self.total_duration),
            int(self.total_duration / (1000 * 60 * 60))))
        print('')
        
        for meta_type, counter in self.meta_counters.items():
            print('Overall number of instances of meta type "{}": {:,}'.format(meta_type, len(counter.keys())))
            print('')
            print('{} most frequent "{}" instances:'.format(self.top, meta_type))
            for value, count in counter.most_common(self.top):
                print(value.ljust(20) + '{:,}'.format(count).rjust(12))

        for stat_id in self.stat_ids:
            counter = self.stats_duration[stat_id]
            if len(counter) == 0:
                continue
            print('')
            print(stat_id.upper() + ':')
            above = 0
            for value in sorted(counter):
                count = counter[value] / (60 * 60 * 1000)
                if value <= 100:
                    print(str(value).ljust(10) + '{:12.2f}'.format(count).rjust(12))
                else:
                    above += count
            if above > 0:
                print('100+'.ljust(10) + '{:12.2f}'.format(above).rjust(12))


def main(args):
    parser = argparse.ArgumentParser(description='Export aligned speech samples.')

    parser.add_argument('--cache', type=str,
                        help='Use provided file as statistics cache (if existing, all other input options are ignored)')
    parser.add_argument('--aligned', type=str, action='append',
                        help='Read alignment file ("<...>.aligned") as input')
    parser.add_argument('--catalog', type=str, action='append',
                        help='Read alignment references of provided catalog ("<...>.catalog") as input')
    parser.add_argument('--no-progress', action='store_true',
                        help='Prevents showing progress bars')

    args = parser.parse_args()

    def progress(it=None, desc='Processing', total=None):
        print(desc)
        return it if args.no_progress else log_progress(it, interval=args.progress_interval, total=total)
    AlignmentStatistics.progress = progress

    if args.cache is not None and path.exists(args.cache):
        with open(args.cache, 'rb') as stats_file:
            stats = load(stats_file)
    else:
        stats = AlignmentStatistics()
        if args.catalog is not None:
            for catalog_path in args.catalog:
                stats.load_catalog(catalog_path, ignore_missing=True)
        if args.aligned is not None:
            for aligned_path in args.aligned:
                stats.load_aligned(aligned_path)
        if args.cache is not None:
            with open(args.cache, 'wb') as stats_file:
                dump(stats, stats_file)

    stats.print_stats()


if __name__ == '__main__':
    main(sys.argv[1:])
    os.system('stty sane')
