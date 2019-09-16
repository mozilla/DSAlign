import os
import sys
import csv
import math
import json
import random
import logging
import argparse
import statistics
import os.path as path

from tqdm import tqdm
from pydub import AudioSegment
from collections import Counter
from multiprocessing import Pool

unknown = '<unknown>'


def fail(message, code=1):
    logging.fatal(message)
    exit(code)


def engroup(lst, get_key):
    groups = {}
    for obj in lst:
        key = get_key(obj)
        if key in groups:
            groups[key].append(obj)
        else:
            groups[key] = [obj]
    return groups


def get_set_sizes(population_size):
    margin_of_error = 0.01
    fraction_picking = 0.50
    z_score = 2.58  # Corresponds to confidence level 99%
    numerator = (z_score ** 2 * fraction_picking * (1 - fraction_picking)) / (
            margin_of_error ** 2
    )
    sample_size = 0
    for train_size in range(population_size, 0, -1):
        denominator = 1 + (z_score ** 2 * fraction_picking * (1 - fraction_picking)) / (
                margin_of_error ** 2 * train_size
        )
        sample_size = int(numerator / denominator)
        if 2 * sample_size + train_size <= population_size:
            break
    return population_size - 2 * sample_size, sample_size


def load_segment(audio_path):
    return audio_path, AudioSegment.from_file(audio_path)


def load_segment_dry(audio_path):
    if path.isfile(audio_path):
        logging.debug('Would load file "{}"'.format(audio_path))
    else:
        fail('File not found: "{}"'.format(audio_path))
    return audio_path, AudioSegment.empty()


def main(args):
    parser = argparse.ArgumentParser(description='Export aligned speech samples.')

    parser.add_argument('--audio', type=str,
                        help='Take audio file as input (requires "--aligned <file>")')
    parser.add_argument('--aligned', type=str,
                        help='Take alignment file ("<...>.aligned") as input (requires "--audio <file>")')
    parser.add_argument('--catalog', type=str,
                        help='Take alignment and audio file references of provided catalog ("<...>.catalog") as input')
    parser.add_argument('--ignore-missing', action="store_true",
                        help='Ignores catalog entries with missing files')
    parser.add_argument('--target-dir', type=str, required=True,
                        help='Existing target directory for storing generated sets (files and directories)')
    parser.add_argument('--filter', type=str,
                        help='Python expression that computes a boolean value from sample data fields. '
                             'If the result is True, the sample will be dropped.')
    parser.add_argument('--criteria', type=str, default='100',
                        help='Python expression that computes a number as quality indicator from sample data fields.')
    parser.add_argument('--partition', type=str, action='append',
                        help='Expression of the form "<number>:<partition>" where all samples with a quality indicator '
                             '(--criteria) above or equal the given number and below the next bigger one are assigned '
                             'to the specified partition. Samples below the lowest partition criteria are assigned to '
                             'partition "other".')
    parser.add_argument('--split', action="store_true",
                        help='Split each partition except "other" into train/dev/test sub-sets.')
    parser.add_argument('--split-field', type=str,
                        help='Sample meta field that should be used for splitting (e.g. "speaker")')
    parser.add_argument('--split-seed', type=int,
                        help='Random seed for set splitting')
    parser.add_argument('--debias', type=str, action='append',
                        help='Sample meta field to group samples for debiasing (e.g. "speaker"). '
                             'Group sizes will be capped according to --debias-sigma-factor')
    parser.add_argument('--debias-sigma-factor', type=float, default=3.0,
                        help='Standard deviation (sigma) factor after which the sample number of a group gets capped')
    parser.add_argument('--loglevel', type=int, default=20,
                        help='Log level (between 0 and 50) - default: 20')
    parser.add_argument('--no-progress', action="store_true",
                        help='Prevents showing progress bars')
    parser.add_argument('--format', type=str, default='csv',
                        help='Sample list format - one of (json|csv)')
    parser.add_argument('--rate', type=int,
                        help='Export wav-files with this sample rate')
    parser.add_argument('--channels', type=int,
                        help='Export wav-files with this number of channels')
    parser.add_argument('--force', action="store_true",
                        help='Overwrite existing files')
    parser.add_argument('--dry-run', action="store_true",
                        help='Simulates export without writing or creating any file or directory')
    parser.add_argument('--dry-run-fast', action="store_true",
                        help='Simulates export without writing or creating any file or directory. '
                             'In contrast to --dry-run this faster simulation will not load samples.')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers for loading and re-sampling audio files. Default: Number of CPUs')
    parser.add_argument('--pretty', action="store_true",
                        help='Writes indented JSON output')

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr, level=args.loglevel if args.loglevel else 20)

    def progress(iter, desc='Processing', total=None):
        desc = desc.rjust(24)
        return iter if args.no_progress else tqdm(iter, desc=desc, total=total, ncols=120)

    logging.debug("Start")

    pairs = []

    def check_path(target_path, fs_type='file'):
        if not (path.isfile(target_path) if fs_type == 'file' else path.isdir(target_path)):
            logging.fatal('{} not existing: "{}"'.format(fs_type[0].upper() + fs_type[1:], target_path))
            exit(1)
        return path.abspath(target_path)

    def make_absolute(base_path, spec_path):
        if not path.isabs(spec_path):
            spec_path = path.join(base_path, spec_path)
        spec_path = path.abspath(spec_path)
        return spec_path if path.isfile(spec_path) else None

    target_dir = check_path(args.target_dir, fs_type='directory')
    if args.audio:
        if args.aligned:
            pairs.append((check_path(args.audio), check_path(args.aligned)))
        else:
            fail('If you specify "--audio", you also have to specify "--aligned"')
    elif args.aligned:
        fail('If you specify "--aligned", you also have to specify "--audio"')
    elif args.catalog:
        catalog = check_path(args.catalog)
        catalog_dir = path.dirname(catalog)
        with open(catalog, 'r') as catalog_file:
            catalog_entries = json.load(catalog_file)
        for entry in progress(catalog_entries, desc='Reading catalog'):
            audio = make_absolute(catalog_dir, entry['audio'])
            aligned = make_absolute(catalog_dir, entry['aligned'])
            if audio is None or aligned is None:
                if args.ignore_missing:
                    continue
                if audio is None:
                    fail('Problem loading catalog "{}": Missing referenced audio file "{}"'
                         .format(args.catalog, entry['audio']))
                if aligned is None:
                    fail('Problem loading catalog "{}": Missing referenced alignment file "{}"'
                         .format(args.catalog, entry['aligned']))
            pairs.append((audio, aligned))
    else:
        fail('You have to either specify "--audio" and "--aligned" or "--catalog"')

    dry_run = args.dry_run or args.dry_run_fast
    load_samples = not args.dry_run_fast

    partition_specs = []
    if args.partition is not None:
        for partition_expr in args.partition:
            parts = partition_expr.split(':')
            if len(parts) != 2:
                fail('Wrong partition specification: "{}"'.format(partition_expr))
            partition_specs.append((float(parts[0]), str(parts[1])))
    partition_specs.sort(key=lambda p: p[0], reverse=True)

    fragments = []
    for audio_path, aligned_path in progress(pairs, desc='Loading alignments'):
        with open(aligned_path, 'r') as aligned_file:
            aligned_fragments = json.load(aligned_file)
        for fragment in aligned_fragments:
            fragment['audio_path'] = audio_path
            fragments.append(fragment)

    if args.filter is not None:
        kept_fragments = []
        for fragment in progress(fragments, desc='Filtering'):
            if not eval(args.filter, {'math': math}, fragment):
                kept_fragments.append(fragment)
        if len(kept_fragments) < len(fragments):
            logging.info('Filtered out {} samples'.format(len(fragments) - len(kept_fragments)))
        fragments = kept_fragments
        if len(fragments) == 0:
            fail('Filter left no samples samples')

    for fragment in progress(fragments, desc='Computing qualities'):
        fragment['quality'] = eval(args.criteria, {'math': math}, fragment)

    def get_meta(fragment, meta_field):
        if 'meta' in fragment:
            meta = fragment['meta']
            if meta_field in meta:
                for value in meta[meta_field]:
                    return value
        return unknown

    if args.debias is not None:
        for debias in args.debias:
            grouped = engroup(fragments, lambda f: get_meta(f, debias))
            if unknown in grouped:
                fragments = grouped[unknown]
                del grouped[unknown]
            else:
                fragments = []
            counts = list(map(lambda f: len(f), grouped.values()))
            mean = statistics.mean(counts)
            sigma = statistics.pstdev(counts, mu=mean)
            cap = int(mean + args.debias_sigma_factor * sigma)
            counter = Counter()
            for group, values in progress(grouped.items(), desc='Debiasing "{}"'.format(debias)):
                if len(values) > cap:
                    values.sort(key=lambda g: g['quality'])
                    counter[group] += len(values) - cap
                    values = values[-cap:]
                fragments.extend(values)
            logging.info('Dropped for debiasing "{}":'.format(debias))
            for group, count in counter.most_common():
                logging.info(' - "{}": {}'.format(group, count))

    def get_partition(f):
        quality = f['quality']
        for minimum, partition_name in partition_specs:
            if quality >= minimum:
                return partition_name
        return 'other'

    lists = {}

    def ensure_list(name):
        lists[name] = []
        if not args.force:
            for p in [name, name + '.' + args.format]:
                if path.exists(path.join(target_dir, p)):
                    fail('"{}" already existing - use --force to ignore'.format(p))

    if args.split_seed is not None:
        random.seed(args.split_seed)
    partitions = engroup(fragments, get_partition)
    for partition, partition_fragments in partitions.items():
        logging.info('Partition "{}":'.format(partition))
        if not args.split or partition == 'other':
            ensure_list(partition)
            for fragment in partition_fragments:
                fragment['list-name'] = partition
            logging.info(' - samples: {}'.format(len(partition_fragments)))
        else:
            train_size, sample_size = get_set_sizes(len(partition_fragments))
            if args.split_field:
                portions = engroup(partition_fragments, lambda f: get_meta(f, args.split_field)).values()
                portions.sort(key=lambda p: len(p))
                train_set, dev_set, test_set = [], [], []
                for offset, sample_set in [(0, dev_set), (1, test_set)]:
                    for portion in portions[offset::2]:
                        if len(sample_set) < sample_size:
                            sample_set.extend(portion)
                        else:
                            train_set.extend(portion)
            else:
                random.shuffle(partition_fragments)
                test_set = partition_fragments[:sample_size]
                partition_fragments = partition_fragments[sample_size:]
                dev_set = partition_fragments[:sample_size]
                train_set = partition_fragments[sample_size:]
            for set_name, set_fragments in [('train', train_set), ('dev', dev_set), ('test', test_set)]:
                list_name = partition + '-' + set_name
                ensure_list(list_name)
                for fragment in set_fragments:
                    fragment['list-name'] = list_name
                logging.info(' - sub-set "{}" - samples: {}'.format(set_name, len(set_fragments)))

    for list_name in lists.keys():
        dir_name = path.join(target_dir, list_name)
        if not path.isdir(dir_name):
            if dry_run:
                logging.debug('Would create directory "{}"'.format(dir_name))
            else:
                os.mkdir(dir_name)

    def list_fragments():
        audio_files = engroup(fragments, lambda f: f['audio_path'])
        pool = Pool(args.workers)
        ls = load_segment if load_samples else load_segment_dry
        for audio_path, audio in pool.imap_unordered(ls, audio_files.keys()):
            file_fragments = audio_files[audio_path]
            if args.channels is not None:
                audio = audio.set_channels(args.channels)
            if args.rate is not None:
                audio = audio.set_frame_rate(args.rate)
            file_fragments.sort(key=lambda f: f['start'])
            for fragment in file_fragments:
                if load_samples:
                    yield audio[fragment['start']:fragment['end']], fragment
                else:
                    yield audio, fragment

    for audio_segment, fragment in progress(list_fragments(), desc='Exporting samples', total=len(fragments)):
        list_name = fragment['list-name']
        group_list = lists[list_name]
        sample_name = 'sample-{:010d}.wav'.format(len(group_list))
        rel_path = path.join(list_name, sample_name)
        abs_path = path.join(target_dir, rel_path)
        if dry_run:
            logging.debug('Would write file "{}"'.format(abs_path))
        else:
            with open(abs_path, "wb") as wav_file:
                audio_segment.export(wav_file, format="wav")
                file_size = wav_file.tell()
            group_list.append((rel_path, file_size, fragment))

    for list_name, group_list in progress(lists.items(), desc='Writing lists'):
        if args.format == 'json':
            json_path = path.join(target_dir, list_name + '.json')
            if dry_run:
                logging.debug('Would write file "{}"'.format(json_path))
            else:
                entries = []
                for rel_path, file_size, fragment in group_list:
                    entry = {
                        'audio': rel_path,
                        'size': file_size,
                        'transcript': fragment['aligned'],
                        'duration': fragment['end'] - fragment['start']
                    }
                    if 'aligned-raw' in fragment:
                        entry['transcript-raw'] = fragment['aligned-raw']
                    entries.append(entry)
                with open(json_path, 'w') as json_file:
                    json.dump(entries, json_file)
        else:
            csv_path = path.join(target_dir, list_name + '.csv')
            if dry_run:
                logging.debug('Would write file "{}"'.format(csv_path))
            else:
                with open(csv_path, 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    for rel_path, file_size, fragment in group_list:
                        writer.writerow([rel_path, file_size, fragment['aligned']])


if __name__ == '__main__':
    main(sys.argv[1:])
    os.system('stty sane')
