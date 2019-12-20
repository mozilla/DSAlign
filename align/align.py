import os
import sys
import json
import logging
import argparse
import subprocess
import os.path as path
import numpy as np
import textdistance
import wavSplit
import wavTranscriber
import multiprocessing
from collections import Counter
from search import FuzzySearch
from tqdm import tqdm
from text import Alphabet, TextCleaner, levenshtein, similarity
from utils import enweight

algos = ['WNG', 'jaro_winkler', 'editex', 'levenshtein', 'mra', 'hamming']
sim_desc = 'From 0.0 (not equal at all) to 100.0 (totally equal)'
named_numbers = {
    'tlen': ('transcript length', int, None),
    'mlen': ('match length', int, None),
    'SWS': ('Smith-Waterman score', float, 'From 0.0 (not equal at all) to 100.0+ (pretty equal)'),
    'WNG': ('weighted N-gram similarity', float, sim_desc),
    'jaro_winkler': ('Jaro-Winkler similarity', float, sim_desc),
    'editex': ('Editex similarity', float, sim_desc),
    'levenshtein': ('Levenshtein similarity', float, sim_desc),
    'mra': ('MRA similarity', float, sim_desc),
    'hamming': ('Hamming similarity', float, sim_desc),
    'CER': ('character error rate', float, 'From 0.0 (no different words) to 100.0+ (total miss)'),
    'WER': ('word error rate', float, 'From 0.0 (no wrong characters) to 100.0+ (total miss)')
}

args = None
model = None
sample_rate = 0
alphabet = None


def fail(message, code=1):
    logging.fatal(message)
    exit(code)


def read_script(script_path):
    tc = TextCleaner(alphabet,
                     dashes_to_ws=not args.text_keep_dashes,
                     normalize_space=not args.text_keep_ws,
                     to_lower=not args.text_keep_casing)
    with open(script_path, 'r') as script_file:
        content = script_file.read()
        if script_path.endswith('.script'):
            for phrase in json.loads(content):
                tc.add_original_text(phrase['text'], meta=phrase)
        elif args.text_meaningful_newlines:
            for phrase in content.split('\n'):
                tc.add_original_text(phrase)
        else:
            tc.add_original_text(content)
    return tc


def init_stt(output_graph_path, alphabet_path, lm_path, trie_path, rate):
    global model, sample_rate
    sample_rate = rate
    logging.debug('Process {}: Loaded models'.format(os.getpid()))
    model = wavTranscriber.load_model(output_graph_path, alphabet_path, lm_path, trie_path)


def stt(sample):
    time_start, time_end, audio = sample
    logging.debug('Process {}: Transcribing...'.format(os.getpid()))
    transcript = wavTranscriber.stt(model, audio, sample_rate)
    logging.debug('Process {}: {}'.format(os.getpid(), transcript))
    return time_start, time_end, ' '.join(transcript.split())


def init_align(w_args, w_alphabet):
    global args, alphabet
    args = w_args
    alphabet = w_alphabet


def align(triple):
    tlog, script, aligned = triple

    logging.debug("Loading script from %s..." % script)
    tc = read_script(script)
    search = FuzzySearch(tc.clean_text,
                         max_candidates=args.align_max_candidates,
                         candidate_threshold=args.align_candidate_threshold,
                         match_score=args.align_match_score,
                         mismatch_score=args.align_mismatch_score,
                         gap_score=args.align_gap_score)

    logging.debug("Loading transcription log from %s..." % tlog)
    with open(tlog, 'r') as transcription_log_file:
        fragments = json.load(transcription_log_file)
    end_fragments = (args.start + args.num_samples) if args.num_samples else len(fragments)
    fragments = fragments[args.start:end_fragments]
    for index, fragment in enumerate(fragments):
        meta = {}
        for key, value in list(fragment.items()):
            if key not in ['start', 'end', 'transcript']:
                meta[key] = value
                del fragment[key]
        fragment['meta'] = meta
        fragment['index'] = index
        fragment['transcript'] = fragment['transcript'].strip()

    reasons = Counter()

    def skip(index, reason):
        logging.info('Fragment {}: {}'.format(index, reason))
        reasons[reason] += 1

    def split_match(fragments, start=0, end=-1):
        n = len(fragments)
        if n < 1:
            return
        elif n == 1:
            weighted_fragments = [(0, fragments[0])]
        else:
            # so we later know the original index of each fragment
            weighted_fragments = enumerate(fragments)
            # assigns high values to long statements near the center of the list
            weighted_fragments = enweight(weighted_fragments)
            weighted_fragments = map(lambda fw: (fw[0], (1 - fw[1]) * len(fw[0][1]['transcript'])), weighted_fragments)
            # fragments with highest weights first
            weighted_fragments = sorted(weighted_fragments, key=lambda fw: fw[1], reverse=True)
            # strip weights
            weighted_fragments = list(map(lambda fw: fw[0], weighted_fragments))
        for index, fragment in weighted_fragments:
            match = search.find_best(fragment['transcript'], start=start, end=end)
            match_start, match_end, sws_score, match_substitutions = match
            if sws_score > (n - 1) / (2 * n):
                fragment['match-start'] = match_start
                fragment['match-end'] = match_end
                fragment['sws'] = sws_score
                fragment['substitutions'] = match_substitutions
                for f in split_match(fragments[0:index], start=start, end=match_start):
                    yield f
                yield fragment
                for f in split_match(fragments[index + 1:], start=match_end, end=end):
                    yield f
                return
        for _, _ in weighted_fragments:
            yield None

    matched_fragments = split_match(fragments)
    matched_fragments = list(filter(lambda f: f is not None, matched_fragments))

    similarity_algos = {}

    def phrase_similarity(algo, a, b):
        if algo in similarity_algos:
            return similarity_algos[algo](a, b)
        algo_impl = lambda aa, bb: None
        if algo.lower() == 'wng':
            algo_impl = similarity_algos[algo] = lambda aa, bb: similarity(
                aa,
                bb,
                direction=1,
                min_ngram_size=args.align_wng_min_size,
                max_ngram_size=args.align_wng_max_size,
                size_factor=args.align_wng_size_factor,
                position_factor=args.align_wng_position_factor)
        elif algo in algos:
            algo_impl = similarity_algos[algo] = getattr(textdistance, algo).normalized_similarity
        else:
            logging.fatal('Unknown similarity metric "{}"'.format(algo))
            exit(1)
        return algo_impl(a, b)

    def get_similarities(a, b, n, gap_text, gap_meta, direction):
        if direction < 0:
            a, b, gap_text, gap_meta = a[::-1], b[::-1], gap_text[::-1], gap_meta[::-1]
        similarities = list(map(
            lambda i: (args.align_word_snap_factor if gap_text[i + 1] == ' ' else 1) *
                      (args.align_phrase_snap_factor if gap_meta[i + 1] is None else 1) *
                      (phrase_similarity(args.align_similarity_algo, a, b + gap_text[1:i + 1])),
            range(n)))
        best = max((v, i) for i, v in enumerate(similarities))[1] if n > 0 else 0
        return best, similarities

    for index in range(len(matched_fragments) + 1):
        if index > 0:
            a = matched_fragments[index - 1]
            a_start, a_end = a['match-start'], a['match-end']
            a_len = a_end - a_start
            a_stretch = int(a_len * args.align_stretch_fraction)
            a_shrink = int(a_len * args.align_shrink_fraction)
            a_end = a_end - a_shrink
            a_ext = a_shrink + a_stretch
        else:
            a = None
            a_start = a_end = 0
        if index < len(matched_fragments):
            b = matched_fragments[index]
            b_start, b_end = b['match-start'], b['match-end']
            b_len = b_end - b_start
            b_stretch = int(b_len * args.align_stretch_fraction)
            b_shrink = int(b_len * args.align_shrink_fraction)
            b_start = b_start + b_shrink
            b_ext = b_shrink + b_stretch
        else:
            b = None
            b_start = b_end = len(search.text)

        assert a_end <= b_start
        assert a_start <= a_end
        assert b_start <= b_end
        if a_end == b_start or a_start == a_end or b_start == b_end:
            continue
        gap_text = tc.clean_text[a_end - 1:b_start + 1]
        gap_meta = tc.meta[a_end - 1:b_start + 1]

        if a:
            a_best_index, a_similarities = get_similarities(a['transcript'],
                                                            tc.clean_text[a_start:a_end],
                                                            min(len(gap_text) - 1, a_ext),
                                                            gap_text,
                                                            gap_meta,
                                                            1)
            a_best_end = a_best_index + a_end
        if b:
            b_best_index, b_similarities = get_similarities(b['transcript'],
                                                            tc.clean_text[b_start:b_end],
                                                            min(len(gap_text) - 1, b_ext),
                                                            gap_text,
                                                            gap_meta,
                                                            -1)
            b_best_start = b_start - b_best_index

        if a and b and a_best_end > b_best_start:
            overlap_start = b_start - len(b_similarities)
            a_similarities = a_similarities[overlap_start - a_end:]
            b_similarities = b_similarities[:len(a_similarities)]
            best_index = max((sum(v), i) for i, v in enumerate(zip(a_similarities, b_similarities)))[1]
            a_best_end = b_best_start = overlap_start + best_index

        if a:
            a['match-end'] = a_best_end
        if b:
            b['match-start'] = b_best_start

    def apply_number(number_key, index, fragment, show, get_value):
        kl = number_key.lower()
        should_output = getattr(args, 'output_' + kl)
        min_val, max_val = getattr(args, 'output_min_' + kl), getattr(args, 'output_max_' + kl)
        if kl.endswith('len') and min_val is None:
            min_val = 1
        if should_output or min_val or max_val:
            val = get_value()
            if not kl.endswith('len'):
                show.insert(0, '{}: {:.2f}'.format(number_key, val))
                if should_output:
                    fragment[kl] = val
            reason_base = '{} ({})'.format(named_numbers[number_key][0], number_key)
            reason = None
            if min_val and val < min_val:
                reason = reason_base + ' too low'
            elif max_val and val > max_val:
                reason = reason_base + ' too high'
            if reason:
                skip(index, reason)
                return True
        return False

    substitutions = Counter()
    result_fragments = []
    for fragment in matched_fragments:
        index = fragment['index']
        time_start = fragment['start']
        time_end = fragment['end']
        fragment_transcript = fragment['transcript']
        result_fragment = {
            'start': time_start,
            'end': time_end
        }
        sample_numbers = []

        if apply_number('tlen', index, result_fragment, sample_numbers, lambda: len(fragment_transcript)):
            continue
        result_fragment['transcript'] = fragment_transcript

        if 'match-start' not in fragment or 'match-end' not in fragment:
            skip(index, 'No match for transcript')
            continue
        match_start, match_end = fragment['match-start'], fragment['match-end']
        if match_end - match_start <= 0:
            skip(index, 'Empty match for transcript')
            continue
        original_start = tc.get_original_offset(match_start)
        original_end = tc.get_original_offset(match_end)
        result_fragment['text-start'] = original_start
        result_fragment['text-end'] = original_end

        meta_dict = {}
        for meta in list(tc.collect_meta(match_start, match_end)) + [fragment['meta']]:
            for key, value in meta.items():
                if key == 'text':
                    continue
                if key in meta_dict:
                    values = meta_dict[key]
                else:
                    values = meta_dict[key] = []
                if value not in values:
                    values.append(value)
        result_fragment['meta'] = meta_dict

        result_fragment['aligned-raw'] = tc.original_text[original_start:original_end]

        fragment_matched = tc.clean_text[match_start:match_end]
        if apply_number('mlen', index, result_fragment, sample_numbers, lambda: len(fragment_matched)):
            continue
        result_fragment['aligned'] = fragment_matched

        if apply_number('SWS', index, result_fragment, sample_numbers, lambda: 100 * fragment['sws']):
            continue

        should_skip = False
        for algo in algos:
            should_skip = should_skip or apply_number(algo, index, result_fragment, sample_numbers,
                                                      lambda: 100 * phrase_similarity(algo,
                                                                                      fragment_matched,
                                                                                      fragment_transcript))
        if should_skip:
            continue

        if apply_number('CER', index, result_fragment, sample_numbers,
                        lambda: 100 * levenshtein(fragment_transcript, fragment_matched) /
                                len(fragment_matched)):
            continue

        if apply_number('WER', index, result_fragment, sample_numbers,
                        lambda: 100 * levenshtein(fragment_transcript.split(), fragment_matched.split()) /
                                len(fragment_matched.split())):
            continue

        substitutions += fragment['substitutions']

        result_fragments.append(result_fragment)
        logging.debug('Fragment %d aligned with %s' % (index, ' '.join(sample_numbers)))
        logging.debug('- T: ' + args.text_context * ' ' + '"%s"' % fragment_transcript)
        logging.debug('- O: %s|%s|%s' % (
            tc.clean_text[match_start - args.text_context:match_start],
            fragment_matched,
            tc.clean_text[match_end:match_end + args.text_context]))
        if args.play:
            subprocess.check_call(['play',
                                   '--no-show-progress',
                                   args.audio,
                                   'trim',
                                   str(time_start / 1000.0),
                                   '=' + str(time_end / 1000.0)])
    with open(aligned, 'w') as result_file:
        result_file.write(json.dumps(result_fragments, indent=4 if args.output_pretty else None))
    return aligned, len(result_fragments), len(fragments) - len(result_fragments), reasons


def main():
    global args, alphabet
    parser = argparse.ArgumentParser(description='Force align speech data with a transcript.')

    parser.add_argument('--audio', type=str,
                        help='Path to speech audio file')
    parser.add_argument('--tlog', type=str,
                        help='Path to STT transcription log (.tlog)')
    parser.add_argument('--script', type=str,
                        help='Path to original transcript (plain text or .script file)')
    parser.add_argument('--catalog', type=str,
                        help='Path to a catalog file with paths to transcription log or audio, original script and '
                             '(target) alignment files')
    parser.add_argument('--aligned', type=str,
                        help='Alignment result file (.aligned)')
    parser.add_argument('--force', action="store_true",
                        help='Overwrite existing files')
    parser.add_argument('--ignore-missing', action="store_true",
                        help='Ignores catalog entries with missing paths')
    parser.add_argument('--loglevel', type=int, required=False, default=20,
                        help='Log level (between 0 and 50) - default: 20')
    parser.add_argument('--no-progress', action="store_true",
                        help='Prevents showing progress bars')
    parser.add_argument('--play', action="store_true",
                        help='Play audio fragments as they are matched using SoX audio tool')
    parser.add_argument('--text-context', type=int, required=False, default=10,
                        help='Size of textual context for logged statements - default: 10')
    parser.add_argument('--start', type=int, required=False, default=0,
                        help='Start alignment process at given offset of transcribed fragments')
    parser.add_argument('--num-samples', type=int, required=False,
                        help='Number of fragments to align')
    parser.add_argument('--alphabet', required=False,
                        help='Path to an alphabet file (overriding the one from --stt-model-dir)')

    audio_group = parser.add_argument_group(title='Audio pre-processing options')
    audio_group.add_argument('--audio-vad-aggressiveness', type=int, choices=range(4), required=False,
                             help='Determines how aggressive filtering out non-speech is (default: 3)')

    stt_group = parser.add_argument_group(title='STT options')
    stt_group.add_argument('--stt-model-dir', required=False,
                           help='Path to a directory with output_graph, lm, trie and (optional) alphabet file ' +
                                '(default: "data/en"')
    stt_group.add_argument('--stt-no-own-lm', action="store_true",
                           help='Deactivates creation of individual language models per document.' +
                                'Uses the one from model dir instead.')
    stt_group.add_argument('--stt-workers', type=int, required=False, default=1,
                           help='Number of parallel STT workers - should 1 for GPU based DeepSpeech')
    stt_group.add_argument('--stt-min-duration', type=int, required=False, default=100,
                           help='Minimum speech fragment duration in milliseconds to translate (default: 100)')
    stt_group.add_argument('--stt-max-duration', type=int, required=False,
                           help='Maximum speech fragment duration in milliseconds to translate (default: no limit)')

    text_group = parser.add_argument_group(title='Text pre-processing options')
    text_group.add_argument('--text-meaningful-newlines', action="store_true",
                            help='Newlines from plain text file separate phrases/speakers. '
                                 '(see --align-phrase-snap-factor)')
    text_group.add_argument('--text-keep-dashes', action="store_true",
                            help='No replacing of dashes with spaces. Dependent of alphabet if kept at all.')
    text_group.add_argument('--text-keep-ws', action="store_true",
                            help='No normalization of whitespace. Keep it as it is.')
    text_group.add_argument('--text-keep-casing', action="store_true",
                            help='No lower-casing of characters. Keep them as they are.')

    align_group = parser.add_argument_group(title='Alignment algorithm options')
    align_group.add_argument('--align-workers', type=int, required=False,
                             help='Number of parallel alignment workers - defaults to number of CPUs')
    align_group.add_argument('--align-max-candidates', type=int, required=False, default=10,
                             help='How many global 3gram match candidates are tested at max (default: 10)')
    align_group.add_argument('--align-candidate-threshold', type=float, required=False, default=0.92,
                             help='Factor for how many 3grams the next candidate should have at least ' +
                                  'compared to its predecessor (default: 0.92)')
    align_group.add_argument('--align-match-score', type=int, required=False, default=100,
                             help='Matching score for Smith-Waterman alignment (default: 100)')
    align_group.add_argument('--align-mismatch-score', type=int, required=False, default=-100,
                             help='Mismatch score for Smith-Waterman alignment (default: -100)')
    align_group.add_argument('--align-gap-score', type=int, required=False, default=-100,
                             help='Gap score for Smith-Waterman alignment (default: -100)')
    align_group.add_argument('--align-shrink-fraction', type=float, required=False, default=0.1,
                             help='Length fraction of the fragment that it could get shrinked during fine alignment')
    align_group.add_argument('--align-stretch-fraction', type=float, required=False, default=0.25,
                             help='Length fraction of the fragment that it could get stretched during fine alignment')
    align_group.add_argument('--align-word-snap-factor', type=float, required=False, default=1.5,
                             help='Priority factor for snapping matched texts to word boundaries '
                                  '(default: 1.5 - slightly snappy)')
    align_group.add_argument('--align-phrase-snap-factor', type=float, required=False, default=1.0,
                             help='Priority factor for snapping matched texts to word boundaries '
                                  '(default: 1.0 - no snapping)')
    align_group.add_argument('--align-similarity-algo', type=str, required=False, default='wng',
                             help='Similarity algorithm during fine-alignment - one of '
                                  'wng|editex|levenshtein|mra|hamming|jaro_winkler (default: wng)')
    align_group.add_argument('--align-wng-min-size', type=int, required=False, default=1,
                             help='Minimum N-gram size for weighted N-gram similarity '
                                  'during fine-alignment (default: 1)')
    align_group.add_argument('--align-wng-max-size', type=int, required=False, default=3,
                             help='Maximum N-gram size for weighted N-gram similarity '
                                  'during fine-alignment (default: 3)')
    align_group.add_argument('--align-wng-size-factor', type=float, required=False, default=1,
                             help='Size weight for weighted N-gram similarity '
                                  'during fine-alignment (default: 1)')
    align_group.add_argument('--align-wng-position-factor', type=float, required=False, default=2.5,
                             help='Position weight for weighted N-gram similarity '
                                  'during fine-alignment (default: 2.5)')

    output_group = parser.add_argument_group(title='Output options')
    output_group.add_argument('--output-pretty', action="store_true",
                              help='Writes indented JSON output"')

    for short in named_numbers.keys():
        long, atype, desc = named_numbers[short]
        desc = (' - value range: ' + desc) if desc else ''
        output_group.add_argument('--output-' + short.lower(), action="store_true",
                                  help='Writes {} ({}) to output'.format(long, short))
        for extreme in ['Min', 'Max']:
            output_group.add_argument('--output-' + extreme.lower() + '-' + short.lower(), type=atype, required=False,
                                      help='{}imum {} ({}) the STT transcript of the audio '
                                           'has to have when compared with the original text{}'
                                           .format(extreme, long, short, desc))

    args = parser.parse_args()

    # Debug helpers
    logging.basicConfig(stream=sys.stdout, level=args.loglevel if args.loglevel else 20)

    def progress(iter, **kwargs):
        return iter if args.no_progress else tqdm(iter, **kwargs)

    def resolve(base_path, spec_path):
        if spec_path is None:
            return None
        if not path.isabs(spec_path):
            spec_path = path.join(base_path, spec_path)
        return spec_path

    def exists(file_path):
        if file_path is None:
            return False
        return os.path.isfile(file_path)

    to_prepare = []

    def enqueue_or_fail(audio, tlog, script, aligned, prefix=''):
        if exists(aligned) and not args.force:
            fail(prefix + 'Alignment file "{}" already existing - use --force to overwrite'.format(aligned))
        if tlog is None:
            if args.ignore_missing:
                return
            fail(prefix + 'Missing transcription log path')
        if not exists(audio) and not exists(tlog):
            if args.ignore_missing:
                return
            fail(prefix + 'Both audio file "{}" and transcription log "{}" are missing'.format(audio, tlog))
        if not exists(script):
            if args.ignore_missing:
                return
            fail(prefix + 'Missing script "{}"'.format(script))
        to_prepare.append((audio, tlog, script, aligned))

    if (args.audio or args.tlog) and args.script and args.aligned and not args.catalog:
        enqueue_or_fail(args.audio, args.tlog, args.script, args.aligned)
    elif args.catalog:
        if not exists(args.catalog):
            fail('Unable to load catalog file "{}"'.format(args.catalog))
        catalog = path.abspath(args.catalog)
        catalog_dir = path.dirname(catalog)
        with open(catalog, 'r') as catalog_file:
            catalog_entries = json.load(catalog_file)
        for entry in progress(catalog_entries, desc='Reading catalog'):
            enqueue_or_fail(resolve(catalog_dir, entry['audio']),
                            resolve(catalog_dir, entry['tlog']),
                            resolve(catalog_dir, entry['script']),
                            resolve(catalog_dir, entry['aligned']),
                            prefix='Problem loading catalog "{}" - '.format(catalog))
    else:
        fail('You have to either specify a combination of "--audio/--tlog,--script,--aligned" or "--catalog"')

    logging.debug('Start')

    app_root = os.environ['APP_ROOT'] if 'APP_ROOT' in os.environ else os.curdir
    model_dir = os.path.expanduser(args.stt_model_dir if args.stt_model_dir else os.path.join(app_root, 'models', 'en'))

    if args.alphabet is not None:
        alphabet_path = args.alphabet
    else:
        alphabet_path = os.path.join(model_dir, 'alphabet.txt')
    if not os.path.isfile(alphabet_path):
        fail('Found no alphabet file')
    logging.debug('Loading alphabet from "{}"...'.format(alphabet_path))
    alphabet = Alphabet(alphabet_path)

    to_align = []
    output_graph_path = None
    for audio, tlog, script, aligned in to_prepare:
        if not exists(tlog):
            if output_graph_path is None:
                logging.debug('Looking for model files in "{}"...'.format(model_dir))
                output_graph_path, lang_lm_path, lang_trie_path = wavTranscriber.resolve_models(model_dir)
            kenlm_path = os.path.join(app_root, 'dependencies', 'kenlm', 'build', 'bin')
            if not path.exists(kenlm_path):
                kenlm_path = None
            deepspeech_path = os.path.join(app_root, 'dependencies', 'deepspeech')
            if not path.exists(deepspeech_path):
                deepspeech_path = None
            if kenlm_path is not None and deepspeech_path is not None and not args.stt_no_own_lm:
                tc = read_script(script)
                clean_text_path = script + '.clean'
                with open(clean_text_path, 'w') as clean_text_file:
                    clean_text_file.write(tc.clean_text)

                arpa_path = script + '.arpa'
                if not path.exists(arpa_path):
                    subprocess.check_call([
                        os.path.join(kenlm_path, 'lmplz'),
                        '--text',
                        clean_text_path,
                        '--arpa',
                        arpa_path,
                        '--o',
                        '5'
                    ])

                lm_path = script + '.lm'
                if not path.exists(lm_path):
                    subprocess.check_call([
                        os.path.join(kenlm_path, 'build_binary'),
                        '-s',
                        arpa_path,
                        lm_path
                    ])

                trie_path = script + '.trie'
                if not path.exists(trie_path):
                    subprocess.check_call([
                        os.path.join(deepspeech_path, 'generate_trie'),
                        alphabet_path,
                        lm_path,
                        trie_path
                    ])
            else:
                lm_path = lang_lm_path
                trie_path = lang_trie_path

            logging.debug('Loading acoustic model from "{}", alphabet from "{}", trie from "{}" and language model from "{}"...'
                          .format(output_graph_path, alphabet_path, trie_path, lm_path))

            # Run VAD on the input file
            logging.debug('Transcribing VAD segments...')
            aggressiveness = int(args.audio_vad_aggressiveness) if args.audio_vad_aggressiveness else 3
            segments, rate, audio_length = wavSplit.vad_segment_generator(audio, aggressiveness)

            def pre_filter():
                for i, segment in enumerate(segments):
                    segment_buffer, time_start, time_end = segment
                    time_length = time_end - time_start
                    if args.stt_min_duration and time_length < args.stt_min_duration:
                        logging.info('Fragment {}: Audio too short for STT'.format(i))
                        continue
                    if args.stt_max_duration and time_length > args.stt_max_duration:
                        logging.info('Fragment {}: Audio too long for STT'.format(i))
                        continue
                    yield (time_start, time_end, np.frombuffer(segment_buffer, dtype=np.int16))

            samples = list(progress(pre_filter(), desc='VAD splitting'))

            pool = multiprocessing.Pool(initializer=init_stt,
                                        initargs=(output_graph_path, alphabet_path, lm_path, trie_path, rate),
                                        processes=args.stt_workers)
            transcripts = progress(pool.imap(stt, samples), desc='Transcribing', total=len(samples))

            fragments = []
            for time_start, time_end, segment_transcript in transcripts:
                if segment_transcript is None:
                    continue
                fragments.append({
                    'start': time_start,
                    'end': time_end,
                    'transcript': segment_transcript
                })
            logging.debug('Excluded {} empty transcripts'.format(len(transcripts) - len(fragments)))

            logging.debug('Writing transcription log to file "{}"...'.format(tlog))
            with open(tlog, 'w') as tlog_file:
                tlog_file.write(json.dumps(fragments, indent=4 if args.output_pretty else None))
        if not path.isfile(tlog):
            fail('Problem loading transcript from "{}"'.format(tlog))
        to_align.append((tlog, script, aligned))

    total_fragments = 0
    dropped_fragments = 0
    reasons = Counter()

    index = 0
    pool = multiprocessing.Pool(initializer=init_align, initargs=(args, alphabet), processes=args.align_workers)
    for aligned_file, file_total_fragments, file_dropped_fragments, file_reasons in \
            progress(pool.imap_unordered(align, to_align), desc='Aligning', total=len(to_align)):
        if args.no_progress:
            index += 1
            logging.info('Aligned file {} of {} - wrote results to "{}"'.format(index, len(to_align), aligned_file))
        total_fragments += file_total_fragments
        dropped_fragments += file_dropped_fragments
        reasons += file_reasons

    logging.info('Aligned {} fragments'.format(total_fragments))
    if total_fragments > 0 and dropped_fragments > 0:
        logging.info('Dropped {} fragments {:0.2f}%:'.format(dropped_fragments,
                                                             dropped_fragments * 100.0 / total_fragments))
        for key, number in reasons.most_common():
            logging.info(' - {}: {}'.format(key, number))


if __name__ == '__main__':
    main()
