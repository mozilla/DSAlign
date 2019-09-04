import os
import sys
import json
import logging
import argparse
import subprocess
import os.path as path
import numpy as np
import wavTranscriber
import multiprocessing
from collections import Counter
from search import FuzzySearch
from tqdm import tqdm
from text import Alphabet, TextCleaner, levenshtein, similarity
from utils import enweight

model = None
sample_rate = 0
worker_index = 0


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


def main(args):
    parser = argparse.ArgumentParser(description='Force align speech data with a transcript.')

    parser.add_argument('audio', type=str,
                        help='Source path of speech audio (WAV format)')
    parser.add_argument('transcript', type=str,
                        help='Source path of original transcript (plain text)')
    parser.add_argument('result', type=str,
                        help='Target path of alignment result file (JSON)')

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

    audio_group = parser.add_argument_group(title='Audio pre-processing options')
    audio_group.add_argument('--audio-vad-aggressiveness', type=int, choices=range(4), required=False,
                             help='Determines how aggressive filtering out non-speech is (default: 3)')

    stt_group = parser.add_argument_group(title='STT options')
    stt_group.add_argument('--stt-model-dir', required=False,
                           help='Path to a directory with output_graph, lm, trie and alphabet files ' +
                                '(default: "data/en"')
    stt_group.add_argument('--stt-no-own-lm', action="store_true",
                           help='Deactivates creation of individual language models per document.' +
                                'Uses the one from model dir instead.')
    stt_group.add_argument('--stt-min-duration', type=int, required=False, default=100,
                           help='Minimum speech fragment duration in milliseconds to translate (default: 100)')
    stt_group.add_argument('--stt-max-duration', type=int, required=False,
                           help='Maximum speech fragment duration in milliseconds to translate (default: no limit)')

    text_group = parser.add_argument_group(title='Text pre-processing options')
    text_group.add_argument('--text-keep-dashes', action="store_true",
                            help='No replacing of dashes with spaces. Dependent of alphabet if kept at all.')
    text_group.add_argument('--text-keep-ws', action="store_true",
                            help='No normalization of whitespace. Keep it as it is.')
    text_group.add_argument('--text-keep-casing', action="store_true",
                            help='No lower-casing of characters. Keep them as they are.')

    align_group = parser.add_argument_group(title='Alignment algorithm options')
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
    align_group.add_argument('--align-stretch-fraction', type=float, required=False, default=1,
                             help='Length fraction of the fragment that it could get stretched for matching')
    align_group.add_argument('--align-snap-factor', type=float, required=False, default=1.5,
                             help='Priority factor for snapping matched texts to word boundaries '
                                  '(default: 1.5 - slightly snappy)')
    align_group.add_argument('--align-min-ngram-size', type=int, required=False, default=1,
                             help='Minimum N-gram size for weighted N-gram similarity during snapping (default: 1)')
    align_group.add_argument('--align-max-ngram-size', type=int, required=False, default=3,
                             help='Maximum N-gram size for weighted N-gram similarity during snapping (default: 3)')
    align_group.add_argument('--align-ngram-size-factor', type=float, required=False, default=1,
                             help='Size weight for weighted N-gram similarity during snapping (default: 1)')
    align_group.add_argument('--align-ngram-position-factor', type=float, required=False, default=2.5,
                             help='Position weight for weighted N-gram similarity during snapping (default: 1)')

    output_group = parser.add_argument_group(title='Output options')
    output_group.add_argument('--output-stt', action="store_true",
                              help='Writes STT transcripts to result file as attribute "transcript"')
    output_group.add_argument('--output-aligned', action="store_true",
                              help='Writes clean aligned original transcripts to result file')
    output_group.add_argument('--output-aligned-raw', action="store_true",
                              help='Writes raw aligned original transcripts to result file')
    output_group.add_argument('--output-wng-min-ngram-size', type=int, required=False, default=1,
                              help='Minimum N-gram size for weighted N-gram similarity filter (default: 1)')
    output_group.add_argument('--output-wng-max-ngram-size', type=int, required=False, default=3,
                              help='Maximum N-gram size for weighted N-gram similarity filter (default: 3)')
    output_group.add_argument('--output-wng-ngram-size-factor', type=int, required=False, default=1,
                              help='Size weight for weighted N-gram similarity filter (default: 1)')
    output_group.add_argument('--output-wng-ngram-position-factor', type=int, required=False, default=3,
                              help='Position weight for weighted N-gram similarity filter (default: 3)')

    named_numbers = {
        'tlen': ('transcript length',          int,   None),
        'mlen': ('match length',               int,   None),
        'SWS':  ('Smith-Waterman score',       float, 'From 0.0 (not equal at all) to 100.0+ (pretty equal)'),
        'WNG':  ('weighted N-gram similarity', float, 'From 0.0 (not equal at all) to 100.0 (totally equal)'),
        'CER':  ('character error rate',       float, 'From 0.0 (no different words) to 100.0+ (total miss)'),
        'WER':  ('word error rate',            float, 'From 0.0 (no wrong characters) to 100.0+ (total miss)')
    }

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
    logging.basicConfig(stream=sys.stderr, level=args.loglevel if args.loglevel else 20)

    def progress(iter, **kwargs):
        return iter if args.no_progress else tqdm(iter, **kwargs)

    logging.debug("Start")

    model_dir = os.path.expanduser(args.stt_model_dir if args.stt_model_dir else 'models/en')
    logging.debug("Looking for model files in %s..." % model_dir)
    output_graph_path, alphabet_path, lang_lm_path, lang_trie_path = wavTranscriber.resolve_models(model_dir)
    logging.debug("Loading alphabet from %s..." % alphabet_path)
    alphabet = Alphabet(alphabet_path)

    logging.debug("Loading original transcript from %s..." % args.transcript)
    with open(args.transcript, 'r') as transcript_file:
        original_transcript = transcript_file.read()
    tc = TextCleaner(alphabet,
                     dashes_to_ws=not args.text_keep_dashes,
                     normalize_space=not args.text_keep_ws,
                     to_lower=not args.text_keep_casing)
    tc.add_original_text(original_transcript)
    clean_text_path = args.transcript + '.clean'
    with open(clean_text_path, 'w') as clean_text_file:
        clean_text_file.write(tc.clean_text)

    transcription_log = os.path.splitext(args.audio)[0] + '.tlog'
    if path.exists(transcription_log):
        logging.debug("Loading transcription log from %s..." % transcription_log)
        with open(transcription_log, 'r') as transcriptions_file:
            fragments = json.loads(transcriptions_file.read())
    else:
        kenlm_path = 'dependencies/kenlm/build/bin'
        if not path.exists(kenlm_path):
            kenlm_path = None
        deepspeech_path = 'dependencies/deepspeech'
        if not path.exists(deepspeech_path):
            deepspeech_path = None
        if kenlm_path and deepspeech_path and not args.stt_no_own_lm:
            arpa_path = args.transcript + '.arpa'
            if not path.exists(arpa_path):
                subprocess.check_call([
                    kenlm_path + '/lmplz',
                    '--text',
                    clean_text_path,
                    '--arpa',
                    arpa_path,
                    '--o',
                    '5'
                ])

            lm_path = args.transcript + '.lm'
            if not path.exists(lm_path):
                subprocess.check_call([
                    kenlm_path + '/build_binary',
                    '-s',
                    arpa_path,
                    lm_path
                ])

            trie_path = args.transcript + '.trie'
            if not path.exists(trie_path):
                subprocess.check_call([
                    deepspeech_path + '/generate_trie',
                    alphabet_path,
                    lm_path,
                    trie_path
                ])
        else:
            lm_path = lang_lm_path
            trie_path = lang_trie_path

        logging.debug('Loading acoustic model from "%s", alphabet from "%s" and language model from "%s"...' %
                      (output_graph_path, alphabet_path, lm_path))

        inference_time = 0.0
        offset = 0

        # Run VAD on the input file
        logging.debug("Transcribing VAD segments...")
        wave_file = args.audio
        aggressiveness = int(args.audio_vad_aggressiveness) if args.audio_vad_aggressiveness else 3
        segments, rate, audio_length = wavTranscriber.vad_segment_generator(wave_file, aggressiveness)

        pool = multiprocessing.Pool(initializer=init_stt,
                                    initargs=(output_graph_path, alphabet_path, lm_path, trie_path, rate),
                                    processes=None)

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
                #logging.debug("Transcribing segment %002d (from %f to %f)..." % (i, time_start / 1000.0, time_end / 1000.0))
                yield (time_start, time_end, np.frombuffer(segment_buffer, dtype=np.int16))

        samples = list(progress(pre_filter(), desc='VAD splitting'))

        transcripts = progress(pool.imap(stt, samples), desc='Transcribing', total=len(samples))

        fragments = []
        for time_start, time_end, segment_transcript in transcripts:
            if segment_transcript is None:
                continue
            fragments.append({
                'start': time_start,
                'end':   time_end,
                'transcript': segment_transcript
            })
        logging.debug("Excluded {} empty transcripts".format(len(transcripts) - len(fragments)))

        logging.debug("Writing transcription log to file %s..." % transcription_log)
        with open(transcription_log, 'w') as transcriptions_file:
            transcriptions_file.write(json.dumps(fragments))

    search = FuzzySearch(tc.clean_text,
                         max_candidates=args.align_max_candidates,
                         candidate_threshold=args.align_candidate_threshold,
                         match_score=args.align_match_score,
                         mismatch_score=args.align_mismatch_score,
                         gap_score=args.align_gap_score)
    result_fragments = []
    substitutions = Counter()
    statistics = Counter()
    end_fragments = (args.start + args.num_samples) if args.num_samples else len(fragments)
    fragments = fragments[args.start:end_fragments]
    for index, fragment in enumerate(fragments):
        fragment['index'] = index
        fragment['transcript'] = fragment['transcript'].strip()

    def skip(index, reason):
        logging.info('Fragment {}: {}'.format(index, reason))
        statistics[reason] += 1

    def split_match(fragments, start=0, end=-1):
        n = len(fragments)
        if n < 1:
            raise StopIteration
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
                raise StopIteration
        for _, _ in weighted_fragments:
            yield None

    matched_fragments = progress(split_match(fragments), desc='Split matching', total=len(fragments))
    matched_fragments = list(filter(lambda f: f is not None, matched_fragments))

    def phrase_similarity(a, b, direction):
        return similarity(a,
                          b,
                          direction=direction,
                          min_ngram_size=args.align_min_ngram_size,
                          max_ngram_size=args.align_max_ngram_size,
                          size_factor=args.align_ngram_size_factor,
                          position_factor=args.align_ngram_position_factor)

    def get_similarities(a, b, gap_text, direction):
        if direction < 0:
            a, b, gap_text = a[::-1], b[::-1], gap_text[::-1]
        n = min(len(gap_text), int(args.align_stretch_fraction * len(b)))
        similarities = list(map(lambda i: (args.align_snap_factor if gap_text[i] == ' ' else 1) *
                                          phrase_similarity(a, b + gap_text[:i], 1),
                                range(n)))
        best = max((v, i) for i, v in enumerate(similarities))[1] if n > 0 else 0
        return best, similarities

    for index in progress(range(len(matched_fragments) + 1), desc='Fine alignment'):
        if index > 0:
            a = matched_fragments[index - 1]
            a_start, a_end = a['match-start'], a['match-end']
        else:
            a = None
            a_start = a_end = 0
        if index < len(matched_fragments):
            b = matched_fragments[index]
            b_start, b_end = b['match-start'], b['match-end']
        else:
            b = None
            b_start = b_end = len(search.text)

        assert a_end <= b_start
        gap_text = search.text[a_end:b_start]
        if a_end == b_start or len(gap_text.strip()) == 0:
            continue

        if a:
            a_best_index, a_similarities = get_similarities(a['transcript'], search.text[a_start:a_end], gap_text, 1)
            a_best_end = a_best_index + a_end
        if b:
            b_best_index, b_similarities = get_similarities(b['transcript'], search.text[b_start:b_end], gap_text, -1)
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
        if should_output or min_val or max_val:
            val = get_value()
            if len(number_key) == 3:
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

    for fragment in progress(matched_fragments, desc='Writing output'):
        index = fragment['index']
        time_start = fragment['start']
        time_end = fragment['end']
        fragment_transcript = fragment['transcript']
        result_fragment = {
            'start': time_start,
            'end':   time_end
        }
        sample_numbers = []

        if apply_number('tlen', index, result_fragment, sample_numbers, lambda: len(fragment_transcript)):
            continue
        if args.output_stt:
            result_fragment['transcript'] = fragment_transcript

        if 'match-start' not in fragment:
            skip(index, 'No match for transcript')
            continue
        match_start, match_end = fragment['match-start'], fragment['match-end']
        original_start = tc.get_original_offset(match_start)
        original_end = tc.get_original_offset(match_end)
        result_fragment['text-start'] = original_start
        result_fragment['text-end'] = original_end

        if args.output_aligned_raw:
            result_fragment['aligned-raw'] = original_transcript[original_start:original_end]

        fragment_matched = tc.clean_text[match_start:match_end]
        if apply_number('mlen', index, result_fragment, sample_numbers, lambda: len(fragment_matched)):
            continue
        if args.output_aligned:
            result_fragment['aligned'] = fragment_matched

        if apply_number('SWS', index, result_fragment, sample_numbers, lambda: 100 * fragment['sws']):
            continue

        if apply_number('WNG', index, result_fragment, sample_numbers,
                       lambda: 100 * similarity(fragment_matched,
                                                fragment_transcript,
                                                min_ngram_size=args.output_wng_min_ngram_size,
                                                max_ngram_size=args.output_wng_max_ngram_size,
                                                size_factor=args.output_wng_ngram_size_factor,
                                                position_factor=args.output_wng_ngram_position_factor)):
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
            tc.clean_text[match_start-args.text_context:match_start],
            fragment_matched,
            tc.clean_text[match_end:match_end+args.text_context]))
        start = match_end
        if args.play:
            subprocess.check_call(['play',
                                   '--no-show-progress',
                                   args.audio,
                                   'trim',
                                   str(time_start / 1000.0),
                                   '='+str(time_end / 1000.0)])
    with open(args.result, 'w') as result_file:
        result_file.write(json.dumps(result_fragments))

    logging.info('Aligned %d fragments' % len(result_fragments))
    skipped = len(fragments) - len(result_fragments)
    logging.info('Skipped %d fragments (%.2f%%):' % (skipped, skipped * 100.0 / len(fragments)))
    for key, number in statistics.most_common():
        logging.info(' - %s: %d' % (key, number))


if __name__ == '__main__':
    main(sys.argv[1:])
