import os
import sys
import json
import logging
import argparse
import subprocess
import os.path as path
import numpy as np
import wavTranscriber
from search import FuzzySearch
from text import Alphabet, TextCleaner, levenshtein


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
    parser.add_argument('--play', action="store_true",
                        help='Play audio fragments as they are matched using SoX audio tool')
    parser.add_argument('--text-context', type=int, required=False, default=10,
                        help='Size of textual context for logged statements - default: 10')

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
    align_group.add_argument('--align-candidate-threshold', type=float, required=False, default=0.8,
                             help='Factor for how many 3grams the next candidate should have at least ' +
                                  'compared to its predecessor (default: 0.8)')
    align_group.add_argument('--align-no-snap-to-token', action="store_true",
                             help='Deactivates snapping to similar neighbour tokens ' +
                                  'at the beginning and end of each phrase')
    align_group.add_argument('--align-stretch-fraction', type=float, required=False, default=1/3,
                             help='Fraction of its original length that a phrase could get expanded or shrunken ' +
                                  'to match the original text (default: 0.33)')

    output_group = parser.add_argument_group(title='Output options')
    output_group.add_argument('--output-stt', action="store_true",
                              help='Writes STT transcripts to result file')
    output_group.add_argument('--output-aligned', action="store_true",
                              help='Writes clean aligned original transcripts to result file')
    output_group.add_argument('--output-aligned-raw', action="store_true",
                              help='Writes raw aligned original transcripts to result file')
    output_group.add_argument('--output-min-length', type=int, required=False,
                              help='Minimum phrase length (default: no limit)')
    output_group.add_argument('--output-max-length', type=int, required=False,
                              help='Maximum phrase length (default: no limit)')
    for b in ['Min', 'Max']:
        for r in ['CER', 'WER']:
            output_group.add_argument('--output-' + b.lower() + '-' + r.lower(), type=float, required=False,
                                      help=b + 'imum ' + ('character' if r == 'CER' else 'word') +
                                           ' error rate (' + r + ') the STT transcript of the audio ' +
                                           'has to have when compared with the original text')

    args = parser.parse_args()

    # Debug helpers
    logging.basicConfig(stream=sys.stderr, level=args.loglevel if args.loglevel else 20)
    logging.debug("Start")

    fragments = []
    fragments_cache_path = args.result + '.cache'
    model_dir = os.path.expanduser(args.stt_model_dir if args.stt_model_dir else 'models/en')
    logging.debug("Looking for model files in %s..." % model_dir)
    output_graph_path, alphabet_path, lang_lm_path, lang_trie_path = wavTranscriber.resolve_models(model_dir)
    logging.debug("Loading alphabet from %s..." % alphabet_path)
    alphabet = Alphabet(alphabet_path)

    logging.debug("Loading original transcript from %s..." % args.transcript)
    with open(args.transcript, 'r') as transcript_file:
        original_transcript = transcript_file.read()
    tc = TextCleaner(original_transcript,
                          alphabet,
                          dashes_to_ws=not args.text_keep_dashes,
                          normalize_space=not args.text_keep_ws,
                          to_lower=not args.text_keep_casing)
    clean_text_path = args.transcript + '.clean'
    with open(clean_text_path, 'w') as clean_text_file:
        clean_text_file.write(tc.clean_text)

    if path.exists(fragments_cache_path):
        logging.debug("Loading cached segment transcripts from %s..." % fragments_cache_path)
        with open(fragments_cache_path, 'r') as result_file:
            fragments = json.loads(result_file.read())
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
        model, _, _ = wavTranscriber.load_model(output_graph_path, alphabet_path, lm_path, trie_path)

        inference_time = 0.0
        offset = 0

        # Run VAD on the input file
        logging.debug("Transcribing VAD segments...")
        wave_file = args.audio
        aggressiveness = int(args.audio_vad_aggressiveness) if args.audio_vad_aggressiveness else 3
        segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(wave_file, aggressiveness)

        for i, segment in enumerate(segments):
            # Run DeepSpeech on the chunk that just completed VAD
            segment_buffer, time_start, time_end = segment
            logging.debug("Transcribing segment %002d (from %f to %f)..." % (i, time_start / 1000.0, time_end / 1000.0))

            audio = np.frombuffer(segment_buffer, dtype=np.int16)
            segment_transcript, segment_inference_time = wavTranscriber.stt(model, audio, sample_rate)
            segment_transcript = ' '.join(segment_transcript.split())
            inference_time += segment_inference_time
            if segment_transcript is None:
                logging.debug("Segment %002d empty" % i)
                continue
            fragments.append({
                'time-start':  time_start,
                'time-length': time_end-time_start,
                'transcript':  segment_transcript
            })
            offset += len(segment_transcript)

        logging.debug("Writing segment transcripts to cache file %s..." % fragments_cache_path)
        with open(fragments_cache_path, 'w') as result_file:
            result_file.write(json.dumps(fragments))

    search = FuzzySearch(tc.clean_text,
                         max_candidates=args.align_max_candidates,
                         candidate_threshold=args.align_candidate_threshold,
                         snap_token=not args.align_no_snap_to_token,
                         stretch_factor=args.align_stretch_fraction)
    start = 0
    result_fragments = []

    # DEBUG
    fragments = fragments[14:15]

    for fragment in fragments:
        time_start = fragment['time-start']
        time_length = fragment['time-length']
        fragment_transcript = fragment['transcript']
        match, match_distance = search.find_best(fragment_transcript)
        if match is not None:
            fragment_matched = tc.clean_text[match.start:match.end]
            cer = levenshtein(fragment_transcript, fragment_matched)/len(fragment_matched)
            wer = levenshtein(fragment_transcript.split(), fragment_matched.split())/len(fragment_matched.split())
            if (args.output_min_cer and cer * 100.0 < args.output_min_cer) or \
               (args.output_max_cer and cer * 100.0 > args.output_max_cer) or \
               (args.output_min_wer and wer * 100.0 < args.output_min_wer) or \
               (args.output_max_wer and wer * 100.0 > args.output_max_wer) or \
               (args.output_min_length and len(fragment_matched) < args.output_min_length) or \
               (args.output_max_length and len(fragment_matched) > args.output_max_length):
                continue
            original_start = tc.get_original_offset(match.start)
            original_end = tc.get_original_offset(match.end)
            result_fragment = {
                'time-start':  time_start,
                'time-length': time_length,
                'text-start':  original_start,
                'text-length': original_end-original_start,
                'cer':         cer,
                'wer':         wer
            }
            if args.output_stt:
                result_fragment['stt'] = fragment_transcript
            if args.output_aligned:
                result_fragment['aligned'] = fragment_matched
            if args.output_aligned_raw:
                result_fragment['aligned-raw'] = original_transcript[original_start:original_end]
            result_fragments.append(result_fragment)
            logging.debug('Sample with WER %.2f CER %.2f' % (wer * 100, cer * 100))
            logging.debug('- T:  ' + args.text_context * ' ' + '%s' % fragment_transcript)
            logging.debug('- O: %s|%s|%s' % (
                tc.clean_text[match.start-args.text_context:match.start],
                fragment_matched,
                tc.clean_text[match.end:match.end+args.text_context]))
            start = match.end
            if args.play:
                subprocess.check_call(['play',
                                       '--no-show-progress',
                                       args.audio,
                                       'trim',
                                       str(time_start/1000.0),
                                       '='+str((time_start + time_length)/1000.0)])
    with open(args.result, 'w') as result_file:
        result_file.write(json.dumps(result_fragments))

    logging.info('Aligned %d fragments.' % len(result_fragments))
    skipped = len(fragments)-len(result_fragments)
    logging.info('Skipped %d fragments (%.2f%%).' % (skipped, skipped*100.0/len(fragments)))

if __name__ == '__main__':
    main(sys.argv[1:])
