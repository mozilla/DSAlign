import sys
import os
import text
import json
import logging
import argparse
import subprocess
import os.path as path
import numpy as np
import wavTranscriber


def main(args):
    parser = argparse.ArgumentParser(description='Force align speech data with a transcript.')
    parser.add_argument('audio', type=str,
                        help='Source path of speech audio (WAV format)')
    parser.add_argument('transcript', type=str,
                        help='Source path of original transcript (plain text)')
    parser.add_argument('result', type=str,
                        help='Target path of alignment result file (JSON)')
    parser.add_argument('--aggressive', type=int, choices=range(4), required=False,
                        help='Determines how aggressive filtering out non-speech is. (Interger between 0-3)')
    parser.add_argument('--model', required=False,
                        help='Path to directory that contains all model files (output_graph, lm, trie and alphabet)')
    parser.add_argument('--loglevel', type=int, required=False,
                        help='Log level (between 0 and 50) - default: 20')
    parser.add_argument('--play', action="store_true",
                        help='Plays audio fragments as they are matched using SoX audio tool')
    args = parser.parse_args()

    # Debug helpers
    logging.basicConfig(stream=sys.stderr, level=args.loglevel if args.loglevel else 20)
    logging.debug("Start")

    fragments = []
    fragments_cache_path = args.result + '.cache'
    model_dir = os.path.expanduser(args.model if args.model else 'models/en')
    logging.debug("Looking for model files in %s..." % model_dir)
    output_graph_path, alphabet_path, lm_path, trie_path = wavTranscriber.resolve_models(model_dir)
    logging.debug("Loading alphabet from %s..." % alphabet_path)
    alphabet = text.Alphabet(alphabet_path)

    if path.exists(fragments_cache_path):
        logging.debug("Loading cached segment transcripts from %s..." % fragments_cache_path)
        with open(fragments_cache_path, 'r') as result_file:
            fragments = json.loads(result_file.read())
    else:
        logging.debug("Loading model from %s..." % model_dir)
        model, _, _ = wavTranscriber.load_model(output_graph_path, alphabet_path, lm_path, trie_path)

        inference_time = 0.0
        offset = 0

        # Run VAD on the input file
        logging.debug("Transcribing VAD segments...")
        wave_file = args.audio
        aggressiveness = int(args.aggressive) if args.aggressive else 3
        segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(wave_file, aggressiveness)

        for i, segment in enumerate(segments):
            # Run DeepSpeech on the chunk that just completed VAD
            segment_buffer, time_start, time_end = segment
            logging.debug("Transcribing segment %002d (from %f to %f)..." % (i, time_start / 1000.0, time_end / 1000.0))

            audio = np.frombuffer(segment_buffer, dtype=np.int16)
            segment_transcript, segment_inference_time = wavTranscriber.stt(model, audio, sample_rate)
            if segment_transcript is None:
                logging.debug("Segment %002d empty" % i)
                continue
            inference_time += segment_inference_time
            fragments.append({
                'time_start': time_start,
                'time_end':   time_end,
                'transcript': segment_transcript
            })
            offset += len(segment_transcript)

        logging.debug("Writing segment transcripts to cache file %s..." % fragments_cache_path)
        with open(fragments_cache_path, 'w') as result_file:
            result_file.write(json.dumps(fragments))

    logging.debug("Loading original transcript from %s..." % args.transcript)
    with open(args.transcript, 'r') as transcript_file:
        original_transcript = transcript_file.read()
    tc = text.TextCleaner(original_transcript, alphabet)
    ls = text.LevenshteinSearch(tc.clean_text)
    start = 0
    for fragment in fragments:
        fragment_transcript = fragment['transcript']
        match_distance, match_offset, match_len = ls.find_best(fragment_transcript)
        if match_offset >= 0 and match_distance < 0.2 * len(fragment_transcript):
            logging.debug('transcribed: %s' % fragment['transcript'])
            original_start = tc.get_original_offset(match_offset)
            original_end = tc.get_original_offset(match_offset+match_len)
            fragment['offset'] = original_start
            fragment['length'] = original_end-original_start
            logging.debug('   original: %s' % ' '.join(original_transcript[original_start:original_end].split()))
            start = match_offset+match_len
            if args.play:
                subprocess.check_call(['play', args.audio, 'trim', str(fragment['time_start']/1000.0), '='+str(fragment['time_end']/1000.0)])
    with open(args.result, 'w') as result_file:
        result_file.write(json.dumps(fragments))

if __name__ == '__main__':
    main(sys.argv[1:])
