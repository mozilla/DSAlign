import sys
import os
import text
import logging
import argparse
import numpy as np
import wavTranscriber


def main(args):
    parser = argparse.ArgumentParser(description='Force align speech data with a transcript.')
    parser.add_argument('audio', type=str,
                        help='Path to speech audio (WAV format)')
    parser.add_argument('transcript', type=str,
                        help='Path to original transcript')
    parser.add_argument('--aggressive', type=int, choices=range(4), required=False,
                        help='Determines how aggressive filtering out non-speech is. (Interger between 0-3)')
    parser.add_argument('--model', required=False,
                        help='Path to directory that contains all model files (output_graph, lm, trie and alphabet)')
    parser.add_argument('--loglevel', type=int, required=False,
                        help='Log level (between 0 and 50) - default: 20')
    args = parser.parse_args()

    # Debug helpers
    logging.basicConfig(stream=sys.stderr, level=args.loglevel if args.loglevel else 20)

    # Loading model
    model_dir = os.path.expanduser(args.model if args.model else 'models/en')
    output_graph, alphabet, lm, trie = wavTranscriber.resolve_models(model_dir)
    model, _, _ = wavTranscriber.load_model(output_graph, alphabet, lm, trie)
    alphabet = text.Alphabet(alphabet)

    inference_time = 0.0

    # Run VAD on the input file
    wave_file = args.audio
    aggressiveness = int(args.aggressive) if args.aggressive else 3
    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(wave_file, aggressiveness)

    with open(args.transcript, 'r') as transcript_file:
        original_transcript = transcript_file.read()
    original_transcript = ' '.join(original_transcript.lower().split())
    original_transcript = alphabet.filter(original_transcript)

    position = 0

    for i, segment in enumerate(segments):
        # Run DeepSpeech on the chunk that just completed VAD
        logging.debug("Transcribing segment %002d..." % i)
        audio = np.frombuffer(segment, dtype=np.int16)
        segment_transcript, segment_inference_time = wavTranscriber.stt(model, audio, sample_rate)
        inference_time += segment_inference_time

        logging.debug("Looking for segment transcript in original transcript...")
        distance, found_offset, found_len = \
            text.minimal_distance(original_transcript,
                                  segment_transcript,
                                  start=position,
                                  threshold=0.1)
        logging.info("Segment transcript: %s" % segment_transcript)
        logging.info("Segment      found: %s" % original_transcript[found_offset:found_offset+found_len])
        logging.info("--")


if __name__ == '__main__':
    main(sys.argv[1:])
