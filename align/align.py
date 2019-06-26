import sys
import os
import logging
import argparse
import numpy as np
import wavTranscriber


def main(args):
    parser = argparse.ArgumentParser(description='Transcribe long audio files using webRTC VAD or use the streaming interface')
    parser.add_argument('audio', type=str,
                        help='Path to the audio file to run (WAV format)')
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

    inference_time = 0.0

    # Run VAD on the input file
    wave_file = args.audio
    aggressiveness = int(args.aggressive) if args.aggressive else 3
    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(wave_file, aggressiveness)
    f = open(wave_file.rstrip(".wav") + ".txt", 'w')
    logging.debug("Saving Transcript @: %s" % wave_file.rstrip(".wav") + ".txt")

    for i, segment in enumerate(segments):
        # Run DeepSpeech on the chunk that just completed VAD
        logging.debug("Processing chunk %002d" % (i,))
        audio = np.frombuffer(segment, dtype=np.int16)
        transcript, segment_inference_time = wavTranscriber.stt(model, audio, sample_rate)
        inference_time += segment_inference_time
        logging.info("Transcript: %s" % transcript)

        f.write(transcript + " ")

    # Summary of the files processed
    f.close()

if __name__ == '__main__':
    main(sys.argv[1:])
