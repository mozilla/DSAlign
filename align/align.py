import sys
import os
import logging
import argparse
import numpy as np
import wavTranscriber

# Debug helpers
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def main(args):
    parser = argparse.ArgumentParser(description='Transcribe long audio files using webRTC VAD or use the streaming interface')
    parser.add_argument('audio', type=str,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--aggressive', type=int, choices=range(4), required=False,
                        help='Determines how aggressive filtering out non-speech is. (Interger between 0-3)')
    parser.add_argument('--model', required=False,
                        help='Path to directory that contains all model files (output_graph, lm, trie and alphabet)')
    args = parser.parse_args()

    # Loading model
    model_dir = os.path.expanduser(args.model if args.model else 'models/en')
    output_graph, alphabet, lm, trie = wavTranscriber.resolve_models(model_dir)
    model = wavTranscriber.load_model(output_graph, alphabet, lm, trie)

    title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 'Model Load Time(s)', 'LM Load Time(s)']
    print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))

    inference_time = 0.0

    # Run VAD on the input file
    wave_file = args.audio
    aggressiveness = int(args.aggressive) if args.aggressive else 3
    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(wave_file, aggressiveness)
    f = open(wave_file.rstrip(".wav") + ".txt", 'w')
    logging.debug("Saving Transcript @: %s" % wave_file.rstrip(".wav") + ".txt")

    for i, segment in enumerate(segments):
        # Run deepspeech on the chunk that just completed VAD
        logging.debug("Processing chunk %002d" % (i,))
        audio = np.frombuffer(segment, dtype=np.int16)
        output = wavTranscriber.stt(model[0], audio, sample_rate)
        inference_time += output[1]
        logging.debug("Transcript: %s" % output[0])

        f.write(output[0] + " ")

    # Summary of the files processed
    f.close()

    # Extract filename from the full file path
    filename, ext = os.path.split(os.path.basename(wave_file))
    logging.debug("************************************************************************************************************")
    logging.debug("%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
    logging.debug("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
    logging.debug("************************************************************************************************************")
    print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))


if __name__ == '__main__':
    main(sys.argv[1:])
