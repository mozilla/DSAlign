import glob
import webrtcvad
import logging
import wavSplit
from deepspeech import Model


def load_model(models, alphabet, lm, trie):
    """
    Load the pre-trained model into the memory
    :param models: Output Graph Protocol Buffer file
    :param alphabet: Alphabet.txt file
    :param lm: Language model file
    :param trie: Trie file
    :return: tuple (DeepSpeech object, Model Load Time, LM Load Time)
    """
    N_FEATURES = 26
    N_CONTEXT = 9
    BEAM_WIDTH = 500
    #LM_ALPHA = 0.75
    #LM_BETA = 1.85

    LM_ALPHA = 1
    LM_BETA = 1.85

    ds = Model(models, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
    ds.enableDecoderWithLM(alphabet, lm, trie, LM_ALPHA, LM_BETA)
    return ds


def stt(ds, audio, fs):
    """
    Run Inference on input audio file
    :param ds: DeepSpeech object
    :param audio: Input audio for running inference on
    :param fs: Sample rate of the input audio file
    :return: tuple (Inference result text, Inference time)
    """
    audio_length = len(audio) * (1 / 16000)
    # Run DeepSpeech
    output = ds.stt(audio, fs)
    return output


def resolve_models(dir_name):
    """
    Resolve directory path for the models and fetch each of them.
    :param dir_name: Path to the directory containing pre-trained models
    :return: tuple containing each of the model files (pb, alphabet, lm and trie)
    """
    pb = glob.glob(dir_name + "/*.pb")[0]
    alphabet = glob.glob(dir_name + "/alphabet.txt")[0]
    lm = glob.glob(dir_name + "/lm.binary")[0]
    trie = glob.glob(dir_name + "/trie")[0]
    return pb, alphabet, lm, trie


def vad_segment_generator(wav_file, aggressiveness):
    """
    Generate VAD segments. Filters out non-voiced audio frames.
    :param wav_file: Input wav file to run VAD on.0
    :param aggressiveness: How aggressive filtering out non-speech is (between 0 and 3)
    :return: Returns tuple of
        segments: a bytearray of multiple smaller audio frames
                  (The longer audio split into multiple smaller one's)
        sample_rate: Sample rate of the input audio file
        audio_length: Duration of the input audio file
    """
    logging.debug("Caught the wav file @: %s" % wav_file)
    audio, sample_rate, audio_length = wavSplit.read_wave(wav_file)
    assert sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = wavSplit.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = wavSplit.vad_collector(sample_rate, 30, 300, 0.5, vad, frames)

    return segments, sample_rate, audio_length
