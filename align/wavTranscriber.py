import glob
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

    ds = Model(models, BEAM_WIDTH)
    ds.enableDecoderWithLM(lm, trie, LM_ALPHA, LM_BETA)
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
    output = ds.stt(audio)
    return output


def resolve_models(dir_name):
    """
    Resolve directory path for the models and fetch each of them.
    :param dir_name: Path to the directory containing pre-trained models
    :return: tuple containing each of the model files (pb, alphabet, lm and trie)
    """
    pb = glob.glob(dir_name + "/*.pb")[0]
    lm = glob.glob(dir_name + "/lm.binary")[0]
    trie = glob.glob(dir_name + "/trie")[0]
    return pb, lm, trie
