import shutil
import sys

import ds_ctcdecoder
from deepspeech_training.util.text import Alphabet, UTF8Alphabet
from ds_ctcdecoder import Scorer, Alphabet as NativeAlphabet


def create_bundle(
    alphabet_path,
    lm_path,
    vocab_path,
    package_path,
    force_utf8,
    default_alpha,
    default_beta,
):
    words = set()
    vocab_looks_char_based = True
    with open(vocab_path) as fin:
        for line in fin:
            for word in line.split():
                words.add(word.encode("utf-8"))
                if len(word) > 1:
                    vocab_looks_char_based = False
    print("{} unique words read from vocabulary file.".format(len(words)))

    cbm = "Looks" if vocab_looks_char_based else "Doesn't look"
    print("{} like a character based model.".format(cbm))

    if force_utf8 != None:  # pylint: disable=singleton-comparison
        use_utf8 = force_utf8
    else:
        use_utf8 = vocab_looks_char_based
        print("Using detected UTF-8 mode: {}".format(use_utf8))

    if use_utf8:
        serialized_alphabet = UTF8Alphabet().serialize()
    else:
        if not alphabet_path:
            raise RuntimeError("No --alphabet path specified, can't continue.")
        serialized_alphabet = Alphabet(alphabet_path).serialize()

    alphabet = NativeAlphabet()
    err = alphabet.deserialize(serialized_alphabet, len(serialized_alphabet))
    if err != 0:
        raise RuntimeError("Error loading alphabet: {}".format(err))

    scorer = Scorer()
    scorer.set_alphabet(alphabet)
    scorer.set_utf8_mode(use_utf8)
    scorer.reset_params(default_alpha, default_beta)
    scorer.load_lm(lm_path)
    # TODO: Why is this not working?
    #err = scorer.load_lm(lm_path)
    #if err != ds_ctcdecoder.DS_ERR_SCORER_NO_TRIE:
    #    print('Error loading language model file: 0x{:X}.'.format(err))
    #    print('See the error codes section in https://deepspeech.readthedocs.io for a description.')
    #    sys.exit(1)
    scorer.fill_dictionary(list(words))
    shutil.copy(lm_path, package_path)
    scorer.save_dictionary(package_path, True)  # append, not overwrite
    print("Package created in {}".format(package_path))