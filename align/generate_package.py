import shutil
import struct
from ds_ctcdecoder import Scorer, Alphabet as NativeAlphabet


class Alphabet(object):
    def __init__(self, config_file):
        self._config_file = config_file
        self._label_to_str = {}
        self._str_to_label = {}
        self._size = 0
        if config_file:
            with open(config_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    if line[0:2] == '\\#':
                        line = '#\n'
                    elif line[0] == '#':
                        continue
                    self._label_to_str[self._size] = line[:-1] # remove the line ending
                    self._str_to_label[line[:-1]] = self._size
                    self._size += 1

    def serialize(self):
        # Serialization format is a sequence of (key, value) pairs, where key is
        # a uint16_t and value is a uint16_t length followed by `length` UTF-8
        # encoded bytes with the label.
        res = bytearray()

        # We start by writing the number of pairs in the buffer as uint16_t.
        res += struct.pack('<H', self._size)
        for key, value in self._label_to_str.items():
            value = value.encode('utf-8')
            # struct.pack only takes fixed length strings/buffers, so we have to
            # construct the correct format string with the length of the encoded
            # label.
            res += struct.pack('<HH{}s'.format(len(value)), key, len(value), value)
        return bytes(res)


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
    with open(vocab_path) as fin:
        for line in fin:
            for word in line.split():
                words.add(word.encode("utf-8"))

    if not alphabet_path:
        raise RuntimeError("No --alphabet path specified, can't continue.")
    serialized_alphabet = Alphabet(alphabet_path).serialize()

    alphabet = NativeAlphabet()
    err = alphabet.deserialize(serialized_alphabet, len(serialized_alphabet))
    if err != 0:
        raise RuntimeError("Error loading alphabet: {}".format(err))

    scorer = Scorer()
    scorer.set_alphabet(alphabet)
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