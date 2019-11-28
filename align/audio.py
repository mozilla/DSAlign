import os
import sys
import sox
import time
import wave
import tempfile

DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WIDTH = 2
DEFAULT_FORMAT = (DEFAULT_RATE, DEFAULT_CHANNELS, DEFAULT_WIDTH)


def get_audio_format(wav_file):
    return wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth()


def set_audio_format(wav_file, audio_format=DEFAULT_FORMAT):
    rate, channels, width = audio_format
    wav_file.setframerate(rate)
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(width)


def convert_audio(src_audio_path, dst_audio_path, file_type=None, audio_format=DEFAULT_FORMAT):
    sample_rate, channels, width = audio_format
    transformer = sox.Transformer()
    transformer.set_output_format(file_type=file_type, rate=sample_rate, channels=channels, bits=width*8)
    transformer.build(src_audio_path, dst_audio_path)
    wait_counter = 0
    while not os.path.exists(dst_audio_path) or not os.path.getsize(dst_audio_path) > 0:
        print('Waiting for file "{}" getting converted to "{}"...'.format(src_audio_path, dst_audio_path))
        wait_counter += 1
        if wait_counter > 10:
            print('Problem converting "{}" to "{}"! Exiting...'.format(src_audio_path, dst_audio_path))
            sys.exit(100)
        time.sleep(1)


def ensure_wav_with_format(src_audio_path, audio_format=DEFAULT_FORMAT):
    if src_audio_path.endswith('.wav'):
        with wave.open(src_audio_path, 'r') as src_audio_file:
            if get_audio_format(src_audio_file) == audio_format:
                return src_audio_path, False
    _, tmp_file_path = tempfile.mkstemp(suffix='.wav')
    convert_audio(src_audio_path, tmp_file_path, file_type='wav', audio_format=audio_format)
    return tmp_file_path, True


def extract_audio(audio_file, start, end):
    assert 0 <= start <= end
    rate = audio_file.getframerate()
    audio_file.setpos(int(start * rate))
    return audio_file.readframes(int((end - start) * rate))
