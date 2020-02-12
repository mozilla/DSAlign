# -*- coding: utf-8 -*-
import os
import csv
import json

from pathlib import Path
from functools import partial
from utils import MEGABYTE, GIGABYTE, Interleaved
from audio import Sample, DEFAULT_FORMAT, AUDIO_TYPE_WAV, AUDIO_TYPE_OPUS, SERIALIZABLE_AUDIO_TYPES

BIG_ENDIAN = 'big'
INT_SIZE = 4
BIGINT_SIZE = 2 * INT_SIZE
MAGIC = b'SAMPLEDB'
INDEXING_FRACTION = 0.05

BUFFER_SIZE = 1 * MEGABYTE
CACHE_SIZE = 1 * GIGABYTE

SCHEMA_KEY = 'schema'
MIME_TYPE_KEY = 'mime-type'
MIME_TYPE_TEXT = 'text/plain'
CONTENT_TYPE_SPEECH = 'speech'
CONTENT_TYPE_TRANSCRIPT = 'transcript'


class CollectionSample(Sample):
    """In-memory sample collection sample representing an utterance.
    Derived from util.audio.Sample and used by sample collection readers and writers."""
    def __init__(self, sample_id, audio_type, raw_data, transcript, audio_format=DEFAULT_FORMAT):
        """
        Creates an in-memory speech sample together with a transcript of the utterance (label).
        :param sample_id: Tracking ID used for debugging
        :param audio_type: See util.audio.Sample.__init__ .
        :param raw_data: See util.audio.Sample.__init__ .
        :param transcript: Transcript of the sample's utterance
        :param audio_format: See util.audio.Sample.__init__ .
        """
        super().__init__(audio_type, raw_data, audio_format=audio_format)
        self.sample_id = sample_id
        self.transcript = transcript


class DirectSDBWriter:
    """Sample collection writer for creating a Sample DB (SDB) file"""
    def __init__(self, sdb_filename, buffering=BUFFER_SIZE, audio_type=AUDIO_TYPE_OPUS):
        self.sdb_filename = sdb_filename
        if audio_type not in SERIALIZABLE_AUDIO_TYPES:
            raise ValueError('Audio type "{}" not supported'.format(audio_type))
        self.audio_type = audio_type
        self.sdb_file = open(sdb_filename, 'wb', buffering=buffering)
        self.offsets = []
        self.num_samples = 0

        self.sdb_file.write(MAGIC)

        meta_data = {
            SCHEMA_KEY: {
                CONTENT_TYPE_SPEECH: {MIME_TYPE_KEY: audio_type},
                CONTENT_TYPE_TRANSCRIPT: {MIME_TYPE_KEY: MIME_TYPE_TEXT}
            }
        }
        meta_data = json.dumps(meta_data).encode()
        self.write_big_int(len(meta_data))
        self.sdb_file.write(meta_data)

        self.offset_samples = self.sdb_file.tell()
        self.sdb_file.seek(2 * BIGINT_SIZE, 1)

    def write_int(self, n):
        return self.sdb_file.write(n.to_bytes(INT_SIZE, BIG_ENDIAN))

    def write_big_int(self, n):
        return self.sdb_file.write(n.to_bytes(BIGINT_SIZE, BIG_ENDIAN))

    def __enter__(self):
        return self

    def add(self, sample):
        def to_bytes(n):
            return n.to_bytes(INT_SIZE, BIG_ENDIAN)
        sample.change_audio_type(self.audio_type)
        opus = sample.audio.getbuffer()
        opus_len = to_bytes(len(opus))
        transcript = sample.transcript.encode()
        transcript_len = to_bytes(len(transcript))
        entry_len = to_bytes(len(opus_len) + len(opus) + len(transcript_len) + len(transcript))
        buffer = b''.join([entry_len, opus_len, opus, transcript_len, transcript])
        self.offsets.append(self.sdb_file.tell())
        self.sdb_file.write(buffer)
        self.num_samples += 1

    def finalize(self):
        if self.sdb_file is None:
            return
        offset_index = self.sdb_file.tell()
        self.sdb_file.seek(self.offset_samples)
        self.write_big_int(offset_index - self.offset_samples - BIGINT_SIZE)
        self.write_big_int(self.num_samples)

        self.sdb_file.seek(offset_index + BIGINT_SIZE)
        self.write_big_int(self.num_samples)
        for index, offset in enumerate(self.offsets):
            self.write_big_int(offset)
            yield index / len(self.offsets)
        offset_end = self.sdb_file.tell()
        self.sdb_file.seek(offset_index)
        self.write_big_int(offset_end - offset_index - BIGINT_SIZE)
        self.sdb_file.close()
        self.sdb_file = None

    def close(self):
        for _ in self.finalize():
            pass

    def __len__(self):
        return len(self.offsets)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SortingSDBWriter:  # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 sdb_filename,
                 tmp_sdb_filename=None,
                 cache_size=CACHE_SIZE,
                 buffering=BUFFER_SIZE,
                 audio_type=AUDIO_TYPE_OPUS):
        self.sdb_filename = sdb_filename
        self.buffering = buffering
        self.tmp_sdb_filename = (sdb_filename + '.tmp') if tmp_sdb_filename is None else tmp_sdb_filename
        if audio_type not in SERIALIZABLE_AUDIO_TYPES:
            raise ValueError('Audio type "{}" not supported'.format(audio_type))
        self.audio_type = audio_type
        self.tmp_sdb = DirectSDBWriter(self.tmp_sdb_filename, buffering=buffering, audio_type=audio_type)
        self.cache_size = cache_size
        self.buckets = []
        self.bucket = []
        self.bucket_offset = 0
        self.bucket_size = 0
        self.overall_size = 0

    def __enter__(self):
        return self

    def finish_bucket(self):
        if len(self.bucket) == 0:
            return
        self.bucket.sort(key=lambda s: s.duration)
        for sample in self.bucket:
            self.tmp_sdb.add(sample)
        self.buckets.append((self.bucket_offset, self.bucket_offset + len(self.bucket)))
        self.bucket_offset += len(self.bucket)
        self.bucket = []
        self.overall_size += self.bucket_size
        self.bucket_size = 0

    def add(self, sample):
        sample.change_audio_type(self.audio_type)
        self.bucket.append(sample)
        self.bucket_size += len(sample.audio.getbuffer())
        if self.bucket_size > self.cache_size:
            self.finish_bucket()

    def finalize(self):
        if self.tmp_sdb is None:
            return
        self.finish_bucket()
        num_samples = len(self.tmp_sdb)
        for frac in self.tmp_sdb.finalize():
            yield frac * INDEXING_FRACTION
        self.tmp_sdb = None
        avg_sample_size = self.overall_size / num_samples
        max_cached_samples = self.cache_size / avg_sample_size
        buffer_size = max(1, int(max_cached_samples / len(self.buckets)))
        sdb_reader = SDB(self.tmp_sdb_filename, buffering=self.buffering)

        def buffered_view(bucket):
            start, end = bucket
            buffer = []
            current_offset = start
            while current_offset < end:
                while len(buffer) < buffer_size and current_offset < end:
                    buffer.insert(0, sdb_reader[current_offset])
                    current_offset += 1
                while len(buffer) > 0:
                    yield buffer.pop(-1)

        bucket_views = list(map(buffered_view, self.buckets))
        interleaved = Interleaved(*bucket_views, key=lambda s: s.duration)
        with DirectSDBWriter(self.sdb_filename, buffering=self.buffering, audio_type=self.audio_type) as sdb_writer:
            factor = (1.0 - 2.0 * INDEXING_FRACTION) / num_samples
            for index, sample in enumerate(interleaved):
                sdb_writer.add(sample)
                yield INDEXING_FRACTION + index * factor
            for frac in sdb_writer.finalize():
                yield (1.0 - INDEXING_FRACTION) + frac * INDEXING_FRACTION
        os.unlink(self.tmp_sdb_filename)

    def close(self):
        for _ in self.finalize():
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SDB:  # pylint: disable=too-many-instance-attributes
    """Sample collection reader for reading a Sample DB (SDB) file"""
    def __init__(self, sdb_filename, buffering=BUFFER_SIZE):
        self.meta = {}
        self.schema = []
        self.offsets = []
        self.sdb_filename = sdb_filename
        self.sdb_file = open(sdb_filename, 'rb', buffering=buffering)
        if self.sdb_file.read(len(MAGIC)) != MAGIC:
            raise RuntimeError('No Sample Database')
        meta_chunk_len = self.read_big_int()
        self.meta = json.loads(self.sdb_file.read(meta_chunk_len))
        if SCHEMA_KEY not in self.meta:
            raise RuntimeError('Missing schema')
        self.schema = self.meta[SCHEMA_KEY]
        content_types = sorted(self.schema.keys())
        for column_index, content_type in enumerate(content_types):
            entry = self.schema[content_type]
            if not isinstance(entry, dict):
                raise RuntimeError('Malformed schema entry for content-type "{}"'.format(content_type))
            entry['column_index'] = column_index

        if CONTENT_TYPE_SPEECH not in self.schema:
            raise RuntimeError('No speech data (missing in schema)')
        speech_column = self.schema[CONTENT_TYPE_SPEECH]
        self.speech_index = speech_column['column_index']
        self.audio_type = speech_column[MIME_TYPE_KEY]
        if self.audio_type not in SERIALIZABLE_AUDIO_TYPES:
            raise RuntimeError('Unsupported audio format: {}'.format(self.audio_type))

        if CONTENT_TYPE_TRANSCRIPT not in self.schema:
            raise RuntimeError('No transcript data (missing in schema)')
        transcript_column = self.schema[CONTENT_TYPE_TRANSCRIPT]
        self.transcript_index = transcript_column['column_index']
        text_type = transcript_column[MIME_TYPE_KEY]
        if text_type != MIME_TYPE_TEXT:
            raise RuntimeError('Unsupported text type: {}'.format(text_type))

        sample_chunk_len = self.read_big_int()
        self.sdb_file.seek(sample_chunk_len + BIGINT_SIZE, 1)
        num_samples = self.read_big_int()
        for _ in range(num_samples):
            self.offsets.append(self.read_big_int())

    def read_int(self):
        return int.from_bytes(self.sdb_file.read(INT_SIZE), BIG_ENDIAN)

    def read_big_int(self):
        return int.from_bytes(self.sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)

    def read_row(self, row_index, *columns):
        columns = list(columns)
        column_data = [None] * len(columns)
        found = 0
        if not 0 <= row_index < len(self.offsets):
            raise ValueError('Wrong sample index: {} - has to be between 0 and {}'
                             .format(row_index, len(self.offsets) - 1))
        self.sdb_file.seek(self.offsets[row_index] + INT_SIZE)
        for index in range(len(self.schema)):
            chunk_len = self.read_int()
            if index in columns:
                column_data[columns.index(index)] = self.sdb_file.read(chunk_len)
                found += 1
                if found == len(columns):
                    return tuple(column_data)
            else:
                self.sdb_file.seek(chunk_len, 1)
        return tuple(column_data)

    def __getitem__(self, i):
        audio_data, transcript = self.read_row(i, self.speech_index, self.transcript_index)
        transcript = transcript.decode()
        sample_id = self.sdb_filename + ':' + str(i)
        return CollectionSample(sample_id, self.audio_type, audio_data, transcript)

    def __iter__(self):
        for i in range(len(self.offsets)):
            yield self[i]

    def __len__(self):
        return len(self.offsets)

    def close(self):
        if self.sdb_file is not None:
            self.sdb_file.close()

    def __del__(self):
        self.close()


class CSV:
    """Sample collection reader for reading a DeepSpeech CSV file"""
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.rows = []
        csv_dir = Path(csv_filename).parent
        with open(csv_filename, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                wav_filename = Path(row['wav_filename'])
                if not wav_filename.is_absolute():
                    wav_filename = csv_dir / wav_filename
                self.rows.append((str(wav_filename), int(row['wav_filesize']), row['transcript']))
        self.rows.sort(key=lambda r: r[1])

    def __getitem__(self, i):
        wav_filename, _, transcript = self.rows[i]
        with open(wav_filename, 'rb') as wav_file:
            return CollectionSample(wav_filename, AUDIO_TYPE_WAV, wav_file.read(), transcript)

    def __iter__(self):
        for i in range(len(self.rows)):
            yield self[i]

    def __len__(self):
        return len(self.rows)


def samples_from_file(filename, buffering=BUFFER_SIZE):
    """Retrieves the right sample collection reader from a filename"""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.sdb':
        return SDB(filename, buffering=buffering)
    if ext == '.csv':
        return CSV(filename)
    raise ValueError('Unknown file type: "{}"'.format(ext))


def samples_from_files(filenames, buffering=BUFFER_SIZE):
    """Retrieves a (potentially interleaving) sample collection reader from a list of filenames"""
    if len(filenames) == 0:
        raise ValueError('No files')
    if len(filenames) == 1:
        return samples_from_file(filenames[0], buffering=buffering)
    cols = list(map(partial(samples_from_file, buffering=buffering), filenames))
    return Interleaved(*cols, key=lambda s: s.duration)
