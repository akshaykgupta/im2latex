import numpy
import logging

from fuel.datasets import IndexableDataset
from fuel.schemes import IterationScheme, SequentialExampleScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)
from picklable_itertools import chain
from picklable_itertools.extras import partition_all
from six.moves import cPickle
from collections import OrderedDict

logger = logging.getLogger(__name__)

class DefiniteIterationScheme(IterationScheme):

    requests_examples = False

    def __init__(self, batch_size, size_dict):
        self.batch_size = batch_size
        self.size_dict = size_dict

    def get_request_iterator(self):
        iterator_list = []
        start = 0
        for size, examples in self.size_dict.items():
            iterator_list.append(partition_all(self.batch_size, xrange(start, start + examples)))
            start += examples
        return chain(*iterator_list)


class PaddingWithEOS(Padding):
    """Padds a stream with given end of sequence idx."""
    def __init__(self, data_stream, eos_idx, **kwargs):
        kwargs['data_stream'] = data_stream
        self.eos_idx = eos_idx
        super(PaddingWithEOS, self).__init__(**kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = list(next(self.child_epoch_iterator))
        data_with_masks = []
        for i, (source, source_data) in enumerate(
                zip(self.data_stream.sources, data)):
            if source not in self.mask_sources:
                channeled_data = [numpy.asarray([sample]) for sample in source_data]
                # print [sample.shape for sample in channeled_data]
                data_with_masks.append(numpy.asarray(channeled_data))
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_data]
            lengths = [shape[0] for shape in shapes]
            max_sequence_length = max(lengths)
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_data[0]).dtype

            padded_data = numpy.ones(
                (len(source_data), max_sequence_length) + rest_shape,
                dtype=dtype) * self.eos_idx
            for i, sample in enumerate(source_data):
                padded_data[i, :len(sample)] = sample
            data_with_masks.append(padded_data)

            mask = numpy.zeros((len(source_data), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            data_with_masks.append(mask)
        return tuple(data_with_masks)

def get_tr_stream(trg_vocab, training_file, sizes, trg_vocab_size = 502, unk_id = 1, batch_size = 80, **kwargs):
    trg_vocab = trg_vocab if isinstance(trg_vocab, dict) else cPickle.load(open(trg_vocab))
    logger.info('Unpickling data...')
    training_data = cPickle.load(open(training_file))
    images = [example[0] for example in training_data]
    targets = [example[1] for example in training_data]
    sizes = sizes if isinstance(sizes, dict) else cPickle.load(open(sizes))
    dataset = IndexableDataset(OrderedDict([('image', images), ('target', targets)]))
    logger.info('Unpickled data, building data stream...')
    stream = DataStream(dataset, iteration_scheme = DefiniteIterationScheme(batch_size, sizes))
    masked_stream = PaddingWithEOS(data_stream = stream, eos_idx = trg_vocab_size-1, mask_sources = ('target'))
    return masked_stream

def get_dev_stream(valid_file, **kwargs):
    valid_data = cPickle.load(open(valid_file))
    images = [example[0] for example in valid_data]
    targets = [example[1] for example in valid_data]
    dataset = IndexableDataset(OrderedDict([('input', images), ('output', targets)]))
    return DataStream(dataset, iteration_scheme = SequentialExampleScheme(len(images)))
