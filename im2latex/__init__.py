import logging

from collections import Counter
from theano import tensor, scan
import theano
from toolz import merge

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import (ComputationGraph, apply_noise, apply_dropout,
                          apply_batch_normalization, get_batch_normalization_updates)
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector

from testing.checkpoint import CheckpointNMT, LoadNMT
from testing.model import BidirectionalEncoder, Decoder
from testing.sampling import BleuValidator, Sampler
from testing.cnn_model import CNNEncoder
from testing.datastream import get_tr_stream, get_dev_stream

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)


def main(config, use_bokeh=False):

    tr_stream = get_tr_stream(**config)
    # dev_stream = get_dev_stream(**config)
    # Create Theano variables
    logger.info('Creating theano variables')
    source_image = tensor.ftensor4('image')
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')
    sampling_input = tensor.ftensor4('input')
    sampling_output = tensor.lmatrix('output')

    # Construct model
    logger.info('Building RNN encoder-decoder')
    cnn_encoder = CNNEncoder(config['batch_norm'])
    image_embedding = cnn_encoder.conv_sequence.apply(source_image)
    if config['use_rnn']:
        encoder = BidirectionalEncoder(
            config['enc_embed'], config['enc_nhids'])
        encoder_inputs = image_embedding.dimshuffle(2, 3, 0, 1)
        encoded_images, _ = theano.map(encoder.apply, 
            sequences = encoder_inputs, 
            name = 'parallel_encoders')
    else:
        encoded_images = image_embedding.dimshuffle(2, 3, 0, 1)
    encoded_shape = encoded_images.shape
    annotation_vector = encoded_images.reshape((-1, encoded_shape[2], encoded_shape[3]))
    annotation_vector_mask = tensor.ones(annotation_vector.shape[:2])
    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2)

    cost = decoder.cost(annotation_vector, annotation_vector_mask, target_sentence, target_sentence_mask)

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # Initialize model
    logger.info('Initializing model')
    cnn_encoder.conv_sequence.weights_init = IsotropicGaussian(config['weight_scale'])
    cnn_encoder.conv_sequence.biases_init = Constant(0)
    if config['use_rnn']:
        encoder.weights_init = IsotropicGaussian(config['weight_scale'])
        encoder.biases_init = Constant(0)
        encoder.push_initialization_config()
        encoder.bidir.prototype.weights_init = Orthogonal()
        encoder.initialize()
    decoder.weights_init = IsotropicGaussian(config['weight_scale'])
    decoder.biases_init = Constant(0)
    decoder.push_initialization_config()
    decoder.transition.weights_init = Orthogonal()
    decoder.initialize()
    cnn_encoder.conv_sequence.push_initialization_config()
    cnn_encoder.conv_sequence.initialize()

    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # Apply weight noise for regularization
    if config['weight_noise_ff'] > 0.0:
        logger.info('Applying weight noise to ff layers')
        cnn_params = Selector(cnn_encoder.conv_sequence).get_parameters().values()
        enc_params = []
        if config['use_rnn']:
            enc_params += Selector(encoder.fwd_fork).get_parameters().values()
            enc_params += Selector(encoder.back_fork).get_parameters().values()
        dec_params = Selector(
            decoder.sequence_generator.readout).get_parameters().values()
        dec_params += Selector(
            decoder.sequence_generator.fork).get_parameters().values()
        dec_params += Selector(decoder.transition.initial_transformer).get_parameters().values()
        cg = apply_noise(cg, cnn_params+enc_params+dec_params, config['weight_noise_ff'])

    # Apply batch normalization
    if config['batch_norm']:
        logger.info('Applying batch normalization')
        cg = apply_batch_normalization(cg)
        pop_updates = get_batch_normalization_updates(cg)
        extra_updates = [(p, m * 0.05 + p * (1 - 0.05))
                         for p, m in pop_updates]
    else:
        extra_updates = []

    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
    logger.info("Total number of parameters: {}".format(len(shapes)))

    # Print parameter names
    if config['use_rnn']:
        enc_dec_param_dict = merge(Selector(cnn_encoder.conv_sequence).get_parameters(),
                                   Selector(encoder).get_parameters(),
                                   Selector(decoder).get_parameters())
    else:
        enc_dec_param_dict = merge(Selector(cnn_encoder.conv_sequence).get_parameters(),
                                   Selector(decoder).get_parameters())
    logger.info("Parameter names: ")
    for name, value in enc_dec_param_dict.items():
        logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    logger.info("Total number of parameters: {}"
                .format(len(enc_dec_param_dict)))

    # Set up training model
    logger.info("Building model")
    training_model = Model(cost)

    # Set extensions
    logger.info("Initializing extensions")
    extensions = [
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
        CheckpointNMT(config['saveto'],
                      every_n_batches=config['save_freq'])
    ]

    # Set up beam search and sampling computation graphs if necessary
    if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
        logger.info("Building sampling model")
        sampling_image_embedding = cnn_encoder.conv_sequence.apply(sampling_input)
        if config['use_rnn']:
            sampling_encoder_inputs = sampling_image_embedding.dimshuffle(2, 3, 0, 1)
            sampling_encoded_images, _ = theano.map(encoder.apply,
                sequences = sampling_encoder_inputs,
                name = 'parallel_encoders_inf')
        else:
            sampling_encoded_images = sampling_image_embedding.dimshuffle(2, 3, 0, 1)
        sampling_encoded_shape = sampling_encoded_images.shape
        sampling_annotation_vector = sampling_encoded_images.reshape((-1, 
                sampling_encoded_shape[2], sampling_encoded_shape[3]))
        sampling_annotation_vector_mask = tensor.ones(sampling_annotation_vector.shape[:2])
        generated = decoder.generate(sampling_annotation_vector)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs

    # Add sampling
    if config['hook_samples'] >= 1:
        logger.info("Building sampler")
        extensions.append(
            Sampler(model=search_model, data_stream=tr_stream,
                    hook_samples=config['hook_samples'],
                    every_n_batches=config['sampling_freq'],
                    trg_vocab=config['trg_vocab']))
    # Add early stopping based on bleu
    if 'bleu_script' in config:
        logger.info("Building bleu validator")
        extensions.append(
            BleuValidator(sampling_input, samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          normalize=config['normalized_bleu'],
                          every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot('Cs-En', channels=[['decoder_cost_cost']],
                 after_batch=True))

    # Set up training algorithm
    logger.info("Initializing training algorithm")
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                 eval(config['step_rule'])()])
    )
    # algorithm.add_updates(extra_updates)

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()
