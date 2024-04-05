import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    '''Add tool-specific arguments.
    Args:
        subparsers: Parser object before addition of arguments specific to
            `train`.
    Returns:
        parser: Parser object with additional parameters.
    '''

    subparser = subparsers.add_parser(
        'train',
        description='Trains Cellmincer denoising model.',
        help='Trains Cellmincer denoising model. Checkpoints are regularly written to the output directory as `last.ckpt`.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    subparser.add_argument(
        '-d',
        '--datasets',
        nargs='+',
        type=str,
        dest='datasets',
        default=None,
        required=True,
        help='One or more preprocessed dataset directories as training data.')
    
    subparser.add_argument(
        '-o',
        '--output',
        nargs=None,
        type=str,
        dest='output_dir',
        default=None,
        required=True,
        help='Directory to save model checkpoints.')
    
    subparser.add_argument(
        '--config',
        nargs=None,
        type=str,
        dest='config',
        default=None,
        required=True,
        help='Model and training configuration YAML.')
    
    subparser.add_argument(
        '--pretrain',
        nargs=None,
        type=str,
        dest='pretrain',
        default=None,
        required=False,
        help='Pre-trained model weights. Used as model initialization if no valid checkpoints provided.')
    
    subparser.add_argument(
        '--resume',
        nargs=None,
        type=str,
        dest='resume',
        default=None,
        required=False,
        help='Checkpoint for resuming training from an aborted run. Used as model initialization if no valid intermediate checkpoints provided.')
    
    subparser.add_argument(
        '--checkpoint',
        nargs=None,
        type=str,
        dest='checkpoint',
        default=None,
        required=False,
        help='Checkpoint file for resuming training on preemptible machines.')

    subparser.add_argument(
        '--gpus',
        nargs=None,
        type=int,
        dest='gpus',
        default=1,
        required=False,
        help='Number of GPUs to use for training. Must be at least 1.')
    
    subparser.add_argument(
        '--use_memmap',
        dest='use_memmap',
        action='store_true',
        help='Use memory-mapped training data for large operations. If not every dataset (after padding and masking) can simultaneously fit in memory, they will be written to disk as temporary files.')

    return subparsers
