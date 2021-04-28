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
        description='Trains denoising model.',
        help='Trains denoising model.')
    
    subparser.add_argument(
        '-i',
        '--input',
        nargs='+',
        type=str,
        dest='inputs',
        default=None,
        required=True,
        help='Model training input directory.')
    
    subparser.add_argument(
        '-o',
        '--output',
        nargs=None,
        type=str,
        dest='output_dir',
        default=None,
        required=True,
        help='Directory to save model/training state.')
    
    subparser.add_argument(
        '--config',
        nargs=None,
        type=str,
        dest='config',
        default=None,
        required=True,
        help='Model training configuration.')
    
    subparser.add_argument(
        '--checkpoint',
        nargs=None,
        type=str,
        dest='checkpoint',
        default=None,
        required=False,
        help='Checkpoint file for resuming training.')

    return subparsers
