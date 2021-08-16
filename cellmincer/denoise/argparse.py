import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    '''Add tool-specific arguments.
    Args:
        subparsers: Parser object before addition of arguments specific to
            `denoise`.
    Returns:
        parser: Parser object with additional parameters.
    '''

    subparser = subparsers.add_parser(
        'denoise',
        description='Denoises data with trained model.',
        help='Denoises data with trained model.')

    subparser.add_argument(
        '-i',
        '--input-dir',
        nargs=None,
        type=str,
        dest='input_dir',
        default=None,
        required=True,
        help='Input movie object.')

    subparser.add_argument(
        '-o',
        '--output',
        nargs=None,
        type=str,
        dest='output_dir',
        default=None,
        required=False,
        help='Directory where outputs are written.')
    
    subparser.add_argument(
        '--model',
        nargs=None,
        type=str,
        dest='model',
        default=None,
        required=True,
        help='Model state dictionary.')
    
    subparser.add_argument(
        '--config',
        nargs=None,
        type=str,
        dest='config',
        default=None,
        required=True,
        help='Denoising configuration YAML.')
    
    subparser.add_argument(
        '--clean',
        nargs=None,
        type=str,
        dest='clean',
        default=None,
        required=False,
        help='Clean reference movie.')

    return subparsers
