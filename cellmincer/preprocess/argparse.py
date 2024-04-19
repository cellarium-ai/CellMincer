import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    '''Add tool-specific arguments.
    Args:
        subparsers: Parser object before addition of arguments specific to
            `preprocess`.
    Returns:
        parser: Parser object with additional parameters.
    '''

    subparser = subparsers.add_parser(
        'preprocess',
        description='Prepares raw dataset for training or denoising.',
        help='Prepares raw dataset for training or denoising. This tool detrends the data and computes global features, writing all outputs to a dedicated output directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparser.add_argument(
        '-i',
        '--input',
        nargs=None,
        type=str,
        dest='input_file',
        default=None,
        required=True,
        help='Input dataset file to preprocess. Accepted formats are .TIF, .NPY, .NPZ, .BIN.')

    subparser.add_argument(
        '-o',
        '--output',
        nargs=None,
        type=str,
        dest='output_dir',
        default=None,
        required=True,
        help='Directory where outputs are written.')

    subparser.add_argument(
        '--manifest',
        nargs=None,
        type=str,
        dest='manifest',
        default=None,
        required=True,
        help='Input data manifest.')

    subparser.add_argument(
        '--config',
        nargs=None,
        type=str,
        dest='config',
        default=None,
        required=True,
        help='Preprocessing configuration YAML.')

    return subparsers
