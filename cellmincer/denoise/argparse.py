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
        '--avi_frames',
        nargs=2,
        type=int,
        dest='avi_frames',
        default=None,
        required=False,
        help='Range of frames to output as .AVI.')
    
    subparser.add_argument(
        '--avi_sigma',
        nargs=2,
        type=int,
        dest='avi_sigma',
        default=None,
        required=False,
        help='Pixel intensity clip range of .AVI, as stds of intensity distribution.')
    
    subparser.add_argument(
        '--no-avi',
        dest='avi_enabled',
        action='store_false',
        help='Skip writing .AVI.')

    return subparsers
