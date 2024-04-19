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
        description='Denoises preprocessed data with trained model.',
        help='Denoises preprocessed data with trained model. By default, both an original-scale and a detrended version of the denoised data are written, along with an .AVI movie file for visualization.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparser.add_argument(
        '-d',
        '--dataset',
        nargs=None,
        type=str,
        dest='dataset',
        default=None,
        required=True,
        help='Preprocessed dataset directory of CellMincer assets.')

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
        dest='model_ckpt',
        default=None,
        required=True,
        help='Path to trained model checkpoint.')
    
    subparser.add_argument(
        '--avi-frame-range',
        nargs=2,
        type=int,
        dest='avi_frame_range',
        default=None,
        required=False,
        help='Range of frames to output as .AVI, declared as start (inclusive) and end (exclusive).')
    
    subparser.add_argument(
        '--avi-zscore-range',
        nargs=2,
        type=int,
        dest='avi_zscore_range',
        default=None,
        required=False,
        help='Pixel intensity clip range of .AVI, as z-scores of intensity distribution.')
    
    subparser.add_argument(
        '--no-avi',
        dest='avi_enabled',
        action='store_false',
        help='Flag that skips writing .AVI.')

    return subparsers
