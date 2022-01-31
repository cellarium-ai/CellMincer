'''Command-line tool functionality for `cellmincer denoise`.'''

import yaml
import logging
import os
import sys
from datetime import datetime

from cellmincer.cli.base_cli import AbstractCLI
from cellmincer.denoise.main import Denoise


class CLI(AbstractCLI):
    '''CLI implements AbstractCLI from the cellmincer.cli package.'''

    def __init__(self):
        self.name = 'denoise'
        self.args = None

    def get_name(self) -> str:
        return self.name

    def validate_args(self, args):
        '''Validate parsed arguments.'''

        # Ensure that if there's a tilde for $HOME in the file path, it works.
        try:
            args.input_dir = os.path.expanduser(args.input_dir)
            if args.output_dir:
                args.output_dir = os.path.expanduser(args.output_dir)
            else:
                args.output_dir = args.input_dir
            args.model = os.path.expanduser(args.model)
            args.config = os.path.expanduser(args.config)
        except TypeError:
            raise ValueError('Problem with provided input paths.')

        self.args = args

        return args

    def run(self, args):
        '''Run the main tool functionality on parsed arguments.'''

        try:
            with open(args.config, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            raise RuntimeError(f'Error loading the input YAML file {args.input_yaml_file}!')
        
        # Send logging messages to stdout as well as a log file.
        if 'log_dir' in config:
            log_file = os.path.join(config['log_dir'], 'cellmincer_denoise.log')
            logging.basicConfig(
                level=logging.INFO,
                format='cellmincer:denoise:%(asctime)s: %(message)s',
                filename=log_file,
                filemode='w')
        else:
            logging.basicConfig(level=logging.INFO)
        
        console = logging.StreamHandler()
        formatter = logging.Formatter('cellmincer:denoise:%(asctime)s: %(message)s', '%H:%M:%S')
        console.setFormatter(formatter)  # Use the same format for stdout.
        logging.getLogger('').addHandler(console)  # Log to stdout

        # Log the command as typed by user.
        logging.info('Command:\n' + ' '.join(['cellmincer', 'denoise'] + sys.argv[2:]))
        
        # denoise data
        Denoise(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_state=args.model,
            config=config,
            avi_enabled=args.avi_enabled,
            avi_frames=args.avi_frames,
            avi_sigma=args.avi_sigma).run()
