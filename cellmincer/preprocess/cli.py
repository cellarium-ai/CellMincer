'''Command-line tool functionality for `cellmincer preprocess`.'''

import yaml
import logging
import os
import sys
from datetime import datetime

from cellmincer.cli.base_cli import AbstractCLI
from cellmincer.preprocess.main import Preprocess


class CLI(AbstractCLI):
    '''CLI implements AbstractCLI from the cellmincer.cli package.'''

    def __init__(self):
        self.name = 'preprocess'
        self.args = None

    def get_name(self) -> str:
        return self.name

    def validate_args(self, args):
        '''Validate parsed arguments.'''

        # Ensure that if there's a tilde for $HOME in the file path, it works.
        try:
            args.input_file = os.path.expanduser(args.input_file)
            args.output_dir = os.path.expanduser(args.output_dir)
            args.manifest = os.path.expanduser(args.manifest)
            args.config = os.path.expanduser(args.config)
        except TypeError:
            raise ValueError('Problem with provided input paths.')

        self.args = args

        return args

    def run(self, args):
        '''Run the main tool functionality on parsed arguments.'''
        try:
            with open(args.manifest, 'r') as f:
                manifest = yaml.safe_load(f)
        except yaml.YAMLError:
            raise RuntimeError(f'Error loading the manifest YAML file {args.manifest}!')

        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError:
            raise RuntimeError(f'Error loading the config YAML file {args.config}!')
        
        # Send logging messages to stdout as well as a log file.
        if 'log_dir' in config:
            log_file = os.path.join(config['log_dir'], 'cellmincer_preprocess.log')
            logging.basicConfig(
                level=logging.INFO,
                format='cellmincer:preprocess:%(asctime)s: %(message)s',
                filename=log_file,
                filemode='w')
        else:
            logging.basicConfig(level=logging.INFO)
        
        console = logging.StreamHandler()
        formatter = logging.Formatter('cellmincer:preprocess:%(asctime)s: %(message)s', '%H:%M:%S')
        console.setFormatter(formatter)  # Use the same format for stdout.
        logging.getLogger('').addHandler(console)  # Log to stdout and a file.

        # Log the command as typed by user.
        logging.info('Command:\n' + ' '.join(['cellmincer', 'preprocess'] + sys.argv[2:]))
                                      
        # Preprocess
        Preprocess(
            input_file=args.input_file,
            output_dir=args.output_dir,
            manifest=manifest,
            config=config).run()
