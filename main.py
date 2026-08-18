import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    overrides = args.override
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json
    for kv in overrides: # e_48: sweep overrides on top of the json
        key, _, value = kv.partition('=')
        try:
            args[key] = json.loads(value)
        except json.JSONDecodeError:
            args[key] = value

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                        help='key=value pairs applied after the json '
                             '(value parsed as json, else kept as string)')
    return parser

if __name__ == '__main__':
    main()
