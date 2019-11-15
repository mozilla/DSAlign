import os
import sys
import json
import argparse
import os.path as path

from tqdm import tqdm


def load_catalog(catalog_path):
    with open(catalog_path, 'r') as catalog_file:
        return json.load(catalog_file)


def save_catalog(entries, catalog_path):
    with open(catalog_path, 'w') as catalog_file:
        json.dump(entries, catalog_file)


def add_default_arguments(command):
    command.add_argument('catalog', type=str, help='Catalog file to generate or extend')


def add_dir(catalog, directory):
    print('Adding directory "{}"...'.format(directory))
    entries = load_catalog(catalog)
    for root, dirs, files in os.walk(directory):
        
    save_catalog(entries, catalog)


def add_catalog(catalog, src_catalog):
    print('Adding catalog "{}"...'.format(catalog))
    entries = load_catalog(catalog)

    save_catalog(entries, catalog)


def main():
    parser = argparse.ArgumentParser(description='Creates and maintains catalog files.')

    sub_commands = parser.add_subparsers(dest='sub_commands', help='Catalog commands')

    add_dir = sub_commands.add_parser('add-dir')
    add_default_arguments(add_dir)
    add_dir.add_argument('directory', type=str, help='Path to a directory to recursively scan for files '
                                                     'to add to the catalog')

    add_catalog = sub_commands.add_parser('add-catalog')
    add_default_arguments(add_catalog)
    add_catalog.add_argument('src-catalog', type=str, help='Path to a catalog whose entries to add')

    args = parser.parse_args()
    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop('sub_commands').replace('-', '_')](**kwargs)


if __name__ == '__main__':
    main()
