#!/usr/bin/env python
"""
Tool for combining and converting paths within catalog files
"""
import sys
import json
import argparse

from glob import glob
from pathlib import Path


def fail(message):
    print(message)
    sys.exit(1)


def build_catalog():
    catalog_paths = []
    for source_glob in CLI_ARGS.sources:
        catalog_paths.extend(glob(source_glob))
    items = []
    for catalog_original_path in catalog_paths:
        catalog_path = Path(catalog_original_path).absolute()
        print('Loading catalog "{}"'.format(str(catalog_original_path)))
        if not catalog_path.is_file():
            fail('Unable to find catalog file "{}"'.format(str(catalog_path)))
        with open(catalog_path, 'r') as catalog_file:
            catalog_items = json.load(catalog_file)
        base_path = catalog_path.parent.absolute()
        for item in catalog_items:
            new_item = {}
            for entry, entry_original_path in item.items():
                entry_path = Path(entry_original_path)
                entry_path = entry_path if entry_path.is_absolute() else (base_path / entry_path).absolute()
                if ((len(CLI_ARGS.check) == 1 and CLI_ARGS.check[0] == 'all')
                        or entry in CLI_ARGS.check) and not entry_path.is_file():
                    note = 'Catalog "{}" - Missing file for "{}" ("{}")'.format(
                        str(catalog_original_path), entry, str(entry_original_path))
                    if CLI_ARGS.on_miss == 'fail':
                        fail(note + ' - aborting')
                    if CLI_ARGS.on_miss == 'ignore':
                        print(note + ' - keeping it as it is')
                        new_item[entry] = str(entry_path)
                    elif CLI_ARGS.on_miss == 'drop':
                        print(note + ' - dropping catalog item')
                        new_item = None
                        break
                    else:
                        print(note + ' - removing entry from item')
                else:
                    new_item[entry] = str(entry_path)
            if CLI_ARGS.output is not None and new_item is not None and len(new_item.keys()) > 0:
                items.append(new_item)
    if CLI_ARGS.output is not None:
        catalog_path = Path(CLI_ARGS.output).absolute()
        print('Writing catalog "{}"'.format(str(CLI_ARGS.output)))
        if CLI_ARGS.make_relative:
            base_path = catalog_path.parent
            for item in items:
                for entry in item.keys():
                    item[entry] = str(Path(item[entry]).relative_to(base_path))
        if CLI_ARGS.order_by is not None:
            items.sort(key=lambda i: i[CLI_ARGS.order_by] if CLI_ARGS.order_by in i else '')
        with open(catalog_path, 'w') as catalog_file:
            json.dump(items, catalog_file, indent=2)


def handle_args():
    parser = argparse.ArgumentParser(description='Tool for combining catalog files and/or ordering, checking and '
                                                 'converting paths within catalog files')
    parser.add_argument('--output', help='Write collected catalog items to this new catalog file')
    parser.add_argument('--make-relative', action='store_true',
                        help='Make all path entries of all items relative to new catalog file\'s parent directory')
    parser.add_argument('--check',
                        help='Comma separated list of path entries to check for existence '
                             '("all" for checking every entry, default: no checks)')
    parser.add_argument('--on-miss', default='fail', choices=['fail', 'drop', 'remove', 'ignore'],
                        help='What to do if a path is not existing: '
                             '"fail" (exit program), '
                             '"drop" (drop catalog item) or '
                             '"remove" (remove path entry from catalog item) or '
                             '"ignore" (keep it as it is)')
    parser.add_argument('--order-by', help='Path entry used for sorting items in target catalog')
    parser.add_argument('sources', nargs='+', help='Source catalog files (supporting wildcards)')
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    CLI_ARGS.check = [] if CLI_ARGS.check is None else CLI_ARGS.check.split(',')
    build_catalog()
