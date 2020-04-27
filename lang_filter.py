#!/usr/bin/env python3
"""
lang_filter.py
$ ./lang_filter.py -l {language} -f {filename_1} -f {filename_2}
"""
# First party packages
import argparse
import os
import sys
import re
import time

# Third party packages
import pandas as pd


def get_inputs():
    """
    Purpose:
        Gets inputs for collect.py
    Args:
        None
    Returns:
        args.lang    (str): Language to filter for
        args.files  (list): List of files to filter on
    """
    parser = argparse.ArgumentParser(description='Filter out tweets by selected language')
    parser.add_argument('-l', '--language', dest='lang', default='english', help='Language to filter for')
    parser.add_argument('-f', '--file', dest='files', action='append', help='List of files to filter on')
    parser.add_argument('-d', '--directory', dest='direc', help='Optional file directory parameter')
    args = parser.parse_args()

    return args.lang, args.files, args.direc



def parse_args(lang, files, direc):
    
    # Error out if no filenames passed
    if files is None:
        sys.exit("ERROR: No input files")

    if direc is None:
        # Default to data subdirectory
        direc = os.path.dirname(os.path.abspath(__file__))
        direc = os.path.join(direc, 'data')
        print(direc)
    elif os.path.exists(direc) is False:
        # Error out if passed path does not exist
        sys.exit("ERROR: Passed directory (-d) does not exist")

    full_filenames = []
    for filename in files:
        # Append directory to filenames
        full_filenames.append(os.path.join(direc, filename))

    return lang, full_filenames

def main():
    lang, files, direc = get_inputs()
    lang, files = parse_args(lang, files, direc)
    print(files)
    
    return None


if __name__ == '__main__':
    main()
