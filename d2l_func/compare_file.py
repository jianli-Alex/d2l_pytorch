#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: compare content between two python file, and generate a html to show
            the difference by difflib
"""

import sys
import difflib
import argparse


def read_file(filename):
    try:
        with open(filename, "r+") as f:
            return f.readlines()
    except IOError:
        print(f"Error! Not file: {filename}")
        sys.exit(1)


def compare_file(filename1, filename2, outfile):
    # read content in file1, file2
    content1 = read_file(filename1)
    content2 = read_file(filename2)

    # use difflib to compare
    diff = difflib.HtmlDiff()
    result = diff.make_file(content1, content2)
    with open(outfile, "w+") as f:
        return f.writelines(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename1",
        type=str,
        help="the file1 name you want to compare(.py file)"
    )

    parser.add_argument(
        "--filename2",
        type=str,
        help="the file2 name you want to compare(.py file)"
    )

    parser.add_argument(
        "--outfile",
        type=str,
        help="the output file name you want to save(html)"
    )

    # parse argument
    args = parser.parse_args()
    compare_file(args.filename1, args.filename2, args.outfile)
