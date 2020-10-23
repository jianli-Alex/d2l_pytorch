#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
function: realize the tree of catalogue
"""

import os
import fnmatch
import argparse


def file_tree(path):
    """params path: the path of file_tree you want to print"""
    for root, files_dir, files_name in os.walk(path):
        # the depth of dir
        depth = root.count("/")
        print(" " * (depth - 1) + "|-" + root)

        # filter the files in the dir in a list
        for item in fnmatch.filter(files_name, "*"):
            print(" " * depth + "|-" + item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="the path of file_tree you want to print"
    )

    args = parser.parse_args()
    file_tree(args.path)
