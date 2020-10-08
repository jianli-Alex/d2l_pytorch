#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""realize some decorate"""

import time


def cal_time(func):
    """calculate execute time"""

    def now_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("execute time is %.3f seconds" % (end - start))

    return now_time
