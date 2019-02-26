# import argparse
# import codecs
# import glob
import os
# import re
# import shutil
# import subprocess
# import sys
# from collections import namedtuple
# from contextlib import contextmanager
# from datetime import datetime, timedelta
# from functools import reduce
# from itertools import chain
# from time import sleep

import cv2

# import myutils as ut
# import tile


class Video(object):

    def __init__(self, path):
        abspath = os.path.abspath(path)

        if not os.path.isfile(abspath):
            raise FileNotFoundError(abspath)

        self.path = abspath
        self.cap = None

    def __len__(self):
        if self.cap is None:
            return 0

        else:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, key):
        if self.cap is None:
            raise Exception('VideoCapture is closed')

        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, key)
            ret, frame = self.cap.read()
            if not ret:
                raise IndexError
            return frame

    def __iter__(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise IndexError
            yield frame

    def __next__(self):
        if self.cap is None:
            raise Exception('VideoCapture is closed')

        else:
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration
            return frame

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cap:
            self.cap.release()
            self.cap = None

    def set(self, pos):
        if self.cap is None:
            raise Exception('VideoCapture is closed')
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * pos)
        return self

    def get(self, pos):
        if self.cap is None:
            raise Exception('VideoCapture is closed')
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * pos)
        return next(self)

    @property
    def height(self):
        if self.cap is None:
            return 0

        else:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self):
        if self.cap is None:
            return 0

        else:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def ar(self):
        if self.cap is None:
            return 0

        else:
            return self.width / self.height

    @property
    def fps(self):
        if self.cap is None:
            return 0

        else:
            return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def length(self):
        if self.cap is None:
            return 0

        else:
            return len(self) / self.fps

    @property
    def pos(self):
        if self.cap is None:
            return 0

        else:
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def time(self):
        if self.cap is None:
            return 0

        else:
            return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    @property
    def sec(self):
        if self.cap is None:
            return 0

        else:
            return round(self.time)

    @property
    def ms(self):
        if self.cap is None:
            return 0

        else:
            return int(self.cap.get(cv2.CAP_PROP_POS_MSEC)) % 1000
