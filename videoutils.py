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
import traceback

import numpy as np
import cv2
import vapoursynth as vs

import myutils as ut
import videotools as vt
# import tile

PLUGINS_DIR = 'C:/Program Files (x86)/VapourSynth/plugins64'
PLUGINS = (('avsr', f'{PLUGINS_DIR}/vsavsreader.dll'),
           ('lsmas', f'{PLUGINS_DIR}/vslsmashsource.dll'))

STORE_ = {}


################################################################################

class Video(object):

    def __init__(self, path, *args, **kwargs):
        abspath = os.path.abspath(path)

        if not os.path.isfile(abspath):
            raise FileNotFoundError(abspath)

        self.path = abspath
        self.cap_ = None
        self.ar_ = None

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
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise Exception('VideoCapture cannot be opened')
        self.cap_ = cap
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cap:
            self.cap.release()
            self.cap_ = None
        if exc_type:
            print('VSVideo exit:', exc_type)

    def set(self, sec):
        if self.cap is None:
            raise Exception('VideoCapture is closed')
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * sec)
        return self

    def get(self, sec):
        if self.cap is None:
            raise Exception('VideoCapture is closed')
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * sec)
        return next(self)

    @property
    def cap(self):
        if self.cap_ is None:
            raise Exception('VideoCapture is closed')
        return self.cap_

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
            if not self.ar_:
                ar = vt.get_ar(self.path)
                if ar:
                    self.ar_ = ar[0] / ar[1]
                else:
                    self.ar_ = self.width / self.height
            return self.ar_

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
            return len(self) / self.fps if self.fps > 1 else 0

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


################################################################################

class VSVideo(Video):

    core = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        size = kwargs.get('size', 'half')
        if not (size in ('half', 'full') or type(size) in (int, tuple)):
            raise TypeError('Invalid type of "size"', size)
        self.size = size
        self.current_idx = -1
        self.gray_mode = kwargs.get('gray_mode', False)

    def __len__(self):
        return self.video.num_frames

    def __getitem__(self, key):
        n = len(self) + key if key < 0 else key
        if not 0 <= n < len(self):
            raise IndexError
        self.current_idx = key
        return self.get_image(self.get_frame(n))

    def __iter__(self):
        for i, frame in enumerate(self.video.frames()):
            if i <= self.current_idx:
                continue
            self.current_idx = i
            yield self.get_image(frame)

    def __next__(self):
        self.current_idx += 1
        if self.current_idx >= len(self):
            raise StopIteration
        return self[self.current_idx]

    def __enter__(self):
        # print(os.path.isfile(self.path))
        try:
            video = self.get_videonode(self.path)
            self.cap_ = video

            # エラー回避
            STORE_[None] = video
            return self

        except:
            traceback.print_exc()
            print('???')
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        # global STORE_
        if self.cap_ is not None:
            # STORE_ = self.cap_
            self.cap_ = None
        if exc_type:
            print('VSVideo exit:', exc_type)

    def get_image(self, frame):
        if type(self.size) is str:
            size = self.size
        else:
            size = self.W, self.H
        img = get_vsimage(frame, size=size, fmt=self.fmt,
                          gray_mode=self.gray_mode)
        if img is None:
            raise Exception('Unknown error in get_image')
        return img

    def set(self, sec):
        self.current_idx = int(self.fps * sec) - 1
        return self

    def get(self, sec):
        return self[int(self.fps * sec)]

    @property
    def height(self):
        return self.video.height

    @property
    def width(self):
        return self.video.width

    @property
    def H(self):
        if type(self.size) is int:
            return self.size
        elif type(self.size) is tuple:
            return self.size[1]
        elif self.size == 'full' or 'YUV422' in self.fmt:
            return self.video.height
        elif self.size == 'half':
            return self.video.height // 2

    @property
    def W(self):
        if type(self.size) is int:
            for n in (10, 8, 4, 2):
                if self.size % n == 0:
                    return n * round(self.size * self.ar / n)
            raise ValueError('Invalid size:', self.size)
        elif type(self.size) is tuple:
            return self.size[0]
        elif self.size == 'full' or 'YUV422' in self.fmt:
            return self.video.width
        elif self.size == 'half':
            return self.video.width // 2

    @property
    def fps(self):
        return self.video.fps

    @property
    def pos(self):
        return self.current_idx

    @property
    def time(self):
        return self.pos / self.fps

    @property
    def ms(self):
        int(self.time * 1000 / 1000)

    def get_frame(self, n):
        return self.video.get_frame(n)

    @property
    def video(self):
        if self.cap_ is None:
            raise Exception('VideoNode is None')
        return self.cap_

    @property
    def fmt(self):
        return self.video.format.name

    @classmethod
    def get_core(cls):
        if cls.core is not None:
            return cls.core
        print('get_core')
        core = vs.get_core()
        for name, dllpath in PLUGINS:
            if not hasattr(core, name):
                # print(dir(core), name, hasattr(core, name))
                core.std.LoadPlugin(dllpath)
        cls.core = core
        return core


    @classmethod
    def get_videonode(cls, file):
        if not os.path.isfile(file):
            raise FileNotFoundError(file)

        ext = ut.extname(file)
        core = cls.get_core()

        if ext == '.mp4':
            return core.lsmas.LibavSMASHSource(file)

        elif ext == '.avs':
            return core.avsr.Import(file)

        else:
            raise Exception('Invalid ext:', ext)


def get_vsimage(frame, size='half', fmt='YUV420P8', gray_mode=False):
    '''
    size: 'half' or 'full' or (W, H)
    '''
    if gray_mode:
        return np.asarray(frame.get_read_array(0))

    if not fmt in ('YUV420P8', 'YUV420P10', 'YUV422P8', 'YUV422P10'):
        raise ValueError('Unknown format:', fmt)

    y, u, v = map(np.asarray, map(frame.get_read_array, range(3)))

    if fmt.endswith('P10'):
        y, u, v = (np.right_shift(x, 2).astype('u1') for x in (y, u, v))

    if fmt.startswith('YUV422'):
        u, v = (x.repeat(2, axis=1) for x in (u, v))

    if size == 'half':
        # y = y[::2, ::2]
        y = cv2.resize(y, u.shape[::-1], interpolation=cv2.INTER_AREA)

    elif size == 'full':
        u, v = (cv2.resize(x, y.shape[::-1]) for x in (u, v))

    elif type(size) is tuple:
        ips = (cv2.INTER_AREA if x.shape[0] > size[1] else cv2.INTER_LINEAR
               for x in (y, u, v))
        y, u, v = (cv2.resize(x, size, interpolation=ip)
                   for ip, x in zip(ips, (y, u, v)))

    else:
        raise TypeError('Invalid type of "size"', size)

    img = np.stack([y, u, v], axis=2)
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    return img


################################################################################

def openvideo(path, *args, **kwargs):
    if ut.extname(path) in ('.mp4', '.avs'):
        return VSVideo(path, *args, **kwargs)
    else:
        return Video(path, *args, **kwargs)


videohandle = openvideo


################################################################################

def resize(img, size):
    if type(size) in (int, float):
        size = tuple(int(size*s) for s in img.shape[1::-1])

    if img.shape[0] == size[1] and img.shape[1] == size[0]:
        return img

    if size[1] > img.shape[0]:
        # 拡大
        interpolation = cv2.INTER_LINEAR
    else:
        # 縮小
        interpolation = cv2.INTER_AREA
    return cv2.resize(img, size, interpolation=interpolation)


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def imshow(img, wait=0, title='image', scale=None):
    if scale:
        if type(scale) in (int, float):
            img = resize(img, tuple(int(scale*s) for s in img.shape[1::-1]))
        elif type(scale) is tuple:
            img = resize(img, scale)
    cv2.imshow(title, np.asarray(img))
    cv2.waitKey(wait)


################################################################################

def yuv2bgr(y, u, v, t='yuv'):
    Y, Cb, Cr = (np.asarray(x).astype('f4') for x in (y, u, v))
    Cb, Cr = Cb - 128, Cr - 128
    if t == 'yuv':
        U, V = 0.872 * Cb, 1.23 * Cr
        R = Y + 1.13983 * V
        G = Y - 0.39465 * U - 0.58060 * V
        B = Y + 2.03211 * U
    elif t == 'bt601': # 'yuv'と同一
        R = Y + 1.402 * Cr
        G = Y - 0.344136 * Cb - 0.714136 * Cr
        B = Y + 1.772 * Cb
    elif t == 'bt709':
        R = Y + 1.5748 * Cr
        G = Y - 0.187324 * Cb - 0.468124 * Cr
        B = Y + 1.8556 * Cb
    r, g, b = (x.clip(0, 255).astype('u1') for x in (R, G, B))
    return b, g, r


def grayscale(image):
    assert image.ndim == 3
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
