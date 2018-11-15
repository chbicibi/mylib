#! /usr/bin/env python3

'''
2018.8.16 ver 0.1
2018.8.18 ver 0.2
2018.8.23 ver 0.3
2018.8.26 ver 0.4
2018.9.02 ver 0.5
2018.9.13 ver 0.6
2018.9.16 ver 0.7
'''

import ctypes
import glob
import hashlib
import json
import os
import pickle
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from itertools import chain, product


################################################################################
# shell
################################################################################

@contextmanager
def chdir(path):
    prev_path = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_path)


def open_loc(file):
    subprocess.run(['explorer', '/select,', file.replace('/', '\\')])


def popen(*cmds):
    proc = subprocess.Popen(cmds,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    while True:
        line = proc.stdout.readline()
        if line:
            yield line.decode()
        if not line and proc.poll() is not None:
            break


def iglobm(pathname, recursive=True, sep=os.sep):
    keys = list(product(*(s.split(',')
                          for s in re.findall(r'{(.+?)}', pathname))))
    pattern = re.sub(r'(?<={).+?(?=})', '', pathname)
    if keys:
        it = chain(*(glob.iglob(pattern.format(*k), recursive=recursive)
                     for k in keys))
    else:
        it = glob.iglob(pattern, recursive=recursive)
    return (f.replace('/', sep) for f in it)


def globm(pathname, recursive=True):
    return list(iglobm(pathname, recursive=recursive))


def rmempty(path, rm=False):
    if not os.path.isdir(path):
        return True
    L = [x for x in (rmempty(os.path.join(path, f), rm=True)
                     for f in os.listdir(path)) if x]
    if not L and rm:
        print('Removing:', path)
        os.rmdir(path)
    return L


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def realpath(path):
    abspath = os.path.abspath(path)
    if not os.path.exists(abspath):
        raise FileNotFoundError
    if os.path.islink(abspath):
        return realpath(os.readlink(abspath))
    dirname = os.path.dirname(abspath)
    basename = os.path.basename(abspath)
    if abspath == dirname:
        return abspath
    return os.path.join(realpath(dirname), basename)


################################################################################
# stopwatch
################################################################################

def parse_time(a):
    if 0 < a < 1:
        return f'{a:.3g}秒'
    sec = int(a)
    d = f'{sec // 86400}日'          if sec >= 86400 else ''
    h = f'{sec % 86400 // 3600}時間' if sec >= 3600  else ''
    m = f'{sec % 3600 // 60}分'      if sec >= 60    else ''
    s = f'{sec % 60}秒'
    return d + h + m + s


def stopwatch_old(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        isec = int(elapsed_time)
        if isec < 60:
            stime = f'{elapsed_time:.3g}秒'
        else:
            stime = f'{isec}秒 ({parse_time(isec)})'
        print(f'{func.__name__}: {stime}')
        return result
    return wrapper


@contextmanager
def stopwatch(name='anonymous'):
    start = time.time()
    yield lambda: time.time() - start
    elapsed_time = time.time() - start
    isec = int(elapsed_time)
    if isec < 60:
        stime = f'{elapsed_time:.3g}秒'
    else:
        stime = f'{isec}秒 ({parse_time(isec)})'
    print(f'[Stopwatch@{strnow()}] {name}: {stime}')


def strnow(format='%Y/%m/%d %H:%M:%S'):
    return datetime.now().strftime(format)


################################################################################
# path
################################################################################

def split3(path):
    abspath = os.path.abspath
    dirname, basename = os.path.split(abspath)
    return (dirname, *os.path.splitext(basename))


def repair_path(path):
    table = str.maketrans('\\/:*?"<>|', '￥／：＊？”＜＞｜', '')
    return path.translate(table)


def uniq_path(path, ftype='file', key=r'_(\d{1,3})$', suf=lambda n: f'_{n}'):
    if not ftype in ['file', 'dir']:
        raise Exception(f'ftype が不正です: {ftype}')
    if not re.match(key, suf(0)):
        raise Exception('suf が key にマッチしません')
    exist_f = os.path.isfile if ftype == 'file' else os.path.isdir
    split_f = os.path.splitext if ftype == 'file' else lambda path: (path, '')
    if exist_f(path):
        root, ext = split_f(path)
        m = re.search(key, root)
        n = int(m.groups()[0]) if m else 0
        return uniq_path(re.sub(key, '', root) + suf(n + 1) + ext,
                         ftype=ftype, key=key, suf=suf)
    return path


def md5(s):
    return hashlib.md5(s).hexdigest()


def fsort(l):
    key = lambda s: [a or int(b)
                     for a, b in re.compile(r'(\D+)|(\d+)').findall('_' + s)]
    return sorted(l, key=key)


################################################################################
# pickle
################################################################################

def save(file, item, to_json=False):
    if to_json:
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False, indent=2)
    else:
        with open(file, 'wb') as f:
            pickle.dump(item, f)


def load(file, default=None, update=True, from_json=False):
    if not os.path.isfile(file):
        data = default
        if data and update:
            save(file, data, to_json=from_json)
        return data
    if from_json:
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(file, 'rb') as f:
            return pickle.load(f)


################################################################################
# multi thread
################################################################################

def run_mt(task, args_it, max_workers=1, Q=[]):
    def target():
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(task, args_it):
                if Q:
                    break
    try:
        worker = threading.Thread(target=target)
        worker.start()
        while worker.is_alive():
            time.sleep(1)
    except KeyboardInterrupt as e:
        Q.append(True)
        worker.join()
        raise e


################################################################################
# cdll
################################################################################

C_TYPES = {
'void': ctypes.c_void_p,
'int64': ctypes.c_int64,
'double': ctypes.c_double
}

class CDLLHandle(object):
    def __init__(self, libname, loader_path='.'):
        self.libname = libname
        self.loader_path = loader_path
        self.cdll = np.ctypeslib.load_library(libname, loader_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # should be unloaded dll handle
        pass

    def get_fortran_function(self, name, argtypes, restype='void', callback=lambda *x: x):
        try:
            f_ = getattr(self.cdll, name)
            f_.argtypes = [ctypes.POINTER(C_TYPES[t]) for t in argtypes]
            f_.restype = C_TYPES[restype]
            def pyf(*args):
                c_args = [C_TYPES[t](a) for a, t in zip(args, argtypes)]
                c_ptrs = (ctypes.byref(a) for a in c_args)
                res = f_(*c_ptrs)
                return callback(*(a.value for a in c_args), res)
            pyf.f_ptr = f_
            return pyf
        finally:
            pass
