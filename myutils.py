import ctypes
import glob
import hashlib
import json
import os
import pickle
import re
import smtplib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from email.message import EmailMessage
from functools import reduce, wraps
from itertools import chain, product


SRC_DIR = os.path.dirname(__file__)


################################################################################
# constants
################################################################################

KB1 = 1024
MB1 = 1048576
GB1 = 1073741824


################################################################################
# shell
################################################################################

@contextmanager
def chdir(path):
    prev_path = os.getcwd()
    if path:
        mkdir(path)
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
    if isinstance(pathname, str):
        keys = list(product(*(s.split(',')
                              for s in re.findall(r'{(.+?)}', pathname))))
        pattern = re.sub(r'(?<={).+?(?=})', '', pathname)
        if keys:
            it = chain(*(glob.iglob(pattern.format(*k), recursive=recursive)
                         for k in keys))
        else:
            it = glob.iglob(pattern, recursive=recursive)
        for f in it:
            yield f.replace('/', sep)
    elif iter(pathname):
        for p in pathname:
            yield from iglobm(p, recursive=True, sep=os.sep)
    else:
        raise TypeError


def globm(pathname, recursive=True, sep=os.sep):
    return list(iglobm(pathname, recursive=recursive, sep=os.sep))


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
    if path:
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


def filesize(path='.', follow_symlinks=False):
    if isinstance(path, os.DirEntry):
        return filesize(path.path, follow_symlinks=follow_symlinks)

    if not follow_symlinks and os.path.islink(path):
        return 0

    if os.path.isfile(path):
        return os.path.getsize(path)

    # if not os.path.isdir(path):
    #     raise FileNotFoundError

    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file(follow_symlinks=follow_symlinks):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=follow_symlinks):
                total += filesize(entry.path, follow_symlinks=follow_symlinks)
    return total


################################################################################
# stopwatch
################################################################################

class Stopwatch(object):
    def __init__(self, name='anonymous'):
        self.name = name
        self.start = time.time()

    def __call__(self):
        return time.time() - self.start

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = self()
        if elapsed < 60:
            return
            stime = f'{elapsed:.3g}秒'
        else:
            stime = f'{elapsed:.0f}秒 ({parse_time(int(elapsed))})'
        print(f'[Stopwatch@{strnow()}] {self.name}: {stime}')
        if exc_type is not None:
            print('Exception:', exc_type, exc_value)
            print('---')

    def __float__(self):
        return self()

    def __int__(self):
        return int(self())

    def __str__(self):
        return time2str(self())

    def __lt__(self, other):
        return float(self) < other

    def __gt__(self, other):
        return float(self) > other


def time2str(sec):
    sign = '-' if sec < 0 else ''
    abssec = abs(int(sec))
    h = abssec // 3600
    m = abssec % 3600 // 60
    s = abssec % 60
    def f_(res, x):
        if res:
            return  f'{res}:{x:02d}'
        elif x:
            return f'{x:02d}'
        else:
            return ''
    return sign + reduce(f_, [h, m, s], '')


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


def stopwatch(name='anonymous'):
    return Stopwatch(name)


def strnow(format='%Y/%m/%d %H:%M:%S'):
    return datetime.now().strftime(format)


class StrNow(object):
    ''' 呼び出し時の時刻を文字列で返す'''
    def __init__(self, format='%Y%m%d_%H%M%S'):
        self.format = format

    def __str__(self):
        return datetime.now().strftime(self.format)

snow = StrNow()


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


def select_file(path='.', key=None, files=None, nselect=None, idx=None):
    def key_f(file):
        if key is None:
            return True
        elif callable(key):
            return key(file)
        elif isinstance(key, str):
            return re.match(key, file)
        else:
            return key.match(file)

    def exit_():
        print('exit')
        sys.exit(0)

    if not nselect:
        nselect = 0

    if files is None:
        files = filter(key_f, os.listdir(path))
    else:
        path = ''
    files = fsort(files)[-nselect:]

    if not files:
        exit_()

    print('### SELECT FILE ###')
    for i, file in enumerate(files):
        size = filesize(os.path.join(path, file))
        if size < 10 * MB1:
            print(f'[{i}]', file, f'({size//KB1}KB)')
        else:
            print(f'[{i}]', file, f'({size//MB1}MB)')
    print(f'[{len(files)}] Exit')
    print('input number:')

    try:
        if idx is not None:
            n = idx
        else:
            n = int(input())
        file = files[n]
        return os.path.join(path, file)

    except (ValueError, IndexError):
        exit_()


def remove_empty_dirs(path='.', depth=0, ignore_error=False):
    ''' 空/削除済み=>True, それ以外=>False
    '''

    if not os.path.exists(path):
        if os.path.islink(path) or os.path.isfile(path) or os.path.isdir(path):
            print('RM(l):', path)
            try:
                os.remove(path)
            except PermissionError as e:
                print(e)
                if ignore_error:
                    return False
                else:
                    raise
        else:
            raise FileNotFoundError
        return True

    if os.path.isfile(path):
        return False

    if os.path.isdir(path):
        if os.path.islink(path):
            is_empty = remove_empty_dirs(os.readlink(path), depth=depth)
        else:
            print('--' * depth, path)
            is_empty = True
            try:
                with chdir(path):
                    for file in os.listdir('.'):
                        is_empty &= remove_empty_dirs(file, depth=depth+1)
            except PermissionError as e:
                print(e)
                if ignore_error:
                    return False
                else:
                    raise
        if is_empty:
            print('RM(d):', path)
            os.rmdir(path)
        return is_empty

    raise Exception('Unexpected Error')


################################################################################
# pickle
################################################################################

def save(file, item, to_json=False):
    mkdir(os.path.dirname(file))
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

    def get_fortran_function(self, name, argtypes, restype='void',
                             callback=lambda *x: x):
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


################################################################################
# E-mail
################################################################################

class EmailIO(object):
    def __init__(self, to=None, subject=''):
        self.to = to
        self.subject = subject
        self.msgs = []
        self.addr = None
        self.pw = None
        self.load_addr(os.path.join(SRC_DIR, '__local__', 'email.json'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.addr:
            return

        if exc_type is None:
            message = ''.join(self.msgs)
            self.send_email(message)

    def write(self, msg):
        self.msgs.append(msg)

    def print(self, *msgs):
        self.msgs.append(' '.join(map(str, msgs)) + '\n')

    def load_addr(self, file):
        if not os.path.isfile(file):
            print('EmailIO: user info file is not found')
            return
        info = load(file, from_json=True)
        self.addr = info['addr']
        self.pw = info['pw']

    def send_email(self, message):
        to_addr = self.to or self.addr
        subject = ' '.join(filter(None, (self.subject, '[MyPython]')))
        send_email(self.addr, self.pw, to_addr, subject, message)


def send_email(from_addr, pw, to_addr, subject, message):
    msg = EmailMessage()
    msg.set_content(message)

    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr

    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.starttls()
        s.login(from_addr, pw)
        s.send_message(msg)


################################################################################


class TestIOClass(object):
    def __init__(self):
        self.a = []

    def write(self, s):
        print('io:', s)
        self.a.append(s)


def __test__():
    io = TestIOClass()
    print('test', file=io)
    print(io.a)

if __name__ == '__main__':
    __test__()
