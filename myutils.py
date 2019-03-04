import ctypes
import glob
import hashlib
import json
import os
import pickle
import re
import shutil
import smtplib
import subprocess
import sys
import threading
import time
import unicodedata
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
# path
################################################################################

def repair_path(path):
    ''' ファイルパスの不正文字を置換
    '''

    table = str.maketrans('\\/:*?"<>|', '￥／：＊？”＜＞｜', '')
    return path.translate(table)


def uniq_path(path, ftype='file', key=r'_(\d{1,3})$', suf=lambda n: f'_{n}'):
    ''' 同名のファイルが存在した場合にプレフィックスを追加する
    '''

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


def realpath(path):
    ''' ファイルの実体のパスを返す
    '''

    abspath = os.path.abspath(path)
    if not os.path.exists(abspath):
        raise FileNotFoundError(abspath)
    if os.path.islink(abspath):
        return realpath(os.readlink(abspath))
    dirname = os.path.dirname(abspath)
    basename = os.path.basename(abspath)
    if abspath == dirname:
        return abspath
    return os.path.join(realpath(dirname), basename)


def md5(s):
    ''' md5値を返す
    '''

    return hashlib.md5(s).hexdigest()


def fsort(l, key=None):
    ''' 数値を分離して考慮したソートを行う
    '''
    pattern = re.compile(r'(\D+)|(\d+)')

    def f_key(item):
        if key:
            item = key(item)
        if isinstance(item, str):
            return [a or int(b) for a, b in pattern.findall('_' + item)]
        elif hasattr(item, '__iter__') or hasattr(item, '__getitem__'):
            return list(map(f_key, item))
        else:
            return item

    # key = lambda s: [a or int(b) for a, b in pattern.findall('_' + s)]
    return sorted(l, key=f_key)


def select_file(path='.', key=None, files=None, nselect=None, idx=None):
    ''' 条件に合うファイル一覧を表示しユーザーに選択させる
    '''

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


def basename(path, suffix=''):
    fname = os.path.basename(path)
    if suffix:
        bname, ext = os.path.splitext(fname)
        if suffix == ext or suffix == '.*':
            return bname
    return fname


################################################################################
# shell
################################################################################

@contextmanager
def chdir(path):
    ''' カレントディレクトリを変更する
    with chdir(dirname):
        ...
    '''

    prev_path = os.getcwd()
    if path:
        mkdir(path)
        os.chdir(path)
    try:
        yield os.getcwd()
    finally:
        os.chdir(prev_path)


def open_loc(file):
    ''' ファイルが選択された状態でエクスプローラを開く
    '''

    subprocess.run(['explorer', '/select,', file.replace('/', '\\')])


def popen(*cmds):
    ''' 外部コマンドを実行し標準出力をイテレータで返す
    '''

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
    ''' 条件に一致するファイル一覧をイテレータで返す
    '''

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
    ''' 条件に一致するファイル一覧を配列で返す
    '''

    return list(iglobm(pathname, recursive=recursive, sep=os.sep))


# def rmempty(path, rm=False):
#     raise
#     if not os.path.isdir(path):
#         return True
#     L = [x for x in (rmempty(os.path.join(path, f), rm=rm)
#                      for f in os.listdir(path)) if x]
#     if not L and rm:
#         print('Removing:', path)
#         os.rmdir(path)
#     return L


def makedirs(path):
    ''' 深い階層のディレクトリを作成する
    '''
    if path:
        os.makedirs(path, exist_ok=True)


def mkdir(*args, **kwargs):
    return makedirs(*args, **kwargs)


def into_dir(src, dst, force=False):
    ''' ファイルまたはディレクトリをディレクトリに移動する
    '''

    if os.path.isfile(dst):
        raise OSError(f'{dst} is a file')
    with chdir(dst):
        file = os.path.basename(src)
        if os.path.exists(file):
            if force:
                os.remove(file)
            else:
                raise OSError(f'{file} already exists')
    shutil.move(src, dst)


def filesize(path='.', follow_symlinks=False):
    ''' ファイル又はディレクトリの大きさを返す
    '''

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


def remove_empty_dirs(path='.', depth=0):
    ''' 空のディレクトリを再帰的に削除する
    '''

    if not os.path.exists(path):
        if os.path.islink(path) or os.path.isfile(path) or os.path.isdir(path):
            print('RM(l):', path)
            os.remove(path)
        else:
            raise FileNotFoundError(os.path.abspath(path))
        return 0

    if os.path.isfile(path):
        return 1

    if os.path.isdir(path):
        if os.path.islink(path):
            n_files = remove_empty_dirs(os.readlink(path), depth=depth)

        elif os.listdir(path) == 0:
            n_files = 0

        else:
            print(f'{depth:<2}' + '--' * depth, path)
            with chdir(path):
                n_files = sum(remove_empty_dirs(file, depth=depth+1)
                              for file in os.listdir('.'))

        if n_files == 0:
            print('RM(d):', path)
            os.rmdir(path)

        return n_files

    raise Exception('Unexpected Error')


################################################################################
# stopwatch
################################################################################

class Stopwatch(object):
    ''' 実行時間測定用クラス
    with ut.Stopwatch(name):
        ...
    '''

    def __init__(self, name='anonymous', start=None):
        self.name = name
        self.start = start or time.time()

    def __call__(self, name='anonymous'):
        # return time.time() - self.start
        return Stopwatch(name, start=self.start)

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = float(self)
        if elapsed < 10:
            return

        elif elapsed < 60:
            stime = f'{elapsed:.3g}秒'

        else:
            stime = f'{elapsed:.0f}秒 ({parse_time(int(elapsed))})'

        print(f'[Stopwatch@{strnow()}] {self.name}: {stime}')
        if exc_type is not None:
            print('Exception:', exc_type, exc_value)
            print('---')

    def __float__(self):
        return time.time() - self.start

    def __int__(self):
        return int(float(self))

    def __str__(self):
        return time2str(float(self))

    def __lt__(self, other):
        return float(self) < other

    def __gt__(self, other):
        return float(self) > other


def time2str(sec, show_ms=False):
    ''' timeオブジェクトを文字列に変換(コロン連結)
    '''

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
    res = sign + reduce(f_, [h, m, s], '')
    if show_ms:
        return res + f'.{int(sec * 1000) % 1000 :03d}'
    else:
        return res


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


# def stopwatch(name='anonymous'):
#     return Stopwatch(name)

stopwatch = Stopwatch()

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
# string
################################################################################

def clip_str_it(s, l):
    for c in s:
        if unicodedata.east_asian_width(c) in 'FWA':
            l -= 2
        else:
            l -= 1
        if l < 0:
            break
        yield c


def clip_str(string, limit):
    ''' 全角文字を含む文字列を指定の長さ以下に切り詰めて返す
    '''

    return ''.join(clip_str_it(string, limit))


def wrap_str_it(string, limit):
    l = limit
    for c in string:
        csize = 2 if unicodedata.east_asian_width(c) in 'FWA' else 1
        if l < csize:
            yield '\n'
            l = limit
        l -= csize
        yield c


def wrap_str(string, limit):
    ''' 全角文字を含む文字列を指定の長さで折り返す
    '''

    return ''.join(wrap_str_it(string, limit))


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
        with open(file, 'r', encoding='utf_8_sig') as f:
            return json.load(f)
    else:
        with open(file, 'rb') as f:
            return pickle.load(f)


################################################################################
# multi threading
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
