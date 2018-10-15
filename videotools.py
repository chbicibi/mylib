#! /usr/bin/env python3

import cv2
import os
from contextlib import contextmanager

FFPROBE = 'ffprobe'
LOG_DIR = 'D:/Links/Error.log'

@contextmanager
def video_capture(file):
    if not os.path.isfile(file):
        raise FileNotFoundError

    cap = cv2.VideoCapture(file)
    try:
        yield cap
    except(cv2.error, ValueError):
        subprocess.run(['explorer', '/select,', file.replace('/', os.sep)])
        with open(LOG_DIR, 'a') as f:
            for s in [file, traceback.format_exc(), '=' * 100]:
                print(s, file=f)
    except KeyboardInterrupt as e:
        print('中止')
        raise e
    finally:
        print('Releasing VideoCapture')
        cap.release()


def get_format(file):
    with subprocess.Popen([FFPROBE, '-i', file, '-show_format'],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as proc:
        out, err = proc.communicate()
        m = re.search(r'(?<=format_name=).+?(?=[\r\n])', out.decode('utf-8'))
        return m and m.group(0)
