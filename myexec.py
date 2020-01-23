import sys
import subprocess
import threading
import time


class ProcReader(object):
    def __init__(self, cmd, timeout=60):
        if type(cmd) is str:
            shell = True
        else:
            cmd = list(map(str, cmd))
            shell = False
        self.proc = subprocess.Popen(cmd, shell=shell,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        self.timeout = timeout
        self.buffer = []
        self.start_listening()

    def __enter__(self):
        return self

    def __exit__(self, ty, val, tr):
        if self.proc.poll() is None:
            self.proc.terminate()
            # subprocess.run(['taskkill', '/pid', f'{self.proc.pid}'])
            time.sleep(3)

    def __iter__(self):
        limit = self.timeout
        while True:
            if self.timeout is None:
                yield from self.readline()

            elif self.buffer:
                yield from self.readline()
                limit = self.timeout

            elif limit > 0:
                limit -= 1
                time.sleep(1)

            else:
                print('timeout')
                self.proc.wait(timeout=0)
                break

            if self.proc.poll() is not None:
                break

    def run(self):
        for l in self:
            print(l, end='')

    def readline(self):
        while self.buffer:
            yield self.buffer.pop(0)

    def start_listening(self):
        def f_():
            b = b''
            while True:
                c = self.proc.stdout.read(1)
                b += c
                if c == b'\r' or c == b'\n':
                    self.buffer.append(b.decode())
                    b = b''

                if self.proc.poll() is not None:
                    self.buffer.append(b.decode())
                    break

        t = threading.Thread(target=f_, daemon=True)
        t.start()


################################################################################

def __test__():
    pass


def get_args():
    '''
    docstring for get_args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('out', nargs='?', default='new_script',
                        help='Filename of the new script')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    '''
    docstring for main.
    '''
    args = get_args()

    if args.test:
        __test__()
        return

    file = args.out

    if not os.path.splitext(file)[1] == '.py':
        file = file + '.py'

    if not args.force and os.path.exists(file):
        return

    shutil.copy(__file__, file)
    print('create:', file)


if __name__ == '__main__':
    main()
