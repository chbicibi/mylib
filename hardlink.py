import os
import sys

import myutils as ut


def main():
    argv = sys.argv[1:]
    dest = 'link'

    for file in map(os.path.abspath, argv):
        dirname = os.path.dirname(file)
        basename = os.path.basename(file)

        if os.path.isfile(file):
            with ut.chdir(os.path.join(dirname, dest)):
                if not os.path.isfile(basename):
                    os.link(file, basename)


if __name__ == '__main__':
    main()
