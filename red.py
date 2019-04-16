import argparse
import myutils as ut


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='*', default='',
                        help='paht list')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    for path in args.path:
        ut.remove_empty_dirs(path, ignore_errors=False)


if __name__ == '__main__':
    main()
