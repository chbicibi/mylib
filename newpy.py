#! /usr/bin/env python3

'''
Abstruct
'''

import argparse
import os
import shutil


################################################################################

def get_args():
    '''
    docstring for get_args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '-o', default='new_script',
                        help='Filename of the new script')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    '''
    docstring for main.
    '''
    args = get_args()
    file = args.out

    if not os.path.splitext(file)[1] == '.py':
        file = file + '.py'

    if not os.path.exists(file):
        shutil.copy(__file__, file)
        print('create:', file)


if __name__ == '__main__':
    main()
