import argparse
import glob
import random
import sys

import numpy as np

import solve


def main():
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--format', help='Input file path format',
                        required=True)
    parser.add_argument('--date', help='Date', required=True)
    parser.add_argument('--time_window', help='Time window', type=int)
    parser.add_argument('--mode', help='Mode')
    parser.add_argument('--quiet', help='Quiet', action='store_true')
    parser.add_argument('--debug', help='Output debug information',
                        action='store_true')
    args = parser.parse_args()

    if not args.quiet:
        sys.stderr.write('Date : {}\n'.format(args.date))

    files = []
    with open(args.format) as format_list:
        for format_s in format_list:
            format_s = format_s.strip()
            if format_s == '':
                continue
            files += glob.glob(format_s.format(date=args.date))

    solver = solve.Solver()
    solver(files, args.date, args.time_window, args.mode, args.quiet,
           args.debug)


if __name__ == '__main__':
    main()
