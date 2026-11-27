"""Explicit entry points for the new single-frame depth locator."""
import argparse
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','infer','monitor'])
    argv=list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('--help','-h'):
        parser.print_help();return 0
    args=parser.parse_args(argv[:1]);rest=argv[1:]
    if args.command=='train':from .training import main as run
    elif args.command=='infer':from .infer import main as run
    else:from .monitor import main as run
    return run(rest)


if __name__=='__main__':raise SystemExit(main())
