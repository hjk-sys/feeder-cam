"""Command line entry point: python -m feedercam <command>

    run         capture loop (Pi camera, webcam, or --source video.mp4)
"""

import argparse
import logging
import sys

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="feedercam", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set detector.conf=0.5")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the capture + detection loop")
    p.add_argument("--source", help="video file to process instead of the configured camera")
    p.add_argument("--preview", action="store_true", help="show a live window (press q to quit)")
    p.add_argument("--annotate", metavar="OUT.mp4", help="write a video with boxes drawn on it")
    p.add_argument("--max-seconds", type=float, help="stop after this many seconds of footage")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config, args.set)

    if args.command == "run":
        from .pipeline import run
        run(cfg, args.source, args.preview, args.annotate, args.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
