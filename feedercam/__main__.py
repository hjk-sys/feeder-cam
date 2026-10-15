"""Command line entry point: python -m feedercam <command>

    run         capture loop (Pi camera, webcam, or --source video.mp4)
    ingest      load new visits from the incoming folder into SQLite
    classify    predict species for unclassified visits
    export      copy reviewed crops into training data
    stats       quick text summary of the log
"""

import argparse
import logging
import sys

from . import db
from .config import load_config, resolve


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

    sub.add_parser("ingest", help="load incoming visits into the database")
    p = sub.add_parser("classify", help="predict species for visits")
    p.add_argument("--redo", action="store_true", help="re-classify visits labeled by an older model")
    sub.add_parser("export", help="copy reviewed crops into data/training/corrections")
    sub.add_parser("stats", help="print a summary of logged visits")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config, args.set)

    if args.command == "run":
        from .pipeline import run
        run(cfg, args.source, args.preview, args.annotate, args.max_seconds)
        return

    conn = db.connect(resolve(cfg["paths"]["db"]))
    if args.command == "ingest":
        n = db.ingest(conn, resolve(cfg["paths"]["incoming"]))
        print(f"ingested {n} new visits")
    elif args.command == "classify":
        from .classify import classify_pending
        n = classify_pending(conn, cfg, redo=args.redo)
        print(f"classified {n} visits")
    elif args.command == "export":
        from .classify import export_corrections
        n = export_corrections(conn, cfg)
        print(f"copied {n} reviewed crops into training data")
    elif args.command == "stats":
        print_stats(conn, cfg["classifier"]["low_conf"])


def print_stats(conn, low_conf):
    total = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM visits WHERE species IS NULL").fetchone()[0]
    review = conn.execute(
        "SELECT COUNT(*) FROM visits WHERE species_conf < ? AND corrected_species IS NULL",
        (low_conf,),
    ).fetchone()[0]
    print(f"{total} visits logged, {pending} unclassified, {review} waiting for review\n")
    rows = conn.execute(
        """SELECT final_species, COUNT(*) AS n FROM labeled_visits
           WHERE final_species IS NOT NULL GROUP BY final_species ORDER BY n DESC"""
    ).fetchall()
    for r in rows:
        print(f"  {r['n']:5d}  {r['final_species']}")


if __name__ == "__main__":
    sys.exit(main())
