#!/usr/bin/env bash
# On the laptop: copy new visits from the Pi, load them, and classify them.
#
#   scripts/sync.sh pi@feeder.local
#
# Needs SSH key login to the Pi. PI_DIR overrides where the repo lives on the Pi.
set -euo pipefail
PI="${1:?usage: scripts/sync.sh user@pi-hostname}"
PI_DIR="${PI_DIR:-feeder-cam}"
cd "$(dirname "$0")/.."

# -a copies new crops and the grown-since-last-time .jsonl logs.
rsync -a --stats "$PI:$PI_DIR/data/incoming/" data/incoming/

.venv/bin/python -m feedercam ingest
.venv/bin/python -m feedercam classify
.venv/bin/python -m feedercam stats
