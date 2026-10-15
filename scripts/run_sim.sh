#!/usr/bin/env bash
# End-to-end test on a simulated feeder: make a video with a known answer key,
# run the whole pipeline on it, and score the result.
#
#   scripts/run_sim.sh [minutes]
#
# Everything goes in data/sim (override with SIM_DIR), separate from real data.
set -euo pipefail
cd "$(dirname "$0")/.."
MINUTES="${1:-3}"
SIM="${SIM_DIR:-data/sim}"
CKPT="${CHECKPOINT:-models/species_vit.pt}"
CROPS="${CROPS:-data/training/crops}"
PY=.venv/bin/python

# Start each run from an empty log so the score only covers this video.
rm -rf "$SIM/incoming" "$SIM/feeder.db"
SET=(--set device_name=sim --set "paths.incoming=$SIM/incoming"
     --set "paths.db=$SIM/feeder.db" --set "classifier.checkpoint=$CKPT")

$PY scripts/simulate_feeder.py --crops "$CROPS" --out "$SIM/feeder_sim.mp4" --minutes "$MINUTES"
$PY -m feedercam "${SET[@]}" run --source "$SIM/feeder_sim.mp4" --annotate "$SIM/annotated.mp4"
$PY -m feedercam "${SET[@]}" ingest
$PY -m feedercam "${SET[@]}" classify
echo
$PY scripts/score_sim.py --key "$SIM/feeder_sim.json" --db "$SIM/feeder.db"
echo
echo "Annotated video: $SIM/annotated.mp4"
echo "Dashboard: .venv/bin/streamlit run dashboard/app.py -- --set paths.db=$SIM/feeder.db --set paths.incoming=$SIM/incoming"
