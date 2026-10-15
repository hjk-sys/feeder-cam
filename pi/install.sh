#!/usr/bin/env bash
# Set up feeder-cam on a Raspberry Pi (Raspberry Pi OS Bookworm, 64-bit).
# Run from anywhere inside the cloned repo:  bash pi/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

echo "==> System packages"
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-venv

echo "==> Python environment"
# picamera2 comes from apt, not pip, so the venv needs to see system packages.
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-pi.txt

echo "==> Exporting YOLOv8n to NCNN (several times faster than PyTorch on the Pi's CPU)"
.venv/bin/yolo export model=yolov8n.pt format=ncnn imgsz=640

# A Pi-specific config that isn't tracked by git, so `git pull` never conflicts.
if [ ! -f config.pi.yaml ]; then
  sed 's|weights: yolov8n.pt|weights: yolov8n_ncnn_model|' config.yaml > config.pi.yaml
  echo "==> Wrote config.pi.yaml (edit camera/motion settings there)"
fi

echo "==> systemd service (starts on boot, restarts if it crashes)"
sed -e "s|__USER__|$USER|g" -e "s|__REPO__|$REPO|g" pi/feedercam.service \
  | sudo tee /etc/systemd/system/feedercam.service > /dev/null
sudo systemctl daemon-reload

cat <<EOF

Done. Test it in the foreground first:
  .venv/bin/python -m feedercam --config config.pi.yaml run

Then run it as a service:
  sudo systemctl enable --now feedercam
  journalctl -u feedercam -f        # watch the log
EOF
