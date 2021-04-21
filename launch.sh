#!/bin/sh
google-chrome http://127.0.0.1:8050/
python3 main.py --folder data/sample_volume
echo "Please refresh chrome page."