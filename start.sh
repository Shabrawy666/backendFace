#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1