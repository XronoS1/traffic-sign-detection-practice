#!/bin/sh
set -e

python manage.py migrate
python manage.py seed_model_weights
python manage.py runserver 0.0.0.0:8000
