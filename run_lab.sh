#!/bin/bash
cd /code/dsb2019
pip install -e .
jupyter lab --ip 0.0.0.0 --allow-root
