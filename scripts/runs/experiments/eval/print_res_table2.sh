#!/bin/bash

set -e

folder=$1


python scripts/evaluation/overall_performance.py --exps-dir $folder


