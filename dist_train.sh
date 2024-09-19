#!/usr/bin/env bash

#CONFIG=$1
GPUS=$1
PORT=${PORT:-29500}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
$(dirname "$0")/train.py ${@:2}