#!/bin/bash
set -e

work_path=$(dirname $(readlink -f $0))
cd ${work_path}/Forward_Warp/cuda/
python setup.py install
cd ../../
python setup.py install
