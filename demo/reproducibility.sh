#!/bin/bash --login
# reproducibility.sh
# Test the reproducibility of PecanPy between runs.

source ~/.bashrc

rs=100
export PYTHONHASHSEED=$rs

conda activate pecanpy-dev
pecanpy --input karate.edg --output karate1.emd --mode FirstOrderUnweighted --workers 1 --random_state $rs
pecanpy --input karate.edg --output karate2.emd --mode FirstOrderUnweighted --workers 1 --random_state $rs
cmp karate1.emd karate2.emd
rm -f karate1.emd karate2.emd
