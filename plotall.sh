#!/bin/sh
set -e
set -x
set -u

files="run_s1_a0_c0.csv run_s0_a1_c0.csv run_s0_a0_c1.csv"
python ../plot-stats.py -t "Isolated mechanisms: #flocks" -y "#flocks" -s "#clusters" $files
python ../plot-stats.py -t "Isolated mechanisms: #neighbors" -y "#neighbors" -s "#neighbors_mean" $files
python ../plot-stats.py -t "Isolated mechanisms: neighbor distance" -y "neighbor distance" -s "neighbor_distance_mean" $files
python ../plot-stats.py -t "Isolated mechanisms: neighbor distance variance" -y "VAR(neighbor distance)" -s "neighbor_distance_var" $files

files="run_s[1,3,5,7]_a1_c1.csv"
python ../plot-stats.py -t "Separation: #flocks" -y "#flocks" -s "#clusters" $files
python ../plot-stats.py -t "Separation: #neighbors" -y "#neighbors" -s "#neighbors_mean" $files
python ../plot-stats.py -t "Separation: neighbor distance" -y "neighbor distance" -s "neighbor_distance_mean" $files

files="run_s1_a[1,3,5,7]_c1.csv"
python ../plot-stats.py -t "Alignment: #flocks" -y "#flocks" -s "#clusters" $files
python ../plot-stats.py -t "Alignment: #neighbors" -y "#neighbors" -s "#neighbors_mean" $files
python ../plot-stats.py -t "Alignment: neighbor distance" -y "neighbor distance" -s "neighbor_distance_mean" $files

files="run_s1_a1_c[1,3,5,7].csv"
python ../plot-stats.py -t "Cohesion: #flocks" -y "#flocks" -s "#clusters" $files
python ../plot-stats.py -t "Cohesion: #neighbors" -y "#neighbors" -s "#neighbors_mean" $files
python ../plot-stats.py -t "Cohesion: neighbor distance" -y "neighbor distance" -s "neighbor_distance_mean" $files

files="run30_s2_a1_c1.csv run60_s2_a1_c1.csv run120_s2_a1_c1.csv"
python ../plot-stats.py -t "Scaling up: #flocks" -y "#flocks" -s "#clusters" $files
python ../plot-stats.py -t "Scaling up: #neighbors" -y "#neighbors" -s "#neighbors_mean" $files
python ../plot-stats.py -t "Scaling up: neighbor distance" -y "neighbor distance" -s "neighbor_distance_mean" $files
