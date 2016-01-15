import sys
import os
import numpy as np
import matplotlib
import re
matplotlib.use('TkAgg')
from pylab import *
import cxboids
import argparse

# Parse cmdline
parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('-t', '--title')
parser.add_argument('-y', '--ylabel')
parser.add_argument('-s', '--stat', default='#clusters')

args = parser.parse_args()
print args

stat=args.stat
stat_index=cxboids.stats_key.index(stat)

figure(figsize=(20,7))
xlabel("Time")

if args.title:
    title(args.title)
else:
    title(stat)

if args.ylabel:
    ylabel(args.ylabel)
else:
    ylabel(stat)

n_vals = []
s_vals = []
a_vals = []
c_vals = []
for f in args.files:
    m = re.search(r'run(\d*)_s(\d+)_a(\d+)_c(\d+)', f)
    n, s, a, c = m.groups()
    n_vals.append(n)
    s_vals.append(s)
    a_vals.append(a)
    c_vals.append(c)

varyings = ['n', 's', 'a', 'c']
if len(unique(n_vals)) == 1:
    varyings.remove('n')
if len(unique(s_vals)) == 1:
    varyings.remove('s')
if len(unique(a_vals)) == 1:
    varyings.remove('a')
if len(unique(c_vals)) == 1:
    varyings.remove('c')

for i,f in enumerate(args.files):
    data = np.genfromtxt(f, delimiter=', ')
    l = os.path.basename(f)
    n_cols = data.shape[1]
    n_runs = data.shape[0] / len(np.unique(data[:,0]))
    n_steps = np.max(data[:,0])

    label = []
    if 'n' in varyings:
        label.append("$N={}$".format(n_vals[i]))
    if 's' in varyings:
        label.append("$S_w={}$".format(s_vals[i]))
    if 'a' in varyings:
        label.append("$A_w={}$".format(a_vals[i]))
    if 'c' in varyings:
        label.append("$C_w={}$".format(c_vals[i]))

    label = ", ".join(label)

    print f, "n_runs=", n_runs, "n_steps=", n_steps
    data = data.reshape((n_runs,-1,n_cols))
    data = np.mean(data, axis=0)
    #data = data[:n_steps]

    plot(data[:,stat_index], label=label)
    #plot(data[:,8], label=l + ' #neighbors')
    #plot(data[:,10], label=l + ' neighbor_distance')
    #plot(data[:,11], label=l + ' neighbor_distance_var')

    #plot(data[:,6], label=l + ' cluster_size')
    #plot(np.sqrt(data[:,9]), label=f + ' var(#neighbors)')

# Shrink current axis by 20%
box = gca().get_position()
gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend(loc='upper left', bbox_to_anchor=(1.0, 1.02))
show()
