import sys
import numpy as np
import matplotlib
import re
matplotlib.use('TkAgg')
from pylab import *

for f in sys.argv[1:]:
    data = np.genfromtxt(f, delimiter=', ')
    n_runs = len(data[:,0]) / len(np.unique(data[:,0]))
    #m = re.search(r'rf(\d+)_df(\d+)', f)
    #rf, df = m.groups()
    print f, "n_runs=", n_runs#, "rf=", rf, "df=", df
    data = data.reshape((n_runs,-1,14))
    data = np.mean(data, axis=0)

    #plot(data[:,1], label='vel')
    #plot(data[:,3], label='hdg')
    plot(data[:,5], label='#clusters')
    plot(data[:,6], label='cluster_size')
    #plot(data[:,8], label='#neighbors')
    #plot(data[:,10]/data[:,8], label='neighbor_distance')
    #plot(data[:,11]/data[:,8], label='neighbor_distance_var')

legend()
show()
