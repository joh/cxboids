import sys
import cxboids

n_runs = 10

cxboids.num_boids = 30
cxboids.dt = 1.0
cxboids.neighbor_radius = 10
cxboids.desired_distance = 5
cxboids.max_speed = 2.0
cxboids.separation_weight = 1.0
cxboids.alignment_weight = 1.0
cxboids.cohesion_weight = 1.0

def run(n_steps):
    cxboids.init()
    for s in xrange(1, n_steps):
        sys.stdout.write('.')
        sys.stdout.flush()
        cxboids.update()
    return list(cxboids.history)

stats = []
for r in xrange(n_runs):
    print "Run {}: ".format(r),
    s = run(500)
    stats.extend(s)
    print

cxboids.save_stats(stats, 'runs.csv')
