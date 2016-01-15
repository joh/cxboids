import matplotlib
matplotlib.use('TkAgg')
from pylab import *
import numpy as np
import copy
from itertools import izip

dimensions = np.array([100, 100])
num_boids = 30
step = 0
dt = 1.0
neighbor_radius = 10
desired_distance = 5
max_speed = 2.0
separation_weight = 2.0
alignment_weight = 1.0
cohesion_weight = 1.0
history = []

def normalized(v):
    v = np.array(v) # copy
    n = norm(v)
    if n != 0:
        v /= n
    return v

def distance(v0, v1):
    # Distance which takes wrapping into account
    dx0, dy0 = (v1 - v0) % dimensions   # v0 behind v1
    dx1, dy1 = (v0 - v1) % dimensions   # v0 ahead of v1

    dx = dx0 if dx0 < dx1 else -dx1
    dy = dy0 if dy0 < dy1 else -dy1

    d = np.array([dx, dy])

    return d

class Boid(object):
    boid_id = 0

    def __init__(self, pos, vel):
        self.id = Boid.boid_id
        Boid.boid_id += 1
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self._actor = None

        if norm(self.vel) != 0:
            self.hdg = normalized(self.vel)
        else:
            self.hdg = np.array([1,0])

    def __repr__(self):
        return "Boid#{}(id={}, pos={}, vel={})".format(id(self), self.id, self.pos, self.vel)

    def __eq__(self, other):
        return self.id == other.id

    def distance(self, other):
        return distance(self.pos, other.pos)

    def separate(self, neighbors, distances):
        accel = np.array([0., 0.])
        count = 0
        for b, d in izip(neighbors, distances):
            dm = norm(d)
            if dm < desired_distance:
                accel += -1 * (desired_distance - dm) * d
                count += 1

        if count > 0:
            accel = normalized(accel)
            #accel /= count

        return accel

    def align(self, neighbors):
        accel = np.array([0., 0.])
        for b in neighbors:
            accel += b.vel / len(neighbors)

        if len(neighbors) > 0:
            accel = normalized(accel)

        return accel

    def cohese(self, neighbors, distances):
        center = np.array([0., 0.])
        for b, d in izip(neighbors, distances):
            center += (self.pos + d) / len(neighbors)

        accel = np.array([0., 0.])

        if len(neighbors) > 0:
            delta = distance(self.pos, center)
            accel = normalized(delta)

        return accel

    def flock(self, neighbors, distances):
        accel = np.array([0.,0.])
        accel += separation_weight * self.separate(neighbors, distances)
        accel += alignment_weight * self.align(neighbors)
        accel += cohesion_weight * self.cohese(neighbors, distances)
        return accel

    def step(self, neighbors, distances):
        #print "Boid#{} {} neighbors".format(self.id, len(neighbors))

        accel = self.flock(neighbors, distances)
        #print "\taccel={}".format(accel)

        self.vel += accel * dt
        if norm(self.vel) > max_speed:
            self.vel = max_speed * normalized(self.vel)
        self.pos += self.vel * dt
        self.pos %= dimensions  # wrap

        if norm(self.vel) != 0:
            # Keep track of heading only when velocity != 0
            self.hdg = normalized(self.vel)


def get_neighbors(boid):
    neighbors = []
    distances = []
    for b in boids:
        if b == boid:
            continue
        d = boid.distance(b)
        if norm(d) < neighbor_radius:
            neighbors.append(b)
            distances.append(d)

    return neighbors, distances

def expand_cluster(boid):
    visited = set()
    queue = [boid]

    while queue:
        b = queue.pop()
        visited.add(b)
        neighbors,_ = get_neighbors(b)
        for n in neighbors:
            if n not in visited:
                queue.append(n)

    return visited

def find_clusters():
    remaining = set(boids)
    clusters = []
    while remaining:
        boid = remaining.pop()
        cl = expand_cluster(boid)
        clusters.append(len(cl))
        remaining -= set(cl)
    return clusters

def stats():
    velocity = []
    heading = []
    num_neighbors = []
    neighbor_distance = []
    neighbor_heading = []

    for b in boids:
        velocity.append(norm(b.vel))
        heading.append(np.rad2deg(np.arctan2(*b.hdg)))
        neighbors, distances = get_neighbors(b)
        num_neighbors.append(len(neighbors))
        if len(neighbors) > 0:
            neighbor_distance.append(np.mean(map(norm, distances)))
            nhdg = np.rad2deg(np.mean([np.arctan2(*o.hdg) for o in neighbors]))
            neighbor_heading.append(nhdg)

    clusters = find_clusters()

    return (step,
            np.mean(velocity), np.var(velocity),
            np.mean(heading), np.var(heading),
            len(clusters), np.mean(clusters), np.var(clusters),
            np.mean(num_neighbors), np.var(num_neighbors),
            np.mean(neighbor_distance), np.var(neighbor_distance),
            np.min(neighbor_distance), np.max(neighbor_distance),
            np.mean(neighbor_heading), np.var(neighbor_heading),
            )

stats_key = ["step", "vel_mean", "vel_var", "hdg_mean", "hdg_var",
             "#clusters", "cluster_size_mean", "cluster_size_var",
             "#neighbors_mean", "#neighbors_var",
             "neighbor_distance_mean", "neighbor_distance_var",
             "neighbor_distance_min", "neighbor_distance_max",
             "neighbor_heading_mean", "neighbor_heading_var",
             ]

def print_stats(stats):
    step = stats[0]

    if step % 10 == 0:
        print "# " + ", ".join(stats_key)

    fmt = ["%d"] + ["%.1f"] * (len(stats) - 1)
    fmt = ", ".join(fmt)
    print fmt % stats

def save_stats(stats, filename):
    header = ", ".join(stats_key)
    np.savetxt(filename, stats, fmt='%.2f', delimiter=', ', header=header)
    print "Saved %d lines of stats to %s" % (len(stats), filename)

def init():
    global boids, step, history

    if history:
        # Save history to csv
        save_stats(history, "stats.csv")

    history = []
    boids = []
    step = 0
    #boids.append(Boid([40,40], normalized([1.1,1.])))
    #boids.append(Boid([60,60], normalized([-1.,-1.])))
    cx, cy = dimensions / 2

    for i in xrange(num_boids):
        x = np.random.uniform(0, dimensions[0])
        y = np.random.uniform(0, dimensions[1])
        pos = np.array([x, y])
        vel = np.random.uniform(-1, 1, size=2)
        #pos = np.array([50., 50.])
        #vel = np.array([1., 0.])
        #vel = np.random.choice([-1.0, 1.0], size=2)
        b = Boid(pos, vel)
        boids.append(b)

    s = stats()
    history.append(s)
    #print_stats(s)

def draw():
    global boids, step

    if step == 0:
        cla()
        xlim(0, dimensions[0])
        ylim(0, dimensions[1])
        for b in boids:
            b._actor = quiver(b.pos[0], b.pos[1], b.hdg[0], b.hdg[1], \
                    pivot='mid', headaxislength=8, headlength=8)
    else:
        for b in boids:
            q = b._actor
            q.set_offsets(b.pos.copy())
            q.set_UVC(b.hdg[0], b.hdg[1])

def update():
    global boids, step
    #print "========="
    for b in boids:
        #print "step(boid{})".format(id(b))
        neighbors, distances = get_neighbors(b)
        b.step(neighbors, distances)

    step += 1

    s = stats()
    history.append(s)
    #print_stats(s)

if __name__ == '__main__':
    import pycxsimulator

    def Num_Boids(val=num_boids):
        global num_boids
        num_boids = int(abs(val))
        return val

    def Dim_x(val=dimensions[0]):
        global dimensions
        dimensions[0] = int(abs(val))
        return val

    def Dim_y(val=dimensions[1]):
        global dimensions
        dimensions[1] = int(abs(val))
        return val

    def Dt(val=dt):
        global dt
        dt = float(abs(val))
        return val

    def Neighbor_Radius(val=neighbor_radius):
        global neighbor_radius
        neighbor_radius = float(abs(val))
        return val

    def Desired_Distance(val=desired_distance):
        global desired_distance
        desired_distance = float(abs(val))
        return val

    def Max_Speed(val=max_speed):
        global max_speed
        max_speed = float(abs(val))
        return val

    def Separation_Weight(val=separation_weight):
        global separation_weight
        separation_weight = float(abs(val))
        return val

    def Alignment_Weight(val=alignment_weight):
        global alignment_weight
        alignment_weight = float(abs(val))
        return val

    def Cohesion_Weight(val=cohesion_weight):
        global cohesion_weight
        cohesion_weight = float(abs(val))
        return val

    pSetters = [
        Num_Boids,
        Dim_x,
        Dim_y,
        Dt,
        Neighbor_Radius,
        Desired_Distance,
        Max_Speed,
        Separation_Weight,
        Alignment_Weight,
        Cohesion_Weight ]

    pycxsimulator.GUI(parameterSetters=pSetters).start(func=[init, draw, update])

