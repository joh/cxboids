import matplotlib
matplotlib.use('TkAgg')
from pylab import *
import numpy as np
import copy
from itertools import izip

dimensions = np.array([100, 100])
num_boids = 50
dt = 1.0
neighbor_radius = 10
desired_distance = 5
max_speed = 2.0
separation_weight = 0.2
alignment_weight = 0.2
cohesion_weight = 0.2

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

    def cohese(self, neighbors):
        center = np.array([0., 0.])
        for b in neighbors:
            center += b.pos / len(neighbors)

        accel = np.array([0., 0.])

        if len(neighbors) > 0:
            delta = distance(self.pos, center)
            accel = normalized(delta)

        return accel

    def flock(self, neighbors, distances):
        accel = np.array([0.,0.])
        accel += separation_weight * self.separate(neighbors, distances)
        accel += alignment_weight * self.align(neighbors)
        accel += cohesion_weight * self.cohese(neighbors)
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


def init():
    global boids

    boids = []
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

def draw():
    global boids

    xs = [b.pos[0] for b in boids]
    ys = [b.pos[1] for b in boids]
    vs = np.array([b.hdg for b in boids])

    cla()
    #plot(xs, ys, 'bo')
    quiver(xs, ys, vs[:,0], vs[:,1], pivot='mid', headaxislength=8, headlength=8)
    xlim(0, dimensions[0])
    ylim(0, dimensions[1])

def update():
    global boids
    #print "========="
    for b in boids:
        #print "step(boid{})".format(id(b))
        neighbors = []
        distances = []
        for o in boids:
            if o == b:
                continue
            d = b.distance(o)
            if norm(d) < neighbor_radius:
                neighbors.append(o)
                distances.append(d)
        #print "neighbors=", map(id, neighbors)
        #print "distances=", distances
        b.step(neighbors, distances)

if __name__ == '__main__':
    import pycxsimulator

    def Num_Boids(val=num_boids):
        global num_boids
        num_boids = int(abs(val))
        return val

    def Neighbor_Radius(val=neighbor_radius):
        global neighbor_radius
        neighbor_radius = float(val)
        return val

    def Desired_Distance(val=desired_distance):
        global desired_distance
        desired_distance = float(val)
        return val

    def Max_Speed(val=max_speed):
        global max_speed
        max_speed = float(val)
        return val

    def Separation_Weight(val=separation_weight):
        global separation_weight
        separation_weight = float(val)
        return val

    def Alignment_Weight(val=alignment_weight):
        global alignment_weight
        alignment_weight = float(val)
        return val

    def Cohesion_Weight(val=cohesion_weight):
        global cohesion_weight
        cohesion_weight = float(val)
        return val

    pSetters = [
        Num_Boids,
        Neighbor_Radius,
        Desired_Distance,
        Max_Speed,
        Separation_Weight,
        Alignment_Weight,
        Cohesion_Weight ]

    pycxsimulator.GUI(parameterSetters=pSetters).start(func=[init, draw, update])

