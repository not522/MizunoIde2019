import math
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline


class TravelTime(object):
    def __init__(self):
        self._velocity = np.empty(1880, dtype=np.float32)
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.jma2001 = os.path.join(dir_path, os.pardir, 'JMA2001')

    def time(self, dep, dis, elv):
        v0 = self.velocity(0)
        ang = self.take_off.ev(dep, dis)
        x = elv * np.tan(ang)
        dis = dis - x
        if isinstance(dis, np.ndarray):
            dis[dis < 0] = 0
        else:
            if dis < 0:
                dis = 0
        offset = np.sign(elv) * np.hypot(elv, x) / v0
        return self._time.ev(dep, dis) + offset

    def grad(self, dep, dis, elv):
        ang = self.take_off.ev(dep, dis)
        dis -= elv * np.tan(ang)
        if dis < 0:
            dis = 0
        d_dep = self._time.ev(dep, dis, dx=1)
        d_dis = self._time.ev(dep, dis, dy=1)
        return d_dep, d_dis

    def velocity(self, dep):
        if dep < 0:
            return self._velocity[0]
        i_dep = int(2 * dep)
        dep -= i_dep * 0.5
        return (self._velocity[i_dep] * (0.5 - dep)
                + self._velocity[i_dep + 1] * dep) * 2


class TravelTimeS(TravelTime):
    def __init__(self):
        super(TravelTimeS, self).__init__()

        x = []
        y = []
        t = np.empty((106, 236), dtype=np.float32)
        with open(os.path.join(self.jma2001, 'travel_time')) as f:
            for line in f.readlines():
                _, _, _, st, dep, dis = line.split()
                st, dep, dis = map(float, (st, dep, dis))
                if dep not in x:
                    x.append(dep)
                if dis not in y:
                    y.append(dis)
                t[x.index(dep), y.index(dis)] = st
        self._time = RectBivariateSpline(x, y, t, kx=1, ky=1)

        with open(os.path.join(self.jma2001, 'velocity_structure')) as f:
            for i, line in enumerate(f.readlines()):
                _, vs, _ = line.split()
                vs = float(vs)
                self._velocity[i] = float(vs)

        x = []
        y = []
        a = np.empty((106, 236), dtype=np.float32)
        t = np.empty((106, 236), dtype=np.float32)
        with open(os.path.join(self.jma2001, 'take_off_angle')) as f:
            for line in f.readlines():
                ang, dep, dis = map(float, line.split())
                ang = np.deg2rad(ang)
                if dep not in x:
                    x.append(dep)
                if dis not in y:
                    y.append(dis)
                a[x.index(dep), y.index(dis)] = ang
                t[x.index(dep), y.index(dis)] = (
                    math.asin(math.sin(ang) / self.velocity(float(dep))
                              * self.velocity(0)))
        self.angle = RectBivariateSpline(x, y, a, kx=1, ky=1)
        self.take_off = RectBivariateSpline(x, y, t, kx=1, ky=1)


_tts = TravelTimeS()


def time(distance, depth=0, elevation=0):
    distance = np.asarray(distance, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    elevation = np.asarray(elevation, dtype=np.float32)
    return _tts.time(depth, distance, elevation)


def angle(distance, depth):
    return math.pi - _tts.angle.ev(depth, distance)


def velocity(depth):
    return _tts.velocity(depth)
