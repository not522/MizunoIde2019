import math

import numpy as np

import const
from distance import distance
import optimize
import travel


class GridSearch(object):
    def __init__(self, station):
        self.lamin = min(station, key=lambda x: x[0])[0]
        self.lamax = max(station, key=lambda x: x[0])[0]
        self.lomin = min(station, key=lambda x: x[1])[1]
        self.lomax = max(station, key=lambda x: x[1])[1]

        self.lamin = int(math.floor((self.lamin - 2) * 5))
        self.lamax = int(math.ceil((self.lamax + 2) * 5))
        self.lomin = int(math.floor((self.lomin - 2) * 5))
        self.lomax = int(math.ceil((self.lomax + 2) * 5))

        self.grid_tp = {}
        for la in range(self.lamin, self.lamax + 1):
            for lo in range(self.lomin, self.lomax + 1):
                dis = distance(
                    la / 5.0, lo / 5.0, station[:, 0], station[:, 1])
                dep = [const.gridsearch_depth] * len(station)
                elv = station[:, 2]
                if dis.min() <= 2 * const.max_dist_station_source:
                    self.grid_tp[(la, lo)] = travel.time(dis, dep, elv)

    def grid(self, sac, cc):
        grid = np.ones((1800, 1800), dtype=np.float32)
        components = list(set(cc.i_c) | set(cc.j_c))

        clat = []
        clon = []
        for i in components:
            clat.append(sac[i].x[0])
            clon.append(sac[i].x[1])
        clat = np.asarray(clat, dtype=np.float64)
        clon = np.asarray(clon, dtype=np.float64)

        weight = np.zeros(len(sac), dtype=np.float32)
        pos = []
        for la in range(self.lamin, self.lamax + 1):
            for lo in range(self.lomin, self.lomax + 1):
                dis = distance(la / 5.0, lo / 5.0, clat, clon)
                if dis.min() <= 2 * const.max_dist_station_source:
                    pos.append((la, lo))
        ind = np.arange(cc.i_s.size)
        for la, lo in pos:
            dis = distance(la / 5.0, lo / 5.0, clat, clon)
            weight[components] = 1 / (dis ** 2 + const.gridsearch_depth ** 2)
            grid[la, lo] = optimize.cross_correlation(
                cc, ind, self.grid_tp[(la, lo)], weight)
        return grid, pos

    def __call__(self, sac, cc):
        grid, pos = self.grid(sac, cc)
        res = []
        for la, lo in pos:
            if grid[la, lo] == grid[la-2:la+3, lo-2:lo+3].max():
                res.append((la / 5.0, lo / 5.0))
        return res
