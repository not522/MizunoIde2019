import math

import numpy as np
import scipy.optimize

import const
from distance import azimuth
from distance import distance
import travel


def cross_correlation(cc, ind, tp, weight):
    i_s = cc.i_s[ind]
    j_s = cc.j_s[ind]
    i_c = cc.i_c[ind]
    j_c = cc.j_c[ind]
    w = weight[i_c] * weight[j_c]
    return (w * cc(ind, tp[j_s] - tp[i_s])).sum() / w.sum()


def _get_tp(pos, station, cc, ind, station_i, dis=None):
    stations = station[station_i, :]
    if dis is None:
        dis = distance(pos[0], pos[1], stations[:, 0], stations[:, 1])
    dep = [pos[2]] * len(stations)
    elv = stations[:, 2]
    tp = np.zeros(len(station), dtype=np.float32)
    tp[station_i] = travel.time(dis, dep, elv)
    return tp


def _cross_correlation_obj(x, init_pos, cc, ind, station, weight, station_i):
    pos = init_pos.copy()
    pos[0] += x[0] / 40000 * 360
    pos[1] += x[1] / 40000 * 360 / math.cos(np.deg2rad(pos[0]))
    pos[2] += x[2]
    return -cross_correlation(
        cc, ind, _get_tp(pos, station, cc, ind, station_i), weight)


def _cross_correlation_grad(x, init_pos, cc, ind, station, weight, station_i):
    pos = init_pos.copy()
    pos[0] += x[0] / 40000 * 360
    pos[1] += x[1] / 40000 * 360 / math.cos(np.deg2rad(pos[0]))
    pos[2] += x[2]

    stations = station[station_i, :]
    dis = distance(pos[0], pos[1], stations[:, 0], stations[:, 1])
    tp = _get_tp(pos, station, cc, ind, station_i, dis)

    dt = np.zeros((len(station), 3), dtype=np.float32)
    az = azimuth(pos[0], pos[1], stations[:, 0], stations[:, 1])
    ang = travel.angle(dis, pos[2])
    dt[station_i, 0] = np.sin(ang) * np.cos(az)
    dt[station_i, 1] = np.sin(ang) * np.sin(az)
    dt[station_i, 2] = np.cos(ang)
    dt /= travel.velocity(pos[2])

    i_s = cc.i_s[ind]
    j_s = cc.j_s[ind]
    i_c = cc.i_c[ind]
    j_c = cc.j_c[ind]
    g = cc.grad(ind, tp[j_s] - tp[i_s])[:, None]
    w = (weight[i_c] * weight[j_c])[:, None]
    grad = g * w * (dt[j_s, :] - dt[i_s, :])
    return grad.sum(axis=0) / w.sum()


def _optimize(x, cc, ind, station, weight):
    station_i = np.asarray(list(set(cc.i_s[ind]) | set(cc.j_s[ind])))
    args = (x, cc, ind, station, weight, station_i)
    res = scipy.optimize.minimize(
        _cross_correlation_obj, [0, 0, 0], method='SLSQP',
        jac=_cross_correlation_grad, args=args,
        bounds=((-100, 100), (-100, 100),
                (-x[2], const.max_source_depth_search-x[2])),
        options={'ftol': 1e-9})
    pos = x.copy()
    pos[0] += res.x[0] / 40000 * 360
    pos[1] += res.x[1] / 40000 * 360 / math.cos(np.deg2rad(pos[0]))
    pos[2] += res.x[2]
    res.x = pos
    return res


class Optimizer(object):
    def __init__(self, x, cc, ind, station, envelope):
        self.x = x
        self.station = station
        self.envelope = envelope
        self.cc = cc
        self.ind = np.asarray(ind, dtype=np.int32)

        self.n_c = max(self.cc.i_c[ind].max(), self.cc.j_c[ind].max()) + 1

        self.components = \
            set(zip(self.cc.i_s[self.ind], self.cc.i_c[self.ind])) | \
            set(zip(self.cc.j_s[self.ind], self.cc.j_c[self.ind]))

        self.components = np.asarray(list(self.components), dtype=np.int32)
        self.weight = np.zeros(self.n_c, dtype=np.float32)
        i_s = self.components[:, 0]
        i_c = self.components[:, 1]
        dis = distance(x[0], x[1], station[i_s, 0], station[i_s, 1])
        self.weight[i_c] = 1 / (dis * dis + x[2] * x[2])

    def get_tp(self):
        station_i = list(set(self.components[:, 0]))
        stations = self.station[station_i]
        dis = distance(self.x[0], self.x[1], stations[:, 0], stations[:, 1])
        dep = [self.x[2]] * len(stations)
        elv = stations[:, 2]
        tp = np.full(len(self.station), np.inf, dtype=np.float32)
        tp[station_i] = travel.time(dis, dep, elv)
        return tp, min(tp), min(dis)

    def _shift_waveform(self, waveform, tp, tmin):
        n = waveform.shape[1]
        factor2, it = np.modf(tp[self.components[:, 0]] - tmin)
        factor1 = 1 - factor2
        it = it.astype(np.int32)
        cols = np.arange(n)[None, :] + it[:, None]
        cols[cols >= n] = -2
        w1 = waveform[self.components[:, 1:2], cols]
        w1[cols < 0] = 0
        cols += 1
        cols[cols == n] = -2
        w2 = waveform[self.components[:, 1:2], cols]
        w2[cols < 0] = 0
        return w1, w2, factor1, factor2

    def get_template(self, waveform, tp, tmin):
        cnt = np.zeros(self.n_c, dtype=np.int32)
        for i in self.cc.i_c[self.ind]:
            cnt[i] += 1
        for i in self.cc.j_c[self.ind]:
            cnt[i] += 1
        waveform = np.asarray(waveform, dtype=np.float32)
        wave1, wave2, factor1, factor2 = self._shift_waveform(
            waveform, tp, tmin)
        weight = cnt[self.components[:, 1]] * \
            self.weight[self.components[:, 1]]
        wave = factor1[:, None] * wave1 + factor2[:, None] * wave2
        template = np.sum(weight[:, None] * wave, axis=0)
        return template / weight.sum()

    def _update_weight(self, template, tp, tmin):
        env1, env2, factor1, factor2 = self._shift_waveform(
            self.envelope, tp, tmin)
        diff1 = factor1 * np.sum((template - env1) ** 2, axis=1)
        diff2 = factor2 * np.sum((template - env2) ** 2, axis=1)
        self.weight[self.components[:, 1]] = 1 / (diff1 + diff2)

    def _cc_template(self, template, tp, tmin):
        cc_t = np.zeros(len(self.envelope), dtype=np.float32)
        env1, env2, factor1, factor2 = self._shift_waveform(
            self.envelope, tp, tmin)
        cc_t[self.components[:, 1]] += factor1 * (template * env1).sum(axis=1)
        cc_t[self.components[:, 1]] += factor2 * (template * env2).sum(axis=1)
        return cc_t

    def __call__(self):
        while True:
            opt_result = _optimize(
                self.x, self.cc, self.ind, self.station, self.weight)
            self.x = opt_result.x
            if not opt_result.success or self.x[2] < 0.1 or \
               const.max_source_depth_search - 0.1 < self.x[2]:
                return None

            tp, tmin, dmin = self.get_tp()
            if dmin > const.max_dist_station_source:
                return None
            template = self.get_template(self.envelope, tp, tmin)
            cc_tw = self._cc_template(template, tp, tmin)

            cc = self.cc(self.ind,
                         tp[self.cc.j_s[self.ind]] - tp[self.cc.i_s[self.ind]])
            next_ind = cc > const.c_lim
            next_ind *= cc_tw[self.cc.i_c[self.ind]] > const.c_lim_t
            next_ind *= cc_tw[self.cc.j_c[self.ind]] > const.c_lim_t

            if next_ind.all():
                if const.min_source_depth < self.x[2] < const.max_source_depth:
                    return opt_result.fun
                return None

            self.ind = self.ind[next_ind].copy()
            if self.ind.size < const.min_pair_num:
                return None

            self._update_weight(template, tp, tmin)
