import json
import math
import sys

from tqdm import tqdm

import numpy as np
import scipy.signal

import catalog
import const
from distance import distance
from grid_search import GridSearch
from optimize import Optimizer
import plot
from sac import CrossCorrelations
from sac import Sac
import travel
import union_find


class Solver(object):
    def solve_window(self, sac, it, pairs, pairs_mask, component_to_station,
                     grid_search, station):
        t = it * const.timewindow_step
        envelope = []
        for s in sac:
            x = s.sac.data[t:t+const.timewindow_length].copy()
            x[x < 0] = 0
            x = np.sqrt(x)
            x -= np.mean(x)
            x /= np.sum(x ** 2) ** 0.5 + 1e-20
            envelope.append(x)
        envelope = np.asarray(envelope, dtype=np.float32)

        c = scipy.signal.fftconvolve(
            envelope[pairs[:, 1], :], envelope[pairs[:, 0], ::-1], axes=1)
        i = np.argwhere(np.max(c * pairs_mask, axis=1) > const.c_lim).flatten()
        v = c[i, :]
        i_s = component_to_station[pairs[i, 0]]
        j_s = component_to_station[pairs[i, 1]]
        i_c = pairs[i, 0]
        j_c = pairs[i, 1]

        self.cc = CrossCorrelations(v, i_s, j_s, i_c, j_c)
        if len(v) < const.min_pair_num:
            return [], []

        init_pos = grid_search(sac, self.cc)
        res = []
        for pos in init_pos:
            x = np.asarray([pos[0], pos[1], const.gridsearch_depth],
                           dtype=np.float32)
            ind = np.arange(self.cc.i_s.size)
            optimizer = Optimizer(x, self.cc, ind, station, envelope)
            opt_result = optimizer()
            if opt_result is not None:
                res.append((opt_result, x, optimizer))

        uf = union_find.UnionFind(len(res))
        result = []
        info = []
        for i in range(len(res)):
            xi = res[i][1]
            for j in range(i + 1, len(res)):
                xj = res[j][1]
                if abs(xi[0] - xj[0]) < const.merge_source_dist and \
                   abs(xi[1] - xj[1]) < const.merge_source_dist:
                    uf.unite(i, j)

        for group in uf.get_groups():
            source = res[max(group, key=lambda x: res[x][2].ind.size)]
            correlation, init_x, optimizer = source
            x = optimizer.x
            n_cc = optimizer.ind.size
            tp, tmin, dmin = optimizer.get_tp()
            if dmin > const.max_dist_station_source:
                continue
            waveform = []
            for s in sac:
                u = s.sac.data[t:t+const.timewindow_length].copy()
                dis = distance(x[0], x[1], s.x[0], s.x[1])
                u[:] *= dis * dis + x[2] * x[2]
                waveform.append(u)
            template = optimizer.get_template(waveform, tp, tmin)
            mi = np.argmax(template)
            dur = 0
            es = 0
            for itr in range(mi, -1, -1):
                if template[itr] < template[mi] / 4:
                    break
                dur += 1
                es += template[itr]
            for itr in range(mi + 1, len(template)):
                if template[itr] < template[mi] / 4:
                    break
                dur += 1
                es += template[itr]
            es *= 3 * 4 * math.pi * const.rho * const.beta * 1e6
            if dur <= const.min_duration:
                continue
            me = (math.log10(es) - 4.4) / 1.5
            tmax = np.argmax(template) - tmin + t
            x_bootstrap = []
            for _ in range(const.bootstrap_itr):
                ind = np.random.randint(0, self.cc.i_s.size, self.cc.i_s.size)
                _optimizer = Optimizer(init_x, self.cc, ind, station, envelope)
                opt_result = _optimizer()
                if opt_result is not None:
                    x_bootstrap.append(_optimizer.x - x)
                else:
                    x_bootstrap.append([np.inf, np.inf, np.inf])

            x_bootstrap = np.abs(np.asarray(x_bootstrap))
            half = const.bootstrap_itr // 2
            err = [0, 0, 0]
            err[0] = np.sort(x_bootstrap[:, 0])[half]
            err[1] = np.sort(x_bootstrap[:, 1])[half]
            err[2] = np.sort(x_bootstrap[:, 2])[half]

            err[0] *= 40000 / 360
            err[1] *= 40000 / 360 * math.cos(np.deg2rad(x[0]))
            if max(err) > const.max_error:
                continue
            result.append(catalog.make_catalog(
                it, n_cc, tmax, -correlation, dur, me, x, err))
            info.append(optimizer)
        return result, info

    def __call__(self, files, output, time_window, mode, quiet, debug):
        sac = []
        for file_name in files:
            try:
                s = Sac(file_name)
                sac.append(s)
                if debug:
                    print(file_name, s.x)
            except RuntimeError as e:
                if not quiet:
                    print(e)
                pass

        station = []
        for s in sac:
            if s.x not in station:
                station.append(s.x)
        component_to_station = []
        for s in sac:
            component_to_station.append(station.index(s.x))
        station = np.asarray(station, dtype=np.float64)
        component_to_station = np.asarray(component_to_station, dtype=np.int32)

        pairs = []
        pairs_mask = []
        for i in range(len(sac)):
            for j in range(i + 1, len(sac)):
                if component_to_station[i] == component_to_station[j]:
                    continue
                dis = distance(sac[i].x[0], sac[i].x[1], sac[j].x[0],
                               sac[j].x[1])
                if dis < const.min_dist_station_pair or \
                   dis > const.max_dist_station_pair:
                    continue
                diff = int(math.ceil(travel.time(dis)))
                pairs.append([i, j])
                length = const.timewindow_length * 2 - 1
                mask = np.zeros(length, dtype=np.bool_)
                mask[length//2-diff:length//2+diff+1] = True
                pairs_mask.append(mask)
        pairs = np.asarray(pairs, dtype=np.int32)
        pairs_mask = np.asarray(pairs_mask, dtype=np.bool_)

        if not quiet:
            sys.stderr.write('{} sac files\n'.format(len(sac)))
            sys.stderr.write('{} potential pairs'.format(len(pairs)))
        if len(pairs) == 0:
            return

        grid_search = GridSearch(station)

        time_window_start = 0
        time_window_end = 86400 // const.timewindow_step - 1
        if mode is None:
            if time_window is None:
                catalog = []
                for it in tqdm(range(time_window_start, time_window_end),
                               disable=quiet):
                    catalog += self.solve_window(
                        sac, it, pairs, pairs_mask, component_to_station,
                        grid_search, station)[0]
            else:
                catalog = self.solve_window(
                    sac, time_window, pairs, pairs_mask,
                    component_to_station, grid_search, station)[0]
            with open(output + '.json', 'w') as f:
                json.dump(catalog, f, indent=4, sort_keys=True)
        elif mode == "plot":
            if time_window is None:
                for it in tqdm(range(time_window_start, time_window_end),
                               disable=quiet):
                    sources, info = self.solve_window(
                        sac, it, pairs, pairs_mask, component_to_station,
                        grid_search, station)
                    grid = grid_search.grid(sac, self.cc)[0]
                    grid[grid == 1] = 0
                    plot.plot('{}_{:03}.png'.format(output, it), sources,
                              info, grid, grid_search, station, self.cc)
            else:
                sources, info = self.solve_window(
                    sac, time_window, pairs, pairs_mask, component_to_station,
                    grid_search, station)
                grid = grid_search.grid(sac, self.cc)[0]
                grid[grid == 1] = 0
                plot.plot('{}_{:03}.png'.format(output, time_window), sources,
                          info, grid, grid_search, station, self.cc)
