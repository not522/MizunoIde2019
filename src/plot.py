import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.basemap import Basemap
except ImportError:
    pass


def basemap(ax, minlat, maxlat, minlon, maxlon, int_lat=1, int_lon=1):
    m = Basemap(projection='merc', llcrnrlat=minlat - 0.0001,
                urcrnrlat=maxlat + 0.0001, llcrnrlon=minlon - 0.0001,
                urcrnrlon=maxlon + 0.0001, resolution='l')
    m.drawcoastlines(linewidth=0.5, color='k', zorder=1)
    m.drawparallels(np.arange(-90, 90, int_lat),
                    labels=[True, False, False, False], zorder=1)
    m.drawmeridians(np.arange(0, 360, int_lon),
                    labels=[False, False, False, True], zorder=1)
    return m


def plot(output, sources, info, grid, grid_search, stations, cc):
    fig = plt.figure(figsize=(12, 8))

    lamin_i = grid_search.lamin + 5
    lamax_i = grid_search.lamax - 5
    lomin_i = grid_search.lomin + 5
    lomax_i = grid_search.lomax - 5
    lamin = lamin_i / 5.0
    lamax = lamax_i / 5.0
    lomin = lomin_i / 5.0
    lomax = lomax_i / 5.0

    heatmap = grid[lamin_i:lamax_i+2, lomin_i:lomax_i+2]
    interval = 1 if lamax - lamin > 2 else 0.2
    m = basemap(fig.gca().axes, lamin-0.1, lamax+0.1, lomin-0.1, lomax+0.1,
                interval, interval)
    x = np.linspace(0, m.urcrnrx, heatmap.shape[1])
    y = np.linspace(0, m.urcrnry, heatmap.shape[0])
    x, y = np.meshgrid(x, y)
    im = m.pcolormesh(x, y, heatmap, cmap='Reds', vmin=0.3, vmax=0.6,
                      zorder=-1)

    used = np.zeros(len(stations), dtype=np.int32)
    used[cc.i_s] = 1
    used[cc.j_s] = 1
    for optimizer in info:
        used[optimizer.cc.i_s[optimizer.ind]] = 2
        used[optimizer.cc.j_s[optimizer.ind]] = 2
    for x, u in zip(stations, used):
        lon, lat = m(x[1], x[0])
        if u == 2:
            color = 'r'
        elif u == 1:
            color = 'y'
        else:
            color = 'w'
        m.plot(lon, lat, '^', zorder=2, markerfacecolor=color, color='k',
               markersize=8)

    for source in sources:
        lon, lat = m(source['longitude'], source['latitude'])
        m.plot([lon], [lat], '*', ms=20, markerfacecolor='#ffff00',
               color='k', zorder=3)

    axins1 = inset_axes(plt.gca().axes, width='15%', height='2%', loc=4,
                        borderpad=2)
    cbar = plt.colorbar(im, cax=axins1, orientation='horizontal',
                        ticks=[0.3, 0.4, 0.5, 0.6])
    cbar.ax.set_xlabel('ACC')
    cbar.ax.xaxis.set_label_position('top')
    axins1.xaxis.set_ticks_position("bottom")

    plt.savefig(output)
