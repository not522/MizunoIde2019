def make_catalog(time_window, num_station, time, cc, dur, me, x, err):
    return {
        'time window': time_window,
        'number of pairs': num_station,
        'time': time,
        'cross correlation': cc,
        'duration': dur,
        'me': me,
        'latitude': float(x[0]),
        'longitude': float(x[1]),
        'depth': float(x[2]),
        'latitude error': float(err[0]),
        'longitude error': float(err[1]),
        'depth error': float(err[2])
    }
