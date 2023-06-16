import math

import numpy as np


def distance(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    slat1 = math.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = math.sqrt(1 - slat1 * slat1)
    clat2 = np.sqrt(1 - slat2 * slat2)
    slat = slat1 * slat2
    clat = clat1 * clat2
    return np.arccos(np.clip(slat + clat * np.cos(lon1 - lon2), -1, 1)) * \
        20000 / math.pi


def azimuth(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlon = lon1 - lon2
    slat1 = math.sin(lat1)
    clat1 = math.cos(lat1)
    sdlon = np.sin(dlon)
    cdlon = np.cos(dlon)
    return -np.arctan2(sdlon, clat1 * np.tan(lat2) - slat1 * cdlon)
