# File encoding: UTF-8

from django.contrib.gis            import geos
from django.contrib.gis.geoip      import GeoIP
from django.contrib.gis.measure    import Distance

from geopy.distance import distance as geopy_distance

from .common_django import *

def lat_lon(point):
    if not point:
        return None

    if isinstance(point, geos.Point):
        lon, lat = point
        return lat, lon

    if isinstance(point, (tuple, list)):
        lat, lon = point
        return float(lat), float(lon)

    s_lat, s_lon = point.strip().split(",")
    return float(s_lat), float(s_lon)

def parse_point(s):
    ll = lat_lon(s)
    return ll and geos.Point(ll[1], ll[0])
