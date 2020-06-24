def distance(lon1, lat1, lon2, lat2, gc):
    """
        Get great circle distance from the longitude-latitude
        pair ``lon1,lat1`` to ``lon2,lat2``
        Returns great circle distance
        """
    az12, az21, dist = gc.inv(lon1, lat1, lon2, lat2)

    return dist * 0.000539957


def points(lon1, lat1, lon2, lat2, dist, gc, del_s):
    """
    Get great circle points from the longitude-latitude
    pair ``lon1,lat1`` to ``lon2,lat2``
    .. tabularcolumns:: |l|L|
    ==============   =======================================================
    Keyword          Description
    ==============   =======================================================
    del_s            points on great circle computed every del_s kilometers
                     (default 100).
    ==============   =======================================================
    Returns two lists of lons, lats points on great circle
    """
    n_points = int((dist + 0.5 * 1000. * del_s) / (1000. * del_s))
    lon_lats = gc.npts(lon1, lat1, lon2, lat2, n_points)
    lons, lats = [lon1], [lat1]
    for lon, lat in lon_lats:
        lons.append(lon)
        lats.append(lat)
    lons.append(lon2)
    lats.append(lat2)

    return [lons, lats]
