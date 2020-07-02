import calendar
import datetime
import ftplib
import numpy as np
import os
import xarray as xr


def get_n_periods_data(year, period_length):
    year.replace(month=1, day=1)
    y_str = str(year.year)
    if calendar.isleap(year.year):
        year_days = 366
    else:
        year_days = 365
    n_periods = year_days // period_length + 1
    for period in range(n_periods):

        data_file_path = 'current_netCDF/chunked_data/{0}_period{1}_length{2}.nc'.format(y_str, period, period_length)

        if os.path.exists(data_file_path):
            continue

        file_paths = []

        ftp = ftplib.FTP(host='eftp.ifremer.fr', user='gg1f3e8', passwd='xG3jZhT9')

        yday = period * period_length + 1

        for day in range(yday, yday + period_length):
            yday_str = '%03d' % day
            ftp.cwd('/data/globcurrent/v3.0/global_025_deg/total_hs/{0}/{1}'.format(y_str, yday_str))
            yday_path = 'current_netCDF/ftp_files/{0}/{1}/'.format(y_str, yday_str)
            if not os.path.exists(yday_path):
                os.mkdir(yday_path)
            files = [yday_path + filename for filename in ftp.nlst() if '0000' in filename]

            file_paths.extend(files)

        ftp.close()

        try:
            ds = xr.open_mfdataset(file_paths, parallel=True, combine='by_coords', chunks={'time': 1,
                                                                                           'lat': 720 // 3,
                                                                                           'lon': 1440 // 3})
        except FileNotFoundError:
            ftp = ftplib.FTP(host='eftp.ifremer.fr', user='gg1f3e8', passwd='xG3jZhT9')

            yday = period * period_length + 1

            for day in range(yday, yday + period_length):
                yday_str = '%03d' % day
                ftp.cwd('/data/globcurrent/v3.0/global_025_deg/total_hs/{0}/{1}'.format(y_str, yday_str))
                files = [filename for filename in ftp.nlst() if '0000' in filename]

                for filename in files:
                    local_filename = 'current_netCDF/ftp_files/{0}/{1}/'.format(y_str, yday_str) + filename
                    if os.path.exists(local_filename):
                        continue
                    with open(local_filename, 'wb') as f:
                        ftp.retrbinary('RETR %s' % filename, f.write)
            ftp.close()
            ds = xr.open_mfdataset(file_paths, parallel=True, combine='by_coords')

        ds.attrs = {}
        arr2d = np.ones((720, 1440)) * 1.94384
        ds['u_knot'] = arr2d * ds['eastward_eulerian_current_velocity']
        ds['v_knot'] = arr2d * ds['northward_eulerian_current_velocity']

        ds = ds.drop_vars(['eastward_eulerian_current_velocity',
                           'eastward_eulerian_current_velocity_error',
                           'northward_eulerian_current_velocity',
                           'northward_eulerian_current_velocity_error'])
        ds.chunk(chunks={'time': 1, 'lat': 720 // 3, 'lon': 1440 // 3})
        ds.to_netcdf(data_file_path, encoding={'u_knot': {'dtype': 'float32', 'scale_factor': 0.1, '_FillValue': None},
                                               'v_knot': {'dtype': 'float32', 'scale_factor': 0.1, '_FillValue': None}})
        ds.close()


year_in = datetime.datetime(2016, 1, 1)
period_length_in = 5

get_n_periods_data(year_in, period_length_in)
