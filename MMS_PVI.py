#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:07:21 2024

@author: emily
"""
import matplotlib
import datetime as dt
from pymms.data import fpi, epd, fgm
from pymms.data import util
from binned_avg import binned_avg
from turbulence_em import increment
import numpy as np
import xarray as xr

t0 = dt.datetime(2018, 5, 18, 13, 45, 0, 000)
t1 = dt.datetime(2018, 5, 18, 14, 9, 59, 999)
t2 = dt.datetime(2018, 5, 18, 13, 50, 0, 000)

def cdf_varnames(cdf, data_vars=None):
    print(f"Type of cdf inside cdf_varnames: {type(cdf)}")
    varnames = cdf.cdf_info().zVariables
    return varnames

data = epd.load_data(sc='mms3', optdesc='phxtof', start_date=t0, end_date=t1)
data2 = epd.load_data(sc='mms3', optdesc='phxtof', start_date=t2, end_date=t1)
fgm_data = fgm.load_data(sc='mms3', mode='srvy', start_date=t0, end_date=t1)
fgm_data2 = fgm.load_data(sc='mms3', mode='srvy', start_date=t2, end_date=t1)

mag_field_x = binned_avg(
    fgm_data2['time'].data, fgm_data2['B_GSE'][:, 0].data, data2['Epoch'].data)
mag_field_y = binned_avg(
    fgm_data2['time'].data, fgm_data2['B_GSE'][:, 1].data, data2['Epoch'].data)
mag_field_z = binned_avg(
    fgm_data2['time'].data, fgm_data2['B_GSE'][:, 2].data, data2['Epoch'].data)
wibblywobblytimeywimey = fgm_data2['time']  # [:-2400]
inc_b = increment(fgm_data['B_GSE'].data, 4800)
denom = np.sqrt(np.average(inc_b[:, 3]**2))
F = inc_b[:, 1]/denom
avg = binned_avg(wibblywobblytimeywimey.data, F.data, data2['Epoch'].data)
PVI = abs(avg[0:])
