#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:56:26 2023

"""
import matplotlib
import datetime as dt
#from pymms.data import fpi, epd, fgm
#from pymms.data import util
from matplotlib import pyplot as plt, dates as mdates
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt, dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from binned_avg import binned_avg
from turbulence_em import increment
from plot_2d import plot_2d
import pytplot
import pyspedas

t0 = dt.datetime(2018, 5, 18, 13, 45, 0, 000)
t1 = dt.datetime(2018, 5, 18, 14, 9, 59, 999)
t2 = dt.datetime(2018, 5, 18, 13, 50, 0, 000)
pyspedas.mms.eis(probe=3, datatype='extof', trange=['2018-5-18/13:50','2018-5-18/14:10'],time_clip=True)
pyspedas.mms.fgm(probe=3, trange=['2018-5-18/13:50','2018-5-18/14:10'], time_clip=True)
pyspedas.mms.fpi(probe=3, datatype='des-moms',trange=['2018-5-18/13:50','2018-5-18/14:10'],time_clip=True)
pyspedas.mms.fpi(probe=3, datatype='dis-moms',trange=['2018-5-18/13:50','2018-5-18/14:10'],time_clip=True)

data=pyspedas.mms.eis(probe=3, datatype='extof', trange=['2018-5-18/13:45','2018-5-18/14:10'],time_clip=True)
data2=pyspedas.mms.eis(probe=3, datatype='extof', trange=['2018-5-18/13:50','2018-5-18/14:10'],time_clip=True)

fgm_data = pyspedas.mms.fgm(probe=3, trange=['2018-5-18/13:45','2018-5-18/14:10'], time_clip=True)
fgm_data2 = pyspedas.mms.fgm(probe=3, trange=['2018-5-18/13:50','2018-5-18/14:10'], time_clip=True)

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
array7 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t5'].data[:, 6]
array6 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t5'].data[:, 5]
array5 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t5'].data[:, 4]
array4 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t5'].data[:, 3]
array3 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t5'].data[:, 2]
array2 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t5'].data[:, 1]
array1 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t5'].data[:, 0]

array14 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t4'].data[:, 6]
array13 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t4'].data[:, 5]
array12 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t4'].data[:, 4]
array11 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t4'].data[:, 3]
array10 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t4'].data[:, 2]
array9 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t4'].data[:, 1]
array8 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t4'].data[:, 0]

array21 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t3'].data[:, 6]
array20 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t3'].data[:, 5]
array19 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t3'].data[:, 4]
array18 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t3'].data[:, 3]
array17 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t3'].data[:, 2]
array16 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t3'].data[:, 1]
array15 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t3'].data[:, 0]

array28 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t2'].data[:, 6]
array27 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t2'].data[:, 5]
array26 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t2'].data[:, 4]
array25 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t2'].data[:, 3]
array24 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t2'].data[:, 2]
array23 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t2'].data[:, 1]
array22 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t2'].data[:, 0]

array35 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t1'].data[:, 6]
array34 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t1'].data[:, 5]
array33 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t1'].data[:, 4]
array32 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t1'].data[:, 3]
array31 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t1'].data[:, 2]
array30 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t1'].data[:, 1]
array29 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t1'].data[:, 0]

array42 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t0'].data[:, 6]
array41 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t0'].data[:, 5]
array40 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t0'].data[:, 4]
array39 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t0'].data[:, 3]
array38 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t0'].data[:, 2]
array37 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t0'].data[:, 1]
array36 = data2['mms3_epd_eis_srvy_l2_extof_proton_P4_flux_t0'].data[:, 0]

sum1=np.add(array36,array29)
sum2=np.add(sum1,array22)
sum3=np.add(sum2,array15)
sum4=np.add(sum3,array8)
sum5=np.add(sum4,array1)
avg0=sum5/6

sum6=np.add(array37,array30)
sum7=np.add(sum6,array23)
sum8=np.add(sum7,array16)
sum9=np.add(sum8,array9)
sum10=np.add(sum9,array2)
avg1=sum10/6

sum11=np.add(array38,array31)
sum12=np.add(sum11,array24)
sum13=np.add(sum12,array17)
sum14=np.add(sum13,array10)
sum15=np.add(sum14,array3)
avg2=sum15/6

sum16=np.add(array39,array32)
sum17=np.add(sum16,array25)
sum18=np.add(sum17,array18)
sum19=np.add(sum18,array11)
sum20=np.add(sum19,array4)
avg3=sum20/6

sum21=np.add(array40,array33)
sum22=np.add(sum21,array26)
sum23=np.add(sum22,array19)
sum24=np.add(sum23,array12)
sum25=np.add(sum24,array5)
avg4=sum5/6

sum26=np.add(array41,array34)
sum27=np.add(sum26,array27)
sum28=np.add(sum27,array20)
sum29=np.add(sum28,array13)
sum30=np.add(sum29,array6)


def func1(PVI, Q, mailbox):
    lists = []
    cutoff = np.where(PVI >= Q)[0]
    for i in range(cutoff.shape[0]):
        start_ind = cutoff[i]-20
        fin_ind = cutoff[i]+20
        if start_ind < 0:
            continue
        if fin_ind > mailbox.size-1:
            continue
        my_variables = mailbox[start_ind:fin_ind]
        lists.append(my_variables)
    li = list(zip(*lists))
    summation = list(map(sum, li))
    average = list(map(lambda x: x/len(lists), summation))
    return np.asarray(average)


average_PVI0 = func1(PVI, 0, avg0)
average_PVI1 = func1(PVI, 1, avg0)
average_PVI2 = func1(PVI, 2, avg0)
average_PVI3 = func1(PVI, 3, avg0)
average_PVI4 = func1(PVI, 4, avg0)
average_PVI5 = func1(PVI, 5, avg0)

def func2(PVI, Q, mailbox):
    lists = []
    cutoff = np.where(PVI >= Q)[0]
    for i in range(cutoff.shape[0]):
        start_ind = cutoff[i]-60
        fin_ind = cutoff[i]+60
        if start_ind < 0:
            continue
        if fin_ind > mailbox.size-1:
            continue
        my_variables = mailbox[start_ind:fin_ind]
        lists.append(my_variables)
    li = list(zip(*lists))
    summation = list(map(sum, li))
    average = list(map(lambda x: x/len(lists), summation))
    return np.asarray(average)


average_PVI6 = func2(PVI, 0, avg0)
average_PVI7 = func2(PVI, 1, avg0)
average_PVI8 = func2(PVI, 2, avg0)
average_PVI9 = func2(PVI, 3, avg0)
average_PVI10 = func2(PVI, 4, avg0)
average_PVI11 = func2(PVI, 5, avg0)

time = data2['Epoch'].data


sampletime = np.linspace(-50, 50, 40)
plt.rcParams["figure.figsize"] = (15, 6)
plt.plot(sampletime, average_PVI0, label="PVI>0")
plt.plot(sampletime, average_PVI1, label="PVI>1")
plt.plot(sampletime, average_PVI2, label="PVI>2")
plt.plot(sampletime, average_PVI3, label="PVI>3")
plt.xlabel("Time from event(s)")
plt.ylabel("Flux(1/(cm^2 s sr keV)")
plt.title("PVI extof")
plt.legend()
plt.show()

sampletime2 = np.linspace(-60, 60, 120)
plt.rcParams["figure.figsize"] = (15, 6)
plt.plot(sampletime2, average_PVI6, label="PVI>0")
plt.plot(sampletime2, average_PVI7, label="PVI>1")
plt.plot(sampletime2, average_PVI8, label="PVI>2")
plt.plot(sampletime2, average_PVI9, label="PVI>3")
plt.plot(sampletime2, average_PVI10, label="PVI>4")
#plt.plot(sampletime2, average_PVI11, label="PVI>5")
plt.xlabel("Time from event(s)")
plt.ylabel("Flux(1/(cm^2 s sr keV)")
plt.title("PVI extof")
plt.legend()
plt.show()

pytplot.store_data("PVI", data={'x':data2['Epoch'][:-1].data,'y':PVI})
time, data=pytplot.get_data("PVI")

from pyspedas import mms_eis_pad
mms_eis_pad(probe=3, datatype='extof')

pytplot.tplot(['PVI','mms3_epd_eis_srvy_l2_phxtof_48-79keV_proton_flux_omni_pad_spin','mms3_epd_eis_srvy_l2_extof_43-951keV_proton_flux_omni_pad_spin','mms3_epd_eis_srvy_l2_extof_proton_flux_omni','mms3_epd_eis_srvy_l2_phxtof_proton_flux_omni','mms3_dis_energyspectr_omni_fast','mms3_des_energyspectr_omni_fast','mms3_dis_numberdensity_fast','mms3_dis_bulkv_gse_fast','mms3_fgm_b_gse_srvy_l2',],xsize=27, ysize=50)
