#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:51:25 2024

@author: emily
"""
#import packages
from pyspedas import mms
from pytplot import tplot
from pytplot import get_data
import numpy as np
import matplotlib.pyplot as plt
#Probe 1  
mms.fpi(trange = ['2018-05-18/14:02:30', '2018-05-18/14:03'], probe =1, time_clip=True)
ion_velocity_data=get_data('mms1_dis_bulkv_gse_fast')
electron_velocity_data=get_data('mms1_des_bulkv_gse_fast')
number_density=get_data('mms1_des_numberdensity_fast')
ion_vel_times=ion_velocity_data.times
ion_vel_values=ion_velocity_data.y
el_vel_times=electron_velocity_data.times
el_vel_values=electron_velocity_data.y
num_den_times=number_density.times
num_den_values=number_density.y
q=1.6*10E-3
total_vel=el_vel_values-ion_vel_values
curr_dens_x=q*num_den_values*total_vel[:,0]
curr_dens_y=q*num_den_values*total_vel[:,1]
curr_dens_z=q*num_den_values*total_vel[:,2]
#generate plots
plt.plot(num_den_times, curr_dens_x, label="Jx")
plt.plot(num_den_times, curr_dens_y, label="Jy")
plt.plot(num_den_times, curr_dens_z, label="Jz")
plt.xlabel("Unix timestamp")
plt.ylabel("A/m^2*1e-8")
plt.title("Current MMS 1")
plt.legend()
plt.show()
#probe 2
mms.fpi(trange = ['2018-05-18/14:02:30', '2018-05-18/14:03'], probe =2, time_clip=True)
ion_velocity_data=get_data('mms2_dis_bulkv_gse_fast')
electron_velocity_data=get_data('mms2_des_bulkv_gse_fast')
number_density=get_data('mms2_des_numberdensity_fast')
ion_vel_times=ion_velocity_data.times
ion_vel_values=ion_velocity_data.y
el_vel_times=electron_velocity_data.times
el_vel_values=electron_velocity_data.y
num_den_times=number_density.times
num_den_values=number_density.y
q=1.6*10E-3
total_vel=el_vel_values-ion_vel_values
curr_dens_x=q*num_den_values*total_vel[:,0]
curr_dens_y=q*num_den_values*total_vel[:,1]
curr_dens_z=q*num_den_values*total_vel[:,2]
plt.plot(num_den_times, curr_dens_x, label="Jx")
plt.plot(num_den_times, curr_dens_y, label="Jy")
plt.plot(num_den_times, curr_dens_z, label="Jz")
plt.xlabel("Unix timestamp")
plt.ylabel("A/m^2* 1e-8")
plt.title("Current MMS 2")
plt.legend()
plt.show()

#probe 3
mms.fpi(trange = ['2018-05-18/14:02:30', '2018-05-18/14:03'], probe =3, time_clip=True)
ion_velocity_data=get_data('mms3_dis_bulkv_gse_fast')
electron_velocity_data=get_data('mms3_des_bulkv_gse_fast')
number_density=get_data('mms3_des_numberdensity_fast')
ion_vel_times=ion_velocity_data.times
ion_vel_values=ion_velocity_data.y
el_vel_times=electron_velocity_data.times
el_vel_values=electron_velocity_data.y
num_den_times=number_density.times
num_den_values=number_density.y
q=1.6*10E-3
total_vel=el_vel_values-ion_vel_values
curr_dens_x=q*num_den_values*total_vel[:,0]
curr_dens_y=q*num_den_values*total_vel[:,1]
curr_dens_z=q*num_den_values*total_vel[:,2]
plt.plot(num_den_times, curr_dens_x, label="Jx")
plt.plot(num_den_times, curr_dens_y, label="Jy")
plt.plot(num_den_times, curr_dens_z, label="Jz")
plt.xlabel("Unix timestamp")
plt.ylabel("A/m^2 *1e-8")
plt.title("Current MMS 3")
plt.legend()
plt.show()

# Probe 4
mms.fpi(trange = ['2018-05-18/14:02:30', '2018-05-18/14:03'], probe =4, time_clip=True)
ion_velocity_data=get_data('mms4_dis_bulkv_gse_fast')
electron_velocity_data=get_data('mms4_des_bulkv_gse_fast')
number_density=get_data('mms4_des_numberdensity_fast')
ion_vel_times=ion_velocity_data.times
ion_vel_values=ion_velocity_data.y
el_vel_times=electron_velocity_data.times
el_vel_values=electron_velocity_data.y
num_den_times=number_density.times
num_den_values=number_density.y
q=1.6*10E-3
total_vel=el_vel_values-ion_vel_values
curr_dens_x=q*num_den_values*total_vel[:,0]
curr_dens_y=q*num_den_values*total_vel[:,1]
curr_dens_z=q*num_den_values*total_vel[:,2]
plt.plot(num_den_times, curr_dens_x, label="Jx")
plt.plot(num_den_times, curr_dens_y, label="Jy")
plt.plot(num_den_times, curr_dens_z, label="Jz")
plt.xlabel("Unix timestamp")
plt.ylabel("A/m^2 *1e-8")
plt.title("Current MMS 4")
plt.legend()
plt.show()
