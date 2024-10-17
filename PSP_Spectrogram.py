import os
import sys
import csv
import glob
import spacepy
from spacepy import pycdf
import math
import datetime
import numpy as np
import pandas as pd
import time
import bisect
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import itertools
from matplotlib.dates import DateFormatter, AutoDateLocator
import matplotlib.dates as mdates
##############################################################################################
def ebins(varEbin):
    energybin = cdf[varEbin][...]
    energybin = energybin[0, 0, :]
    energybin = np.where(energybin == -1.000000e+31, np.nan, energybin)
    energybindf = pd.DataFrame(data=energybin, columns=['HENERGY'])
    return energybindf

def readcdfvar(varcdf):
    cdfvar = cdf[varcdf][...]
    return cdfvar

def getintegralflux(Hfluxdf, indxE, Hfluxdims):
    intflux = []
    for i in range(0, Hfluxdims[0]):
        if indxE != 0:
            intflux.append(Hfluxdf.iloc[i, indxE:Hfluxdims[1]].sum(skipna=True))
        else:
            intflux.append(np.nan)
    intfluxdf = pd.DataFrame(data=intflux, columns=['integralFlux'])
    intfluxdf.replace(0., np.nan, inplace=True)
    return intfluxdf

#######################################################################################
t1 = dt.datetime(2022, 5, 14, 9, 0, 0, 0)
t2 = dt.datetime(2022, 5, 14, 11, 59, 59, 999)
rootdir = ''
datadir = '/home/emily/Documents/PSP/'
infiles = [datadir + 'psp_isois-epilo_l2-ic_20220514_v15.cdf']
cdf2 = pycdf.CDF('/home/emily/Documents/PSP/psp_fld_l2_mag_RTN_2022051406_v00.cdf')
infiles2 = [datadir + 'psp_swp_spi_sf00_L2_8Dx32Ex8A_20220514_v04.cdf']

date_format = DateFormatter('%Y-%m-%d %H:%M')
locator = AutoDateLocator()

for kk in range(0, len(infiles)):
    fig, ax = plt.subplots(6, 1, figsize=(14, 15))
    fig.tight_layout(pad=2)
    
    infile = infiles[kk]
    print(kk, infile)
    
    sdate = infile.split('_', 6)[4]
    yrbeg = str(sdate[0:4])
    mnbeg = str(sdate[4:6])
    dybeg = str(sdate[6:8])
    print(yrbeg, mnbeg, dybeg)
    
    cdf = pycdf.CDF(infile)
    
    varEnergybins = 'H_ChanP_Energy'
    varEpoch = 'Epoch_ChanP'
    pitchAngleA = 'PA_ChanP' # H Pitch Angles
    pitchAngleB = 'PA_ChanC'# He Pitch Angles
    
    hfluxepoch = cdf[varEpoch][...]
    ndimsepoch = hfluxepoch.shape[0]
    hfluxepoch = hfluxepoch.reshape(ndimsepoch)
    hfluxepochdf = pd.DataFrame(data=hfluxepoch, columns=['datetime'])
    
    energybinsdf = ebins(varEnergybins)
    ndimsEbins = energybinsdf.shape[0]
    minEbin = np.nanmin(energybinsdf)
    maxEbin = np.nanmax(energybinsdf)
    energybins = np.array(energybinsdf.HENERGY)
    
    Hfluxdata = cdf['H_Flux_ChanP'][...]
    ndimsHflux = Hfluxdata.shape
    Hfluxdata = np.where(Hfluxdata == -1.000000e+31, np.nan, Hfluxdata)
    Hfluxdata = np.where(Hfluxdata == 0.0, np.nan, Hfluxdata)
    
    Hefluxdata = cdf['He4_Flux_ChanC'][...]
    ndimsHeflux = Hefluxdata.shape
    Hefluxdata = np.where(Hefluxdata == -1.000000e+31, np.nan, Hefluxdata)
    Hefluxdata = np.where(Hefluxdata == 0.0, np.nan, Hefluxdata)
    
    lookdir = [22, 25, 34, 35, 36, 37, 44, 46]
    lookdirdel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,26,27,28,29,30,31,32,33,38,39,40,41,42,43,45,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79]
    
    Hfluxdata = np.delete(Hfluxdata, [lookdirdel], axis=1)
    print('DIMS: ', Hfluxdata.shape)
    Hefluxdata = np.delete(Hefluxdata, [lookdirdel], axis=1)
    print('DIMS: ', Hefluxdata.shape)
    
    HfluxSum = np.nansum(Hfluxdata, axis=1)
    HfluxSumdf = pd.DataFrame(data = HfluxSum)
    HefluxSum = np.nansum(Hefluxdata, axis=1)
    HefluxSumdf = pd.DataFrame(data = HefluxSum)
    
    Hfluxmean = np.nanmean(Hfluxdata, axis=1)
    Hfluxdf = pd.DataFrame(data = Hfluxmean, columns=[energybins])
    Hefluxmean = np.nanmean(Hefluxdata, axis=1)
    Hefluxdf = pd.DataFrame(data = Hefluxmean, columns=[energybins])
    
    pitchAngleAdata = readcdfvar(pitchAngleA)
    pitchAngleAdata = np.where(pitchAngleAdata == -1.000000e+31, np.nan, pitchAngleAdata)
    print('PITCH ANGLE: ', pitchAngleAdata.shape)
    pitchAngleAdata = np.delete(pitchAngleAdata, [lookdirdel], axis=1)
    print('PITCH ANGLE: ', pitchAngleAdata.shape)
    
    pitchAngleBdata = readcdfvar(pitchAngleB)
    pitchAngleBdata = np.where(pitchAngleBdata == -1.000000e+31, np.nan, pitchAngleBdata)
    print('PITCH ANGLE: ', pitchAngleBdata.shape)
    pitchAngleBdata = np.delete(pitchAngleBdata, [lookdirdel], axis=1)
    print('PITCH ANGLE: ', pitchAngleBdata.shape)
    
    start_time = t1
    end_time = t2
    
    filtered_indices = (hfluxepochdf['datetime'] >= start_time) & (hfluxepochdf['datetime'] <= end_time)
    hfluxepochdf_filtered = hfluxepochdf[filtered_indices]
    
    Hfluxdf_filtered = Hfluxdf.iloc[filtered_indices.values]
    Hefluxdf_filtered = Hefluxdf.iloc[filtered_indices.values]
    pitchAngleAdata_filtered = pitchAngleAdata[filtered_indices.values, :]
    pitchAngleBdata_filtered = pitchAngleBdata[filtered_indices.values, :]
    
    xbins = np.linspace(0, len(hfluxepochdf_filtered) - 1, num=10, dtype=int)
    datelabels = hfluxepochdf_filtered.datetime.iloc[xbins]
    
    # Ensure pitch angles data is in the correct order
    pitchAngleAdata_filtered = pitchAngleAdata_filtered[::-1]
    # Ensure pitch angles data is in the correct order
    pitchAngleBdata_filtered = pitchAngleBdata_filtered[::-1]
    # Example pitch angle data (assuming you load it via Pandas DataFrame)
    pitchAngleAdata_filtered_df = pd.DataFrame(pitchAngleAdata_filtered)

    pitchAngleBdata_filtered_df = pd.DataFrame(pitchAngleBdata_filtered)
    
    
    unique_pitch_angles = np.unique(pitchAngleAdata_filtered.flatten())
    sorted_pitch_angles = np.sort(unique_pitch_angles)

    # Reshape to match the correct plotting shape
   # reshaped_sorted_pitch_angles = sorted_pitch_angles.reshape(360, -1)
    unique_pitch_angles2 = np.unique(pitchAngleBdata_filtered.flatten())
    sorted_pitch_angles2 = np.sort(unique_pitch_angles)
    unique_sorted_pitch_angles = np.unique(np.sort(pitchAngleAdata_filtered.flatten()))
# Verify unique sorted pitch angles
    unique_sorted_pitch_angles2 = np.unique(np.sort(pitchAngleAdata_filtered.flatten()))
    # Reshape to match the correct plotting shape
#    reshaped_sorted_pitch_angles2 = sorted_pitch_angles.reshape(360, -1)

    ybins = [0, 8, 16, 24, 32, 40, 47]
    Elabels = [np.format_float_scientific(energybins[i], precision=0) for i in ybins]
    
    minflux = np.nanmin(Hfluxmean)
    maxflux = np.nanmax(Hfluxmean)
    # SWEAP data filtering
# SWEAP data filtering
    sweap_cdf = pycdf.CDF(infiles2[0])
    sweap_epoch = sweap_cdf['Epoch'][...]
    sweap_energy = sweap_cdf['ENERGY'][...]
    sweap_flux = sweap_cdf['EFLUX'][...]

# Filter SWEAP data to match the six-hour period
    sweap_filtered_indices = (sweap_epoch >= t1) & (sweap_epoch <= t2)
    sweap_epoch_filtered = sweap_epoch[sweap_filtered_indices]
    sweap_energy_filtered = sweap_energy[sweap_filtered_indices, :]
    sweap_flux_filtered = sweap_flux[sweap_filtered_indices, :]
    # Extract a single row for energy bins
    sweap_energy_bins = sweap_energy_filtered[0, :]
   # Replace zero values with a small positive number
    sweap_flux_filtered[sweap_flux_filtered == 0] = 1e-10
    
# Ensure valid min and max values for SWEAP plot
    min_sweap_flux = np.nanmin(sweap_flux_filtered)
    max_sweap_flux = np.nanmax(sweap_flux_filtered)
    # Define extents
    extents = [sweap_epoch_filtered[0], sweap_epoch_filtered[-1], 0, sweap_flux_filtered.shape[1]]
# Ensure valid min and max values for SWEAP plot
    min_sweap_flux = np.nanmin(sweap_flux_filtered)
    max_sweap_flux = np.nanmax(sweap_flux_filtered)
    ax[0].set(xlim=(t1, t2))
    ax[0].plot(cdf2['epoch_mag_RTN'][:], cdf2['psp_fld_l2_mag_RTN'][:, 0], label='Radial')
    ax[0].plot(cdf2['epoch_mag_RTN'][:], cdf2['psp_fld_l2_mag_RTN'][:, 1], label='Tangential')
    ax[0].plot(cdf2['epoch_mag_RTN'][:], cdf2['psp_fld_l2_mag_RTN'][:, 2], label='Normal')
    ax[0].set_ylabel('Magnetic Field [nT]')
    ax[0].tick_params( labelbottom=False)
    ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=11, ncol=1)
    box = ax[0].get_position()
    box.x1 = box.x1 - 0.02  # Adjust the right side of the bounding box
    ax[0].set_position(box)
    # Plotting section for H flux
    im1 = ax[1].imshow(Hfluxdf_filtered.T, cmap='turbo', aspect='auto', origin='lower', norm=matplotlib.colors.LogNorm(minflux, maxflux))
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes('right', size="1%", pad='1%')
    cbar1 = fig.colorbar(im1, location='right', cax=cax1, anchor=(0.1, 0.1), shrink=0.6)
    cbar1.set_label("H Flux [$cm^{-2}\;sr^{-1}\;s^{-1}\;keV^{-1}$]", fontsize=11)
    #ax[1].set_xlabel('Time (Year ' + str(yrbeg) + ')', fontsize=10)
    ax[1].set_ylabel('Energy [keV]', fontsize=12)
    ax[1].set_xticks(xbins)
    ax[1].xaxis.set_major_formatter(date_format)
    ax[1].xaxis.set_major_locator(locator)
    ax[1].set_xticklabels([datelabel.strftime('%Y-%m-%d %H:%M') for datelabel in datelabels], )
    ax[1].set_yticks(ybins)
    ax[1].set_yticklabels(Elabels, fontsize=8)
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].tick_params( labelbottom=False)
    ax[1].tick_params(axis='x', labelsize=9)
    ax[1].tick_params(axis='y', labelsize=9)
    
    # Manually generate unique energy bins
    unique_sweap_energy_bins = np.linspace(17923.486, 62.060547, num=2048)  # Adjust the number based on your data dimension
    print("Manually Generated Unique Energy Bins:\n", unique_sweap_energy_bins)

    # Use these bins for plotting
    sweap_energy_bins = unique_sweap_energy_bins
    # Ensure SWEAP data is in the correct order
    sweap_flux_filtered = sweap_flux_filtered[::-1]
    # Add the SWEAP data plot in ax[2]
    X, Y = np.meshgrid(mdates.date2num(sweap_epoch_filtered), np.arange(sweap_flux_filtered.shape[1]))
    im2 = ax[2].pcolormesh(X, Y, sweap_flux_filtered.T, cmap='turbo', norm=matplotlib.colors.LogNorm(min_sweap_flux, max_sweap_flux))
    divider2 = make_axes_locatable(ax[2])
    cax2 = divider2.append_axes('right', size="1%", pad='1%')
    cbar2 = fig.colorbar(im2, location='right', cax=cax2, anchor=(0.1, 0.1), shrink=0.6)
    cbar2.set_label(r"H Flux [$cm^{-2}\;sr^{-1}\;s^{-1}\;eV^{-1}$]", fontsize=11)


    # Adjust y-tick positions for proper spacing
    ybins_sweap = np.linspace(0, len(sweap_energy_bins) - 1, num=10, dtype=int)[1:-1]
    Elabels_sweap = [np.format_float_scientific(sweap_energy_bins[i], precision=0) for i in ybins_sweap]
    Elabels_sweap.reverse()
# Apply y-ticks and y-tick labels
    ax[2].set_yticks(ybins_sweap)
    ax[2].set_yticklabels(Elabels_sweap, fontsize=9)

# Ensure the plot covers the full range
    ax[2].set_ylim(0, len(sweap_energy_bins) - 1)
    ax[2].set_yticks(ybins_sweap)
    #ax[2].set_xlabel('Time (Year ' + str(yrbeg) + ')', fontsize=10)
    ax[2].set_ylabel('Energy [eV]', fontsize=12)
    ax[2].xaxis.set_major_formatter(date_format)
    ax[2].xaxis.set_major_locator(locator)
    ax[2].xaxis.set_minor_locator(AutoMinorLocator(4))    
    ax[2].set_xlim(mdates.date2num([sweap_epoch_filtered[0], sweap_epoch_filtered[-1]]))
    xticks = np.linspace(mdates.date2num(sweap_epoch_filtered[0]), mdates.date2num(sweap_epoch_filtered[-1]), num=7)
    ax[2].set_ylim(0, len(sweap_energy_bins) - 1)
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels([mdates.num2date(t).strftime('%Y-%m-%d %H:%M') for t in xticks], fontsize=8)
    ax[2].tick_params( labelbottom=False)
    ax[2].tick_params(axis='x', labelsize=9)
    ax[2].tick_params(axis='y', labelsize=9)
    # Plotting section for He flux
    im2 = ax[3].imshow(Hefluxdf_filtered.T, cmap='turbo', aspect='auto', origin='lower', norm=matplotlib.colors.LogNorm(minflux, maxflux))
    divider2 = make_axes_locatable(ax[3])
    cax2 = divider2.append_axes('right', size="1%", pad='1%')
    cbar2 = fig.colorbar(im2, location='right', cax=cax2, anchor=(0.1, 0.1), shrink=0.6)
    cbar2.set_label("He Flux [$cm^{-2}\;sr^{-1}\;s^{-1}\;keV^{-1}$]", fontsize=11)
    #ax[3].set_xlabel('Time (Year ' + str(yrbeg) + ')', fontsize=10)
    ax[3].set_ylabel('Energy [keV]', fontsize=12)
    ax[3].set_xticks(xbins)
    ax[3].xaxis.set_major_formatter(date_format)
    ax[3].xaxis.set_major_locator(locator)
    ax[3].set_xticklabels([datelabel.strftime('%Y-%m-%d %H:%M') for datelabel in datelabels], )
    ax[3].set_yticks(ybins)
    ax[3].set_yticklabels(Elabels, fontsize=8)
    ax[3].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[3].tick_params( labelbottom=False)
    ax[3].tick_params(axis='x', labelsize=9)
    ax[3].tick_params(axis='y', labelsize=9)
    
## Plotting section for pitch angles
#Plotting H Energy Flux spectrogram
    im3 = ax[4].imshow(pitchAngleAdata_filtered.T, cmap='turbo', aspect='auto', origin='lower', norm=matplotlib.colors.LogNorm(minflux, maxflux))
    divider3 = make_axes_locatable(ax[4])
    cax3 = divider3.append_axes('right', size="1%", pad='1%')
    cbar3 = fig.colorbar(im3, location='right', cax=cax3, anchor=(0.1, 0.1), shrink=0.6)
    cbar3.set_label("H Flux [$cm^{-2}\;sr^{-1}\;s^{-1}\;keV^{-1}$]", fontsize=11)
    ybins_pitch = np.linspace(0, pitchAngleAdata_filtered.shape[1] - 1, num=10, dtype=int)
    pitch_labels = [str(int(unique_sorted_pitch_angles[i * len(unique_sorted_pitch_angles) // len(ybins_pitch)])) for i in range(len(ybins_pitch))]
    # ax[4].set_xlabel('Time (Year ' + str(yrbeg) + ')', fontsize=10)
    ax[4].set_ylabel('Pitch Angle [$^\circ$]', fontsize=12)
    ax[4].set_xticks(xbins)
    ax[4].xaxis.set_major_formatter(date_format)
    ax[4].xaxis.set_major_locator(locator)
    ax[4].set_xticklabels([datelabel.strftime('%Y-%m-%d %H:%M') for datelabel in datelabels], )
    ax[4].set_yticks(ybins_pitch)
    ax[4].set_yticklabels(pitch_labels, fontsize=8)
    ax[4].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[4].tick_params( labelbottom=False)
    ax[4].tick_params(axis='x', labelsize=11)
    ax[4].tick_params(axis='y', labelsize=11)
#plotting He Energy Flux spectrogram
    im4 = ax[5].imshow(pitchAngleBdata_filtered.T, cmap='turbo', aspect='auto', origin='lower', norm=matplotlib.colors.LogNorm(minflux, maxflux))
    divider4 = make_axes_locatable(ax[5])
    cax4 = divider4.append_axes('right', size="1%", pad='1%')
    cbar4 = fig.colorbar(im4, location='right', cax=cax4, anchor=(0.1, 0.1), shrink=0.6)
    cbar4.set_label("He Flux [$cm^{-2}\;sr^{-1}\;s^{-1}\;keV^{-1}$]", fontsize=11)
    ax[5].set_xlabel('Time [hh:mm]', fontsize=12)
    ax[5].set_ylabel('Pitch Angle [$^\circ$]', fontsize=12)
    ax[5].set_xticks(xbins)
    ax[5].xaxis.set_major_formatter(date_format)
    ax[5].xaxis.set_major_locator(locator)
    ax[5].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[5].set_xticklabels([mdates.num2date(t).strftime(' %H:%M') for t in xticks], ha='left', fontsize=8 )
    ax[5].set_yticks(ybins_pitch) 
    ax[5].set_yticklabels(pitch_labels, fontsize=8)
    #ax[5].tick_params( labelbottom=False)
    ax[5].tick_params(axis='x', labelsize=11)
    ax[5].tick_params(axis='y', labelsize=11)
    plt.show()
    fig.savefig(yrbeg + mnbeg + dybeg + '_' + str(kk), bbox_inches="tight")
    plt.close()
