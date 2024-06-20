
# Inputs: 
# date (of event of interest)
# N: number of analogues to identify
# R1: region to assess analogues over
# Y1, Y2: period to assess analogues over

import sys
import analogue_functions as my

import xarray as xr
import os
import subprocess
import numpy as np
import iris
import calendar
import matplotlib.pyplot as plt
import matplotlib
#cmaps = matplotlib.colormaps

import cartopy.crs as ccrs
import cartopy as cart
import cartopy.feature as cf
import glob

from logger import LOGGER

def save_figure_file(filename : str) -> None:
    """
    Make a figure name
    """
    # TODO: Change to defined climexp location when deployed
    figurefolder = "figures"
    plt.savefig(os.path.join(figurefolder,filename))
    

## Define Variable
date = [2021, 'Jul', 14]; # date of event
R1 = [59, 41, 23, -5] # region to identfy analogues over
N = 20 # number of analogues to composite
Y1 = 2010 # time period to assess over (start) minimum 1950
Y2 = 2022 # time period to assess over (end) maximum 2023 


## Select data frequency (monthly = "", daily = "_daily", based on naming system on Climate Explorer)
#my.ERA5FILESUFFIX = "_daily"

## Define season
season = my.find_season_from_date(date)

event_msl, event_tp, event_t2m, event_wind = my.event_data_era(date)
LOGGER.debug("Made events: {%s} {%s} {%s} {%s}", event_msl, event_tp, event_t2m, event_wind)

## Top analogues over a definied period

# Identifies top N analogues between Y1 and Y2 for field event_msl 
# based on event date within season for region R1
dates, ED, SLP_comp, field = my.period_outputs_msl(Y1, Y2, event_msl, N, date, season, R1)
LOGGER.info("Analogues dates found: {}".format(dates))

# Get variables data field, pull out the analogue dates and composite
TP_field = my.reanalysis_data_OLD('tp', Y1, Y2, season)
TP_comp = my.extract_region(my.composite_dates(TP_field, dates), R1)
WIND_field = my.reanalysis_data_OLD('sfcWind', Y1, Y2, season)
WIND_comp = my.extract_region(my.composite_dates(WIND_field, dates), R1)
t2m_field = my.reanalysis_data_OLD('t2m', Y1, Y2, season)
t2m_comp = my.extract_region(my.composite_dates(t2m_field, dates), R1)

# Assess if the analogues patterns are significant (mean > stdev)
SLP_sig = my.composite_dates_significance(field-np.mean(field[0].data), dates)
TP_sig = my.extract_region(my.composite_dates_significance(TP_field, dates), R1)
WIND_sig = my.extract_region(my.composite_dates_significance(WIND_field-np.mean(WIND_field[0].data), dates), R1)
t2m_sig = my.extract_region(my.composite_dates_significance(t2m_field-np.mean(t2m_field[0].data), dates), R1)

## Figure - top N analogues of whole period, and surface impacts (rain & wind)
event_msl_reg = my.regrid(my.extract_region(event_msl, R1), SLP_comp)
fig, axs = plt.subplots(nrows=2, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10,4))
lats=event_msl_reg.coord('latitude').points
lons=event_msl_reg.coord('longitude').points
stip_levs = [-2, 0, 2]

# SLP
con_lev = np.linspace(np.min([np.min(SLP_comp.data/100), np.min(event_msl_reg.data/100)]), np.max([np.max(SLP_comp.data/100), np.max(event_msl_reg.data/100)]), 20)
c1 = axs[0,0].contourf(lons, lats, event_msl_reg.data/100, levels=con_lev, cmap = plt.cm.get_cmap('RdBu_r'), transform=ccrs.PlateCarree(), extend='both')
axs[0,0].add_feature(cf.COASTLINE, linewidth=0.5)
c1 = axs[1,0].contourf(lons, lats, SLP_comp.data/100, levels=con_lev, cmap = plt.cm.get_cmap('RdBu_r'), transform=ccrs.PlateCarree(), extend='both')
c5 = axs[1,0].contourf(lons, lats, SLP_sig.data, levels=stip_levs, hatches=['////', None], colors='none', transform=ccrs.PlateCarree())
axs[1,0].add_feature(cf.COASTLINE, linewidth=0.5)

# TP
event_tp = my.extract_region(event_tp, R1)
con_lev = np.linspace(0, np.max([np.max(TP_comp.data), np.max(event_tp.data)]), 20)
c2 = axs[0,1].contourf(lons, lats, event_tp.data, levels=con_lev, cmap = plt.cm.get_cmap('Blues'), transform=ccrs.PlateCarree(), extend='max')
axs[0,1].add_feature(cf.COASTLINE, linewidth=0.5)
c2 = axs[1,1].contourf(lons, lats, TP_comp.data, levels=con_lev, cmap = plt.cm.get_cmap('Blues'), transform=ccrs.PlateCarree(), extend='max')
c5 = axs[1,1].contourf(lons, lats, TP_sig.data, levels=stip_levs, hatches=['////', None], colors='none', transform=ccrs.PlateCarree())
axs[1,1].add_feature(cf.COASTLINE, linewidth=0.5)

# Wind
event_wind = my.extract_region(event_wind, R1)
con_lev = np.linspace(0, np.max([np.max(WIND_comp.data), np.max(event_wind.data)]), 20)
c3 = axs[0,2].contourf(lons, lats, event_wind.data, levels=con_lev, cmap = plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(), extend='max')
axs[0,2].add_feature(cf.COASTLINE, linewidth=0.5)
c3 = axs[1,2].contourf(lons, lats, WIND_comp.data, levels=con_lev, cmap = plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(), extend='max')
c5 = axs[1,2].contourf(lons, lats, WIND_sig.data, levels=stip_levs, hatches=['////', None], colors='none', transform=ccrs.PlateCarree())
axs[1,2].add_feature(cf.COASTLINE, linewidth=0.5)

# t2m
event_t2m = my.extract_region(event_t2m, R1)
con_lev = np.linspace(np.min([np.min(t2m_comp.data-273.15), np.min(event_t2m.data-273.15)]), np.max([np.max(t2m_comp.data-273.15), np.max(event_t2m.data-273.15)]), 20)
c4 = axs[0,3].contourf(lons, lats, event_t2m.data-273.15, levels=con_lev, cmap = plt.cm.get_cmap('RdBu_r'), transform=ccrs.PlateCarree(), extend='max')
axs[0,3].add_feature(cf.COASTLINE, linewidth=0.5)
c4 = axs[1,3].contourf(lons, lats, t2m_comp.data-273.15, levels=con_lev, cmap = plt.cm.get_cmap('RdBu_r'), transform=ccrs.PlateCarree(), extend='max')
c5 = axs[1,3].contourf(lons, lats, t2m_sig.data, levels=stip_levs, hatches=['////', None], colors='none', transform=ccrs.PlateCarree())
axs[1,3].add_feature(cf.COASTLINE, linewidth=0.5)

fig.subplots_adjust(bottom=0.15, left=.25, wspace=0.1, hspace=.2)
cbar_ax = fig.add_axes([0.25, 0.1, 0.15, 0.03])
cbar=fig.colorbar(c1, cax=cbar_ax, orientation='horizontal')
cbar.ax.locator_params(nbins=3)
cbar_ax.set_xlabel('Sea Level Pressure, hPa')
cbar_ax2 = fig.add_axes([0.42, 0.1, 0.15, 0.03])
cbar=fig.colorbar(c2, cax=cbar_ax2, orientation='horizontal')
cbar.ax.locator_params(nbins=3)
cbar_ax2.set_xlabel('Rainfall, mm/day')
cbar_ax3 = fig.add_axes([0.58, 0.1, 0.15, 0.03])
cbar=fig.colorbar(c3, cax=cbar_ax3, orientation='horizontal')
cbar.ax.locator_params(nbins=3)
cbar_ax3.set_xlabel('Wind, m/s')
cbar_ax4 = fig.add_axes([0.75, 0.1, 0.15, 0.03])
cbar=fig.colorbar(c4, cax=cbar_ax4, orientation='horizontal')
cbar.ax.locator_params(nbins=3)
cbar_ax4.set_xlabel('T2M, degC')


#save_figure_file("analogues_maps.png")
# Assess quality of analogues
Q_event = sum(ED)
Q_ana = my.euclidean_quality_analogs(field, dates, N=len(ED))

Q_ana_v2 = []
for val in Q_ana:
    Q_ana_v2.append(1/val)

Q_event_v2 = 1/(Q_event)


### Violin plot of typicality
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,5))
v1 = ax.violinplot(Q_ana_v2, showmeans=True, showextrema=True, showmedians=False)
ax.axhline(Q_event_v2, color='r', linewidth=2)
ax.set_xticks([1])
ax.set_xticklabels([''])
#ax.set_ylabel('Q (x $10^{10}$ m$^2$/s)')
ax.set_title('Quality')

#save_figure_file("analogues_violin.png")


# Create timeseries of the top annual analogue
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
P1_msl = my.reanalysis_data_OLD('msl', Y1, Y2, season)
event = my.extract_region(event_msl, R1)  
P1_field = my.extract_region(P1_msl, R1)
LOGGER.info("Made P1_field: {}".format([P1_field]))
event = my.regrid(event, P1_field[0,...])
S_ann = my.ED_similarity(event, P1_field, R1)   # returns a cube
S_max = S_ann.aggregated_by('year',iris.analysis.MAX)
ax.plot(S_max.coord('year').points, S_max.data)   # data is in cube
LOGGER.info("Time series found: {}".format(S_max))

#save_figure_file("analogues_timeseries.png")



