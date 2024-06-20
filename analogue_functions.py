# Functions for analogues

import subprocess
import iris
import iris.coord_categorisation as icc # type: ignore
from iris.coord_categorisation import add_season_membership
import numpy as np
import cartopy.crs as ccrs
import cartopy as cart
import glob
import matplotlib.cm as mpl_cm
import os, sys
import scipy.stats as sps
from scipy.stats import genextreme as gev
import random
import scipy.io
import xarray as xr
import netCDF4 as nc
import iris.coords # type: ignore
import iris.util # type: ignore
from iris.util import equalise_attributes # type: ignore
from iris.util import unify_time_units # type: ignore
from scipy.stats.stats import pearsonr
import scipy.stats as stats
import calendar
import random
from logger import LOGGER

# Are we using daily or monthly data?
ERA5FILESUFFIX : str = "_daily"

def reanalysis_file_location() -> str:
    '''
    Return the location of the ERA5 data
    '''
    return os.path.join(os.environ["CLIMEXP_DATA"],"ERA5")

def find_reanalysis_filename(var : str, daily : bool = True) -> str:
    '''
    Return the field filename for a given variable
    '''
    suffix : str = ERA5FILESUFFIX
    path : str = os.path.join(reanalysis_file_location(),"era5_{0}{1}.nc".format(var,suffix))
    LOGGER.info("Reading: {}".format(path))
    return path

def find_season_from_date(date : list) -> str:
    '''
    Find the season based on the date, using the month element
    '''
    M = date[1]
    if M == 'Dec' or M == 'Jan' or M == 'Feb': season = 'djf'
    if M == 'Mar' or M == 'Apr' or M == 'May': season = 'mam'
    if M == 'Jun' or M == 'Jul' or M == 'Aug': season = 'jja'
    if M == 'Sep' or M == 'Oct' or M == 'Nov': season = 'son'
    return season

def event_data_era(date : list) -> list:
    '''
    Get ERA data for a defined list of variables on a given event date
    '''
    event_list = []
    # TODO: Convert variable list into a non-local (possibly with a class?)
    for variable in ['msl', 'tp', 't2m', 'sfcWind']:
        event_list.append(reanalysis_data_single_date_OLD(variable, date))
    return event_list

def period_outputs_msl(Y1, Y2, event_msl, N, date, season, R1):
    '''
    Function to identify the N closest analogues of (N: number)
    date between (date format: [YYYY, 'Mon', DD], e.g. [2021, 'Jul', 14])
    Y1 and Y2, (year between 1950 and 2023)
    for sea level pressure field, event_msl.
    '''
    P1_msl = reanalysis_data_OLD('msl', Y1, Y2, season)
    P1_field = extract_region(P1_msl, R1)
    LOGGER.debug("Extracted region from cube {0}".format(P1_field))
    # Note: original code was this, I assume because cubes was a list
    #E = regrid(event_msl, P1_field[0][0,...])
    E = regrid(event_msl, P1_field[0,...])
    P1_dates = analogue_dates(E, P1_field, R1, N*5)[:N+1]
    # Remove the date being searched for
    if str(date[0])+str(list(calendar.month_abbr).index(date[1]))+str(date[2]) in P1_dates:
        P1_dates.remove(str(date[0])+str(list(calendar.month_abbr).index(date[1]))+str(date[2]))
    P1_ED = eucdist_of_datelist(E, P1_field, P1_dates, R1)
    P1_comp = extract_region(composite_dates(P1_msl, P1_dates), R1)
    return P1_dates, P1_ED, P1_comp, P1_field

def ED_similarity(event, P_cube, region):
    '''
    Returns similarity values based on euclidean distance
    '''
    E = extract_region(event, region)
    P = extract_region(P_cube, region)
    D = euclidean_distance(P, E)
    ED_max = np.max(np.max(D))
    S = [(1-x / ED_max) for x in D]
    S_ann = P_cube.collapsed(('longitude','latitude'),iris.analysis.MEAN).copy()
    S_ann.data = S
    return S_ann

def regrid(original, new):
    ''' Regrids onto a new grid '''
    mod_cs = original.coord_system(iris.coord_systems.CoordSystem)
    new.coord(axis='x').coord_system = mod_cs
    new.coord(axis='y').coord_system = mod_cs
    LOGGER.info("Regridding to {}".format(mod_cs))
    new_cube = original.regrid(new, iris.analysis.Linear())
    return new_cube

def extract_region(cube_list, R1):
    '''
    Extract Region (defaults to Europe)
    '''
    const_lat = iris.Constraint(latitude = lambda cell:R1[1] < cell < R1[0])
    if isinstance(cube_list, iris.cube.Cube):
        LOGGER.info("Extracting a region from a single cube object: {}".format([cube_list]))
        reg_cubes_lat = cube_list.extract(const_lat)
        reg_cubes = reg_cubes_lat.intersection(longitude=(R1[3], R1[2]))
    elif isinstance(cube_list, iris.cube.CubeList):
        LOGGER.info("Extracting a region from a cube list: {}".format(cube_list))
        reg_cubes = iris.cube.CubeList([])
        for each in range(len(cube_list)):
            print(each)
            subset = cube_list[each].extract(const_lat)
            reg_cubes.append(subset.intersection(longitude=(R1[3], R1[2])))
    return reg_cubes

def euclidean_distance(field, event):
    '''
    Returns list of D
    Inputs required:
      field = single cube of JJA psi.
      event = cube of single day of event to match.
      BOTH MUST HAVE SAME DIMENSIONS FOR LAT/LON
    '''
    D = [] # to be list of all euclidean distances
    a, b, c = np.shape(field.data)
    XA = event.data.reshape(b*c,1)
    #yrs.append(each.coord('year').points[0])
    XB = field.data.reshape(np.shape(field.data)[0], b*c, 1)
    LOGGER.info("Shape of XB: {}".format(XB.shape))
    for Xb in XB:
        D.append(np.sqrt(np.sum(np.square(XA - Xb))))
    return D


def reanalysis_data(var, Y1=1950, Y2=2023, seas='son'):
    '''
    Loads in reanalysis daily data
    VAR can be psi250, msl, or tp (to add more)
    '''
    cubes = iris.load(find_reanalysis_filename(var), var)
    try:
        cube = cubes[0]
    except:
        LOGGER.error("Error reading cubes for %s", var)
        raise FileNotFoundError
    iris.coord_categorisation.add_year(cube, 'time')
    cube = cube.extract(iris.Constraint(year=lambda cell: Y1 <= cell < Y2))
    iris.coord_categorisation.add_season(cube, 'time')
    cube = cube.extract(iris.Constraint(season=seas))
    return cube

def reanalysis_data_OLD(var, Y1=1950, Y2=2023, seas='son'):
    '''
    Loads in reanalysis daily data
    VAR can be psi250, msl, or tp (to add more)
    OLD version pre-Climate Explorer, kept for reference
    '''
    filename = glob.glob('/net/pc230042/nobackup/users/sager/nobackup_2_old/ERA5-CX-READY/era5_*'+var+'_daily.nc')
    #print(files)
    cube = iris.load(filename)[0]
    # Extract years
    iris.coord_categorisation.add_year(cube, 'time')
    cube = cube.extract(iris.Constraint(year=lambda cell: Y1 <= cell < Y2))
    # Extract single season
    iris.coord_categorisation.add_season(cube, 'time')
    cube = cube.extract(iris.Constraint(season=seas))
    return cube

def analogue_dates(event, reanalysis_cube, region, N):
    '''
    '''
    def cube_date_to_string(cube_date : tuple) -> tuple:
        year,month,day,time = cube_date
        return str(year)+str(month).zfill(2)+str(day).zfill(2), time
    E = extract_region(event, region)
    reanalysis_cube = extract_region(reanalysis_cube, region)
    D = euclidean_distance(reanalysis_cube, E)
    date_list = []
    time_list = []
    for i in np.arange(N):
        #print(i)
        I = np.sort(D)[i]
        for n, each in enumerate(D):
            if I == each:
                a1 = n
        date, time = cube_date_to_string(cube_date(reanalysis_cube[a1,...]))
        date_list.append(date)
        time_list.append(time)
        date_list2 = date_list_checks(time_list, date_list, days=5)
    return date_list2


def cube_date(cube):
    '''
    Returns date of cube (assumes cube single day)
    '''
    if len(cube.coords('year')) > 0:
       pass
    else:
       iris.coord_categorisation.add_year(cube, 'time')
    if len(cube.coords('month')) > 0:
       pass
    else:
       iris.coord_categorisation.add_month(cube, 'time')
    if len(cube.coords('day_of_month')) > 0:
       pass
    else:
       iris.coord_categorisation.add_day_of_month(cube, 'time')
    if len(cube.coords('day_of_year')) > 0:
       pass
    else:
       iris.coord_categorisation.add_day_of_year(cube, 'time')
    year = cube.coord('time').units.num2date(cube.coord('time').points)[0].year
    month = cube.coord('time').units.num2date(cube.coord('time').points)[0].month
    day = cube.coord('time').units.num2date(cube.coord('time').points)[0].day
    time = cube.coord('time').points[0]
    return year, month, day, time

def date_list_checks(time_list, date_list, days=5):
    '''
    Takes date_list and removes:
     1) the original event (if present)
     2) any days within 5 days of another event
    '''
    new_date_list = date_list.copy()
    # Remove neighbours (if present). Auto 120hrs = 5 days
    hrs = days*24
    for i in np.arange(1,len(time_list)):
        for i_earlier in np.arange(i):
            if (time_list[i]-hrs)<=time_list[i_earlier]<=(time_list[i]+hrs) and date_list[i_earlier][:4] == date_list[i][:4]:
                new_date_list.remove(date_list[i])
                break
            else:
                pass
    return new_date_list

def eucdist_of_datelist(event, reanalysis_cube, date_list, region):
    ED_list = []
    E = extract_region(event, region)
    for i, each in enumerate(date_list):
        yr = int(date_list[i][:4])
        mon = calendar.month_abbr[int(date_list[i][4:-2])]
        day = int(date_list[i][-2:])
        field = extract_region(pull_out_day_era(reanalysis_cube, yr, mon, day), region)
        b, c = np.shape(field)
        XA = E.data.reshape(b*c,1)
        XB = field.data.reshape(b*c, 1)
        D = np.sqrt(np.sum(np.square(XA - XB)))
        ED_list.append(D)
    return ED_list

def pull_out_day_era(psi, sel_year, sel_month, sel_day):
    psi_day = extract_date(psi, sel_year, sel_month, sel_day)
    try:
        return psi_day
    except NameError:
        print('ERROR: Date not in data')
        return

def extract_date(cube, yr, mon, day):
   '''
   Extract specific day from cube of a single year
   '''
   if len(cube.coords('year')) > 0:
       pass
   else:
       iris.coord_categorisation.add_year(cube, 'time')
   if len(cube.coords('month')) > 0:
       pass
   else:
       iris.coord_categorisation.add_month(cube, 'time')
   if len(cube.coords('day_of_month')) > 0:
       pass
   else:
       iris.coord_categorisation.add_day_of_month(cube, 'time')
   return cube.extract(iris.Constraint(year=yr, month=mon, day_of_month=day))

def composite_dates(psi, date_list):
    '''
    Returns single composite of all dates
    Inputs required:
      psi = list of cubes, 1 per year - as used to calc D/date_list
      date_list = list of events to composite
    '''
    n = len(date_list)
    FIELD = 0
    for each in range(n):
        year = int(date_list[each][:4])
        month = calendar.month_abbr[int(date_list[each][4:-2])]
        day = int(date_list[each][-2:])
        NEXT_FIELD = pull_out_day_era(psi, year, month, day)
        if NEXT_FIELD == None:
            print('Field failure for: ',+each)
            n = n-1
        else:
            if FIELD == 0:
                FIELD = NEXT_FIELD
            else:
                FIELD = FIELD + NEXT_FIELD
    return FIELD/n


def reanalysis_data_single_date(var : str, date : list):
    '''
    Loads in reanalysis daily data
    VAR can be psi250, msl, or tp (to add more)
    '''
    year, month, day = date
    filename = find_reanalysis_filename(var)
    LOGGER.info("Read file: {} for date {}".format(filename,date))
    cube = iris.load(filename, var)[0]
    cube = extract_date(cube,date[0],date[1],date[2])
    return cube


def reanalysis_data_single_date_OLD(var, date):
    '''
    Loads in reanalysis daily data
    VAR can be t2m, msl, or tp (to add more)
    '''
    filename = glob.glob('/net/pc230042/nobackup/users/sager/nobackup_2_old/ERA5-CX-READY/era5_*'+var+'_daily.nc')
    cube = iris.load(filename, var)[0]
    cube = extract_date(cube,date[0],date[1],date[2])
    return cube

def euclidean_quality_analogs(field, date_list, N=30):
    '''
    For the 30 closest analogs of the event day, calculates analogue quality
    '''
    Q = []
    for i, each in enumerate(date_list):
        YY = np.int64(each[:4])
        MM = calendar.month_abbr[int(each[4:-2])]
        DD = int(each[-2:])
        analog_event = pull_out_day_era(field, YY, MM, DD)
        D = euclidean_distance(field, analog_event) # calc euclidean distances
        Q.append(np.sum(np.sort(D)[:N]))
    return Q


def composite_dates_significance(psi, date_list):
    '''
    Returns single composite of all dates
    Inputs required:
      psi = list of cubes, 1 per year - as used to calc D/date_list
      date_list = list of events to composite
    '''
    n = len(date_list)
    field_list = iris.cube.CubeList([])
    for each in range(n):
        year = int(date_list[each][:4])
        month = calendar.month_abbr[int(date_list[each][4:-2])]
        day = int(date_list[each][-2:])
        field_list.append(pull_out_day_era(psi, year, month, day))
    sig_field = field_list[0].data
    a, b = np.shape(field_list[0].data)
    for i in range(a):
        for j in range(b):
            loc_list = []
            for R in range(n):
                loc_list.append(field_list[R].data[i,j])
            if np.abs(np.mean(loc_list)) > np.abs(np.std(loc_list)):
                sig_field[i,j] = 1
            else:
                sig_field[i,j] = 0
    result_cube = field_list[0]
    result_cube.data = sig_field
    return result_cube