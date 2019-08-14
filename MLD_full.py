#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:12:19 2018

@author: regina
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

# =============================================================================
# Edit parameters in function as necessary.
# =============================================================================
def interpolate(group): 
    if len(group) < 7:
        return np.nan
    reference_row = group[group.depth == 5]
    assert len(reference_row) <= 1
    if len(reference_row) == 0:
        return np.nan
    reference_temp = reference_row.iloc[0].temperature
    x0 = reference_temp - 0.2
    x = group.temperature.values
    y = group.depth.values
    xs = np.sort(x)
    ys = np.array(y)[np.argsort(x)]
    return np.interp(x0, xs, ys)

# =============================================================================
# Open a shell and launch the program in python from the command line.
# Include the positional argument (THEMO csv file) and help command <-h>
# to describe options for visualizations of data. 
# =============================================================================

parser = argparse.ArgumentParser(description='This program linearly interpolates and visualizes the mixed layer depth from continuous csv data.',
                                 epilog='Regina Lionheart, CROSSMAR Lab, University of Haifa')
parser.add_argument('file', help='a path to an csv file containing depth and temperature measurements')
parser.add_argument('--graph', action='store_true', default=True, help='default: whether to output a graph of the MLD Interpolation')
parser.add_argument('--scatter', action='store_true', default=False, help='whether to output a scatter plot of temperatures at depth')
parser.add_argument('--depth', action='store_true', default=False, help='whether to output a depth profile of a particular cast')
args = parser.parse_args()

# =============================================================================
# Import SST data from Themo. 
# =============================================================================

ds = pd.read_csv(args.file, parse_dates=[[0, 1]], index_col=0)
ds = ds.drop(columns=['threshold', 'const_err', 's9_id']) # Drop unusued columns
ds = ds[(ds.temperature < 35)] # Temp threshold can be edited for seasons
ds = ds.drop_duplicates()
ds.describe() # Check for any obviously wrong depth/temperature data
s = ds.groupby('d_stamp_t_stamp').apply(interpolate)
s = s.dropna()
s.columns = ['Date', 'Interpolated MLD']
print(s)

if args.graph:
    ## Time series plot of interoplated MLD
    mld_fig = s.plot(figsize=(11,8), marker = 'o', color='darkblue', 
                     markersize=5)
    plt.title('Interpolated Mixed Layer Depth')
    plt.xlabel('Date')
    plt.ylabel('Depth of the Mixed Layer')
    plt.gca().invert_yaxis()


if args.scatter:
    ## Scatter plot of original dataset that shows outliers at depth
    scatter = plt.figure(figsize=(10,7))
    plt.scatter(ds.temperature, ds.depth) 
    plt.title('Temperature Distribution Over Themo Depths', fontsize=20)
    plt.xlabel('Temperature in C')
    plt.ylabel('Depth')
    plt.legend('Temperature Measureent', fontsize=15, loc=0)
    plt.gca().invert_yaxis()
    
if args.depth:
    ## Depth profile of a single cast, showing interpolated MLD
    cast = input("Enter your cast of interest, eg. ['2018-06-01 12:30:00']: ")
    date = cast
    date = ds.loc[date]
    date = date.sort_values(by='depth', ascending=True)
    x = np.asarray(date.temperature)
    y = np.asarray(date.depth)
    xs = np.sort(x)
    ys = np.array(y)[np.argsort(x)]
    x0 = x[0] - 0.2
    y0 = np.interp(x0, xs, ys)
    fig = plt.figure(figsize=(10,7)), plt.plot(style='.-')
    plt.suptitle('Interpolated MLD from Themo Cast')
    plt.title(cast)
    plt.xlabel('Temperature in C'), plt.ylabel('Depth')
    plt.gca().invert_yaxis()
    plt.plot(x,y, linestyle='--', marker='o', color='b')
    plt.plot(x0,y0, marker='o', markersize='10', color='C3')
    plt.annotate(
            'MLD', xy=(x0, y0), ha='left',
            va='top', textcoords='offset points', bbox=dict(BoxStyle='Round, pad=0.5', fc='yellow',
            alpha=0.5), arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()



