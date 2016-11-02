"""
Python 3 only solution?
"""
import os
from time import sleep
import numpy as np

import plotly.plotly as py
from plotly.graph_objs import *
from plotly.offline import plot, iplot, init_notebook_mode
from plotly.tools import set_credentials_file

set_credentials_file(username='bmmorris', api_key='99nfob8e81')
#init_notebook_mode(connected=True)

from results import MCMCResults
from astropy.time import Time
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/astro/users/bmmorris/git/friedrich')
from friedrich.lightcurve import hat11_params_morris
from friedrich.orientation import (planet_position_cartesian,
                                   project_planet_to_stellar_surface,
                                   observer_view_to_stellar_view,
                                   cartesian_to_spherical,
                                   times_to_occulted_lat_lon)

plotly_star = True
mpl_lightcurve = True

window_ind = int(sys.argv[-1])

#m = MCMCResults.from_stsp('/local/tmp/osg/tmp/hat11-osg/', 33)
#m.to_hdf5('/local/tmp/osg/tmp/hat11-osg/')
#m = MCMCResults.from_hdf5('/local/tmp/osg/tmp/hat11-osg/', 33, hat11_params_morris())
m = MCMCResults.from_stsp('/local/tmp/osg/tmp', window_ind)

def spot_polar_to_latlon(radius, theta, phi, transit_params,
                         light_curve, n_points=20):
    """
    Return lat/lon that traces spot, in degrees.

    Conversion formula discovered here:
    http://williams.best.vwh.net/avform.htm#LL
    """
    lat1 = np.radians(90 - np.degrees(theta))
    lon1 = (phi + 2 * np.pi * (light_curve.times.mean() - transit_params.t0) /
            transit_params.per_rot)

    # s = r theta  -->  theta = s / r
    d = radius
    thetas = np.linspace(0, -2 * np.pi, n_points)[:, np.newaxis]

    lat = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) *
                    np.sin(d) * np.cos(thetas))
    dlon = np.arctan2(np.sin(thetas) * np.sin(d) * np.cos(lat1),
                      np.cos(d) - np.sin(lat1) * np.sin(lat))
    lon = ((lon1 - dlon + np.pi) % (2 * np.pi)) - np.pi
    lat, lon = np.degrees(lat), np.degrees(lon)

    traces = []

    for i in range(lat.shape[1]):
        trace = Scattergeo(lat=lat[:, i], lon=lon[:, i],
                           fillcolor='rgba(0, 0, 0, 0.3)',
                           fill='toself', mode='lines',
                           line=dict(color='black', width=0))
        traces.append(trace)

    return traces


def times_to_shadows(all_times, transit_params, rotate_star=False):
    """Only works for single time inputs at the moment"""
    # rotate_star: added on Oct 14, 2016 for STSP simulations
    traces = []
    for time in all_times:
        times = np.array([time])
        X, Y, Z = planet_position_cartesian(times, transit_params)

        n_points = 20
        latitudes = np.zeros(n_points)
        longitudes = np.zeros(n_points)

        thetas = np.linspace(0, -2 * np.pi, n_points)

        for i in range(n_points):
            x_circle = X + transit_params.rp * np.cos(thetas[i])
            y_circle = Y + transit_params.rp * np.sin(thetas[i])
            spot_x, spot_y, spot_z = project_planet_to_stellar_surface(x_circle,
                                                                       y_circle)

            spot_x_s, spot_y_s, spot_z_s = observer_view_to_stellar_view(spot_x,
                                                                         spot_y,
                                                                         spot_z,
                                                                         transit_params,
                                                                         times,
                                                                         rotate_star=rotate_star)
            spot_r, spot_theta, spot_phi = cartesian_to_spherical(spot_x_s,
                                                                  spot_y_s,
                                                                  spot_z_s)
            longitudes[i] = np.degrees(spot_phi)
            latitudes[i] = np.degrees(np.pi / 2 - spot_theta)

        non_nan = np.logical_not(np.isnan(latitudes) | np.isnan(longitudes))

        trace = Scattergeo(lat=latitudes[non_nan], lon=longitudes[non_nan],
                           fill='toself', mode='lines',
                           fillcolor='rgba(0, 0, 0, 1)',
                           line=dict(color='rgba(0, 0, 0, 1)', width=0))
        traces.append(trace)
    return traces


def render_with_plotly(fig, path, width=1000, height=1000):
    counter = 0
    while not os.path.exists(path) and counter < 10:
        try:
           py.image.save_as(fig, path, width=width, height=height)
        except KeyError:
            print('Plotly API request fail {0}'.format(counter))
            counter += 1
            sleep(5)

transit_params = hat11_params_morris()
transit_params.t0 += 0.25 * transit_params.per
min_chi2 = np.argmin(m.chi2)
r = m.radius[min_chi2, :]
t = m.theta[min_chi2, :]
p = m.phi[min_chi2, :]

spot_traces = spot_polar_to_latlon(r, t, p, transit_params, m.light_curve)

chord_lat, chord_lon = times_to_occulted_lat_lon(m.light_curve.times, 
                                                 transit_params)
occulted_times = np.logical_not(np.isnan(chord_lat))
chord_lat = np.degrees(chord_lat[occulted_times])
chord_lon = np.degrees(chord_lon[occulted_times])
chord_trace = Scattergeo(lat=chord_lat, lon=chord_lon, 
                         mode='lines', line=dict(color='black', width=2))

equator_trace = Scattergeo(lat=np.zeros(5), lon=np.linspace(0, 360, 5), 
                           mode='lines', line=dict(color='rgb(20, 20, 20)')) # , dash='dash'

planet_shadows = times_to_shadows(m.light_curve.times, transit_params)

lat_lon_grids = dict(gridcolor='rgb(200, 200, 200)', 
                     gridwidth=1.5, showgrid=True)
rotation = dict(lat=-10, lon=0, roll=-106)
projection = dict(rotation=rotation, type='orthographic', scale=0.8)
geo = dict(lataxis=lat_lon_grids, lonaxis=lat_lon_grids,
           projection=projection, showland=False, coastlinewidth=0)

layout = Layout(geo=geo, showlegend=False) # title='HAT-P-11, transit 33', 

shadows_in_transit = [shadow for occulted, shadow in zip(occulted_times, planet_shadows) if occulted]

n = len(shadows_in_transit)
skip = 2 

star_dir = 'star_plots_{0:03d}'.format(window_ind)
lc_dir = 'lc_plots_{0:03d}'.format(window_ind)

for directory in [star_dir, lc_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

for i, shadow, time in zip(range(len(shadows_in_transit[::skip])),
                           shadows_in_transit[::skip], 
                           m.light_curve.times[occulted_times][::skip]):
    print(i)
    
    fig = Figure(data=Data(spot_traces + [chord_trace, equator_trace, shadow]), 
                 layout=layout)

    if plotly_star:
        render_with_plotly(fig, os.path.join(star_dir, '{0:04d}.png'.format(i)))
        # try:
        #     py.image.save_as(fig, os.path.join(star_dir, '{0:04d}.png'.format(i)), width=1000, height=1000)
        # except KeyError:
        #     sleep(5)
        #     py.image.save_as(fig, os.path.join(star_dir, '{0:04d}.png'.format(i)), width=1000, height=1000)

    if mpl_lightcurve:
        fig, ax = m.light_curve.plot_transit()
        
        ax.axvline(Time(time, format='jd').plot_date, ls='--', lw=2, color='k')
        
        savefig_kwargs = dict(bbox_inches='tight', dpi=200)
        fig.savefig(os.path.join(lc_dir, '{0:04d}.png'.format(i)), **savefig_kwargs)
        plt.close()

import shutil

if plotly_star:
    shutil.copy('animate', '{0}/.'.format(star_dir))
    os.chdir(star_dir)
    os.system('bash ./animate')
    os.chdir('../')

if mpl_lightcurve:
    shutil.copy('animate', '{0}/.'.format(lc_dir))
    os.chdir(lc_dir)
    os.system('bash ./animate')
    os.chdir('../')