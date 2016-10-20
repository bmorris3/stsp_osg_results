from __future__ import absolute_import, print_function, division

import os
import numpy as np
from glob import glob
import matplotlib
from matplotlib import pyplot as plt
import astropy.units as u
from astropy.constants import R_sun
from astropy.coordinates import SphericalRepresentation
from .lightcurve import LightCurve
import corner
import h5py

__all__ = ['MCMCResults']


class MCMCResults(object):
    def __init__(self, radius=None, lat=None, lon=None, acceptance_rates=None,
                 n_spots=None, burnin=None, window_ind=None):
        self.radius = radius
        self.lat = lat
        self.lon = lon
        self.acceptance_rates = acceptance_rates
        self.n_spots = n_spots
        self.burnin = burnin
        self.window_ind = window_ind

    @classmethod
    def from_stsp(cls, results_dir, window_ind, burnin=0.8):

        table = []
        chain_ind = []
        burnin = burnin
        acceptance_rates = []

        paths = sorted(glob(os.path.join(results_dir,
                                         'window{0:03d}/run???/*_mcmc.txt'
                                         .format(window_ind))))

        for path in paths:

            results_file_size = os.stat(path).st_size

            if results_file_size > 0:
                table = np.loadtxt(path)
                print("Loading", path)

                n_walkers = len(np.unique(table[:, 0]))
                n_accepted_steps = np.count_nonzero(table[:, 2] != 0)
                n_steps_total = np.max(table[:, 2])
                acceptance_rates.append(n_accepted_steps /
                                             n_steps_total / n_walkers)

                if len(table) == 0:
                    table = table
                else:
                    table = np.vstack([table, table])

                chain_ind = np.concatenate([chain_ind, table[:, 0]])

        n_properties_per_spot = 3
        col_offset = 4
        n_spots = (table.shape[1] - col_offset)//n_properties_per_spot

        radius_col = (col_offset + n_properties_per_spot *
                           np.arange(n_spots))
        lat_col = (col_offset + 1 + n_properties_per_spot *
                        np.arange(n_spots))
        lon_col = (col_offset + 2 + n_properties_per_spot *
                        np.arange(n_spots))

        # Note: latitude is defined on (0, pi) rather than (-pi/2, pi/2)
        radius = table[:, radius_col]
        lat = table[:, lat_col]
        lon = table[:, lon_col]

        burnin_int = int(burnin*table.shape[0])

        kwargs = dict(burnin=burnin, acceptance_rates=acceptance_rates,
                      radius=radius, lat=lat, lon=lon, n_spots=n_spots,
                      window_ind=window_ind)

        return cls(**kwargs)

    def to_hdf5(self, results_dir):

        hdf5_results_dir = os.path.join(results_dir, 'hdf5')
        if not os.path.exists(hdf5_results_dir):
            os.makedirs(hdf5_results_dir)

        file_path = os.path.join(hdf5_results_dir,
                                 "window{0:03}.hdf5".format(self.window_ind))
        f = h5py.File(file_path, 'w')

        attrs_to_save = ['radius', 'lat', 'lon', 'acceptance_rates']

        for attr in attrs_to_save:
            f.create_dataset(attr, data=getattr(self, attr))
        f.close()


    @classmethod
    def from_hdf5(cls, results_dir, window_ind):

        saved_attrs = ['radius', 'lat', 'lon', 'acceptance_rates']

        file_path = os.path.join(results_dir, 'hdf5',
                                 "window{0:03}.hdf5".format(window_ind))

        f = h5py.File(file_path, 'r')

        kwargs = dict(window_ind=window_ind)
        for attr in saved_attrs:
            kwargs[attr] = f[attr][:]

        f.close()

        return cls(**kwargs)


    def plot_chains(self):

        n_spots = self.radius.shape[1]
        fig, ax = plt.subplots(n_spots, 3, figsize=(16, 8))
        n_bins = 30

        low = 4
        high = 96

        for i in range(self.radius.shape[1]):

            r_range = np.percentile(self.radius[self.burnin_int:, i], [low, high])
            lat_range = np.percentile(self.lat[self.burnin_int:, i], [low, high])
            lon_range = np.percentile(self.lon[self.burnin_int:, i], [low, high])
            ax[i, 0].hist(self.radius[self.burnin_int:, i], n_bins, color='k',
                          range=r_range)
            ax[i, 1].hist(self.lat[self.burnin_int:, i], n_bins, color='k',
                          range=lat_range)
            ax[i, 2].hist(self.lon[self.burnin_int:, i], n_bins, color='k',
                          range=lon_range)
            ax[i, 0].set_ylabel('Spot {0}'.format(i))
        ax[0, 0].set(title='Radius')
        ax[0, 1].set(title='Latitude')
        ax[0, 2].set(title='Longitude')
        ax[1, 0].set_xlabel('$R_s/R_\star$')
        ax[1, 1].set_xlabel('[radians]')
        ax[1, 2].set_xlabel('[radians]')
        fig.tight_layout()

    def plot_each_spot(self):
        #fig, ax = plt.subplots(5)
        n_spots = self.radius.shape[1]
        burn_in_to_index = int(self.burnin*self.radius.shape[0])
        for i in range(n_spots):
            samples = np.array([self.radius[burn_in_to_index:, i],
                                self.lon[burn_in_to_index:, i]]).T # self.lat[:, i],
            corner.corner(samples)

    def plot_star(self, fade_out=True):
        spots_spherical = SphericalRepresentation(self.lon*u.rad,
                                                  (self.lat - np.pi/2)*u.rad,
                                                  1*R_sun)
        self.spots_spherical = spots_spherical
        fig, ax = plot_star(spots_spherical, fade_out=fade_out)
        #plt.show()

    def plot_corner(self):
        exclude_columns = np.array([0, 1, 2, 3, 5, 8, 11, 14, 17])
        include_columns = np.ones(self.table.shape[1])
        include_columns[exclude_columns] = 0
        fig = corner.corner(self.table[:, include_columns.astype(bool)])#, fig=fig)
        return fig

    def plot_chi2(self):
        n_walkers = len((set(self.chain_ind)))
        fig, ax = plt.subplots(figsize=(12, 12))
        n_spots = self.radius.shape[1]

        for i in range(n_walkers):
            chain_i = self.chain_ind == i

            chi2 = self.table[chain_i, 3]
            #ax.semilogx(range(len(chi2)), chi2)
            ax.semilogx(range(len(chi2)), np.log10(chi2), alpha=0.6)
            ax.set(xlabel='Step', ylabel=r'$\log_{10} \, \chi^2$')



def plot_star(spots_spherical, fade_out=False):
    """
    Parameters
    ----------
    spots_spherical : `~astropy.coordinates.SphericalRepresentation`
        Points in spherical coordinates that represent the positions of the
        star spots.
    """
    oldrcparams = matplotlib.rcParams
    matplotlib.rcParams['font.size'] = 18
    fig, ax = plt.subplots(2, 3, figsize=(16, 16))

    positive_x = ax[0, 0]
    negative_x = ax[1, 0]

    positive_y = ax[0, 1]
    negative_y = ax[1, 1]

    positive_z = ax[0, 2]
    negative_z = ax[1, 2]

    axes = [positive_z, positive_x, negative_z, negative_x, positive_y, negative_y]
    axes_labels = ['+z', '+x', '-z', '-x', '+y', '-y']

    # Set black background
    plot_props = dict(xlim=(-1, 1), ylim=(-1, 1), xticks=(), yticks=())
    drange = np.linspace(-1, 1, 100)
    y = np.sqrt(1 - drange**2)
    bg_color = 'k'
    for axis in axes:
        axis.set(xticks=(), yticks=())
        axis.fill_between(drange, y, 1, color=bg_color)
        axis.fill_between(drange, -1, -y, color=bg_color)
        axis.set(**plot_props)
        axis.set_aspect('equal')

    # Set labels
    positive_x.set(xlabel='$\hat{z}$', ylabel='$\hat{y}$') # title='+x',
    positive_x.xaxis.set_label_position("top")
    positive_x.yaxis.set_label_position("right")

    negative_x.set(xlabel='$\hat{z}$', ylabel='$\hat{y}$') # title='-x',
    negative_x.xaxis.set_label_position("top")

    positive_y.set(xlabel='$\hat{z}$', ylabel='$\hat{x}$')
    positive_y.xaxis.set_label_position("top")

    negative_y.set(xlabel='$\hat{z}$', ylabel='$\hat{x}$')
    negative_y.xaxis.set_label_position("top")
    negative_y.yaxis.set_label_position("right")

    negative_z.set(xlabel='$\hat{y}$', ylabel='$\hat{x}$') # title='-z',
    negative_z.yaxis.set_label_position("right")
    positive_z.set(xlabel='$\hat{y}$', ylabel='$\hat{x}$') # title='+z',
    positive_z.yaxis.set_label_position("right")
    positive_z.xaxis.set_label_position("top")

    for axis, label in zip(axes, axes_labels):
        axis.annotate(label, (-0.9, 0.9), color='w', fontsize=14,
                      ha='center', va='center')

    # Plot gridlines
    n_gridlines = 9
    print("lat grid spacing: {0} deg".format(180./(n_gridlines-1)))
    n_points = 35
    pi = np.pi

    latitude_lines = SphericalRepresentation(np.linspace(0, 2*pi, n_points)[:, np.newaxis]*u.rad,
                                             np.linspace(-pi/2, pi/2, n_gridlines).T*u.rad,
                                             np.ones((n_points, 1))
                                             ).to_cartesian()

    longitude_lines = SphericalRepresentation(np.linspace(0, 2*pi, n_gridlines)[:, np.newaxis]*u.rad,
                                              np.linspace(-pi/2, pi/2, n_points).T*u.rad,
                                              np.ones((n_gridlines, 1))
                                              ).to_cartesian()

    for i in range(latitude_lines.shape[1]):
        for axis in [positive_z, negative_z]:
            axis.plot(latitude_lines.x[:, i], latitude_lines.y[:, i],
                      ls=':', color='silver')
        for axis in [positive_x, negative_x, positive_y, negative_y]:
            axis.plot(latitude_lines.y[:, i], latitude_lines.z[:, i],
                      ls=':', color='silver')

    for i in range(longitude_lines.shape[0]):
        for axis in [positive_z, negative_z]:
            axis.plot(longitude_lines.y[i, :], longitude_lines.x[i, :],
                    ls=':', color='silver')
        for axis in [positive_x, negative_x, positive_y, negative_y]:
            axis.plot(longitude_lines.y[i, :], longitude_lines.z[i, :],
                ls=':', color='silver')


    # Plot spots
    spots_cart = spots_spherical.to_cartesian()
    spots_x = spots_cart.x/R_sun
    spots_y = spots_cart.y/R_sun
    spots_z = spots_cart.z/R_sun

    if fade_out:
        n = float(len(spots_x))
        alpha_range = np.arange(n)

        alpha = (n - alpha_range)/n
    else:
        alpha = 0.5

    for spot_ind in range(spots_x.shape[1]):

        above_x_plane = spots_x[:, spot_ind] > 0
        above_y_plane = spots_y[:, spot_ind] > 0
        above_z_plane = spots_z[:, spot_ind] > 0
        below_x_plane = spots_x[:, spot_ind] < 0
        below_y_plane = spots_y[:, spot_ind] < 0
        below_z_plane = spots_z[:, spot_ind] < 0

        positive_x.plot(spots_y[above_x_plane, spot_ind],
                        spots_z[above_x_plane, spot_ind], '.', alpha=alpha)

        negative_x.plot(-spots_y[below_x_plane, spot_ind],
                        spots_z[below_x_plane, spot_ind], '.', alpha=alpha)

        positive_y.plot(-spots_x[above_y_plane, spot_ind],
                        spots_z[above_y_plane, spot_ind], '.', alpha=alpha)

        negative_y.plot(spots_x[below_y_plane, spot_ind],
                        spots_z[below_y_plane, spot_ind], '.', alpha=alpha)

        positive_z.plot(spots_x[above_z_plane, spot_ind],
                        spots_y[above_z_plane, spot_ind], '.', alpha=alpha)

        negative_z.plot(spots_x[below_z_plane, spot_ind],
                        -spots_y[below_z_plane, spot_ind], '.', alpha=alpha)
    matplotlib.rcParams = oldrcparams
    return fig, ax
