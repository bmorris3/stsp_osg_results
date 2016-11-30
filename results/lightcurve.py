from __future__ import absolute_import, print_function, division

from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import batman
from scipy import optimize
from glob import glob

__all__ = ["LightCurve", "BestLightCurve"]

class LightCurve(object):
    """
    Container object for light curves
    """
    def __init__(self, times=None, fluxes=None, errors=None, quarters=None, name=None):

        #if len(times) < 1:
        #    raise ValueError("Input `times` have no length.")

        if (isinstance(times[0], Time) and isinstance(times, np.ndarray)):
            times = Time(times)
        elif not isinstance(times, Time):
            times = Time(times, format='jd')

        self.times = times
        self.fluxes = fluxes
        if self.times is not None and errors is None:
            errors = np.zeros_like(self.fluxes) - 1
        self.errors = errors
        if self.times is not None and quarters is None:
            quarters = np.zeros_like(self.fluxes) - 1
        self.quarters = quarters
        self.name = name

    def plot(self, params, ax=None, quarter=None, show=True, phase=False,
             plot_kwargs={'color':'b', 'marker':'o', 'lw':0},
             ):
        """
        Plot light curve
        """
        if quarter is not None:
            if hasattr(quarter, '__len__'):
                mask = np.zeros_like(self.fluxes).astype(bool)
                for q in quarter:
                    mask |= self.quarters == q
            else:
                mask = self.quarters == quarter
        else:
            mask = np.ones_like(self.fluxes).astype(bool)

        if ax is None:
            ax = plt.gca()

        if phase:
            x = (self.times.jd - params.t0)/params.per % 1
            x[x > 0.5] -= 1
        else:
            x = self.times.jd

        ax.plot(x[mask], self.fluxes[mask],
                **plot_kwargs)
        ax.set(xlabel='Time' if not phase else 'Phase',
               ylabel='Flux', title=self.name)

        if show:
            plt.show()

    def save_to(self, path, overwrite=False, for_stsp=False):
        """
        Save times, fluxes, errors to new directory ``dirname`` in ``path``
        """
        dirname = self.name
        output_path = os.path.join(path, dirname)
        self.times = Time(self.times)

        if not for_stsp:
            if os.path.exists(output_path) and overwrite:
                shutil.rmtree(output_path)

            if not os.path.exists(output_path):
                os.mkdir(output_path)
                for attr in ['times_jd', 'fluxes', 'errors', 'quarters']:
                    np.savetxt(os.path.join(path, dirname, '{0}.txt'.format(attr)),
                            getattr(self, attr))

        else:
            if not os.path.exists(output_path) or overwrite:
                attrs = ['times_jd', 'fluxes', 'errors']
                output_array = np.zeros((len(self.fluxes), len(attrs)), dtype=float)
                for i, attr in enumerate(attrs):
                    output_array[:, i] = getattr(self, attr)
                np.savetxt(os.path.join(path, dirname+'.txt'), output_array)

    @classmethod
    def from_raw_fits(cls, fits_paths, name=None):
        """
        Load FITS files from MAST into the LightCurve object
        """
        fluxes = []
        errors = []
        times = []
        quarter = []

        for path in fits_paths:
            data = fits.getdata(path)
            header = fits.getheader(path)
            times.append(data['TIME'] + 2454833.0)
            errors.append(data['PDCSAP_FLUX_ERR'])
            fluxes.append(data['PDCSAP_FLUX'])
            quarter.append(len(data['TIME'])*[header['QUARTER']])

        times, fluxes, errors, quarter = [np.concatenate(i)
                                          for i in [times, fluxes, errors, quarter]]

        mask_nans = np.zeros_like(fluxes).astype(bool)
        for attr in [times, fluxes, errors]:
            mask_nans |= np.isnan(attr)

        times, fluxes, errors, quarter = [attr[-mask_nans]
                                           for attr in [times, fluxes, errors, quarter]]

        return LightCurve(times, fluxes, errors, quarters=quarter, name=name)

    @classmethod
    def from_dir(cls, path, for_stsp=False):
        """Load light curve from numpy save files in ``dir``"""
        if not for_stsp:
            times, fluxes, errors, quarters = [np.loadtxt(os.path.join(path, '{0}.txt'.format(attr)))
                                               for attr in ['times_jd', 'fluxes', 'errors', 'quarters']]
        else:
            quarters = None
            times, fluxes, errors = np.loadtxt(path, unpack=True)

        if os.sep in path:
            name = path.split(os.sep)[-1]
        else:
            name = path

        if name.endswith('.txt'):
            name = name[:-4]

        return cls(times, fluxes, errors, quarters=quarters, name=name)

    def normalize_each_quarter(self, rename=None, polynomial_order=2, plots=False):
        """
        Use 2nd order polynomial fit to each quarter to normalize the data
        """
        quarter_inds = list(set(self.quarters))
        quarter_masks = [quarter == self.quarters for quarter in quarter_inds]

        for quarter_mask in quarter_masks:

            polynomial = np.polyfit(self.times[quarter_mask].jd,
                                    self.fluxes[quarter_mask], polynomial_order)
            scaling_term = np.polyval(polynomial, self.times[quarter_mask].jd)
            self.fluxes[quarter_mask] /= scaling_term
            self.errors[quarter_mask] /= scaling_term

            if plots:
                plt.plot(self.times[quarter_mask], self.fluxes[quarter_mask])
                plt.show()

        if rename is not None:
            self.name = rename

    def mask_out_of_transit(self, params=None, oot_duration_fraction=0.25, flip=False):
        """
        Mask out the out-of-transit light curve based on transit parameters
        """
        # Fraction of one duration to capture out of transit

        phased = (self.times.jd - params.t0) % params.per
        near_transit = ((phased < params.duration*(0.5 + oot_duration_fraction)) |
                        (phased > params.per - params.duration*(0.5 + oot_duration_fraction)))
        if flip:
            near_transit = -near_transit
        sort_by_time = np.argsort(self.times[near_transit].jd)
        return dict(times=self.times[near_transit][sort_by_time],
                    fluxes=self.fluxes[near_transit][sort_by_time],
                    errors=self.errors[near_transit][sort_by_time],
                    quarters=self.quarters[near_transit][sort_by_time])

    def mask_in_transit(self, params=None, oot_duration_fraction=0.25):
        return self.mask_out_of_transit(params=params, oot_duration_fraction=oot_duration_fraction,
                                        flip=True)

    def get_transit_light_curves(self, params, plots=False):
        """
        For a light curve with transits only (returned by get_only_transits),
        split up the transits into their own light curves, return a list of
        `TransitLightCurve` objects
        """
        time_diffs = np.diff(sorted(self.times.jd))
        diff_between_transits = params.per/2.
        split_inds = np.argwhere(time_diffs > diff_between_transits) + 1

        if len(split_inds) > 1:

            split_ind_pairs = [[0, split_inds[0][0]]]
            split_ind_pairs.extend([[split_inds[i][0], split_inds[i+1][0]]
                                     for i in range(len(split_inds)-1)])
            split_ind_pairs.extend([[split_inds[-1], len(self.times)]])

            transit_light_curves = []
            counter = -1
            for start_ind, end_ind in split_ind_pairs:
                counter += 1
                if plots:
                    plt.plot(self.times.jd[start_ind:end_ind],
                             self.fluxes[start_ind:end_ind], '.-')

                parameters = dict(times=self.times[start_ind:end_ind],
                                  fluxes=self.fluxes[start_ind:end_ind],
                                  errors=self.errors[start_ind:end_ind],
                                  quarters=self.quarters[start_ind:end_ind],
                                  name=counter)
                transit_light_curves.append(TransitLightCurve(**parameters))
            if plots:
                plt.show()
        else:
            transit_light_curves = []

        return transit_light_curves

    def get_available_quarters(self):
        return list(set(self.quarters))

    def get_quarter(self, quarter):
        this_quarter = self.quarters == quarter
        return LightCurve(times=self.times[this_quarter],
                          fluxes=self.fluxes[this_quarter],
                          errors=self.errors[this_quarter],
                          quarters=self.quarters[this_quarter],
                          name=self.name + '_quarter_{0}'.format(quarter))

    @property
    def times_jd(self):
        return self.times.jd

    def save_split_at_stellar_rotations(self, path, stellar_rotation_period,
                                        overwrite=False):
        dirname = self.name
        output_path = os.path.join(path, dirname)
        self.times = Time(self.times)

        if os.path.exists(output_path) and overwrite:
            shutil.rmtree(output_path)

        stellar_rotation_phase = ((self.times.jd - self.times.jd[0])*u.day %
                                   stellar_rotation_period ) / stellar_rotation_period
        phase_wraps = np.argwhere(stellar_rotation_phase[:-1] >
                                  stellar_rotation_phase[1:])

        split_times = np.split(self.times.jd, phase_wraps)
        split_fluxes = np.split(self.fluxes, phase_wraps)
        split_errors = np.split(self.errors, phase_wraps)
        split_quarters = np.split(self.quarters, phase_wraps)

        header = "JD Flux Uncertainty Quarter"
        for i, t, f, e, q in zip(range(len(split_times)), split_times,
                                 split_fluxes, split_errors, split_quarters):
            np.savetxt(os.path.join(output_path, 'rotation{:02d}.txt'.format(i)),
                       np.vstack([t, f, e, q]).T, header=header)


class BestLightCurve(object):
    def __init__(self, path=None, transit_params=None, times=None, fluxes_kepler=None,
                 errors=None, fluxes_model=None, flags=None):
        self.path = path
        self.default_figsize = (10, 8)#(20, 8)

        if path is not None:
            times, fluxes_kepler, errors, fluxes_model, flags = np.loadtxt(path,
                                                                           unpack=True)

        self.times = Time(times if times.mean() > 2450000 else times + 2454833., format='jd')
        self.fluxes_kepler = fluxes_kepler
        self.errors = errors
        self.fluxes_model = fluxes_model
        self.flags = flags

        self.kepler_lc = LightCurve(times=self.times, fluxes=fluxes_kepler,
                                    errors=errors)
        self.model_lc = LightCurve(times=self.times, fluxes=fluxes_model)
        self.transit_params = transit_params

    def plot_whole_lc(self):

        # Whole light curve

        import seaborn as sns
        sns.set(style='white')

        errorbar_color = '#b3b3b3'
        fontsize = 16

        fig, ax = plt.subplots(2, 1, figsize=self.default_figsize,
                               sharex='col')
        ax[0].errorbar(self.kepler_lc.times.plot_date, self.fluxes_kepler,
                        self.kepler_lc.errors, fmt='.',
                        color='k', ecolor=errorbar_color, capsize=0, label='Kepler')
        ax[0].plot(self.model_lc.times.plot_date, self.fluxes_model, 'r', label='STSP')
        ax[0].set_ylabel('Flux', fontsize=fontsize)

        ax[1].errorbar(self.kepler_lc.times.plot_date,
                       self.fluxes_kepler - self.fluxes_model, self.kepler_lc.errors,
                       fmt='.', color='k', ecolor=errorbar_color, capsize=0)
        ax[1].set_ylabel('Residuals', fontsize=fontsize)
        ax[1].axhline(0, color='r')

        label_times = Time(ax[1].get_xticks(), format='plot_date')
        ax[1].set_xticklabels([lt.strftime("%H:%M") for lt in label_times.datetime])

        ax[1].set_xlabel('Time on {0} UTC'.format(label_times[0].datetime.date()), fontsize=fontsize)

        ax[1].set_xlim([self.kepler_lc.times.plot_date.min(),
                        self.kepler_lc.times.plot_date.max()])

        sns.despine()

        return fig, ax

    def plot_transit(self):

        # Whole light curve

        import seaborn as sns
        sns.set(style='white')

        errorbar_color = '#b3b3b3'
        fontsize = 16

        fig, ax = plt.subplots(1, figsize=(8, 5))
        ax.errorbar(self.kepler_lc.times.plot_date, self.fluxes_kepler,
                        self.kepler_lc.errors, fmt='.',
                        color='k', ecolor=errorbar_color, capsize=0, label='Kepler')
        ax.plot(self.model_lc.times.plot_date, self.fluxes_model, 'r', label='STSP')

        label_times = Time(ax.get_xticks(), format='plot_date')
        ax.set_xticklabels([lt.strftime("%H:%M") for lt in label_times.datetime])

        ax.set_xlabel('Time on {0} UTC'.format(label_times[0].datetime.date()),
                      fontsize=fontsize)
        ax.set_ylabel('Flux', fontsize=fontsize)
        ax.set_xlim([self.kepler_lc.times.plot_date.min(),
                        self.kepler_lc.times.plot_date.max()])

        sns.despine()

        return fig, ax

    def plot_transits(self):

        if self.transit_params is not None:
            kepler_transits = LightCurve(**self.kepler_lc.mask_out_of_transit(params=self.transit_params)
                                         ).get_transit_light_curves(params=self.transit_params)
            model_transits = LightCurve(**self.model_lc.mask_out_of_transit(params=self.transit_params)
                                        ).get_transit_light_curves(params=self.transit_params)

        else:
            kepler_transits = LightCurve(**self.kepler_lc.mask_out_of_transit()
                                         ).get_transit_light_curves()
            model_transits = LightCurve(**self.model_lc.mask_out_of_transit()
                                        ).get_transit_light_curves()

        # Whole light curve

        if len(kepler_transits) > 0:
            fig, ax = plt.subplots(2, len(kepler_transits), figsize=self.default_figsize,
                                   sharex='col')
            scale_factor = 0.4e-6
            for i in range(len(kepler_transits)):
                ax[0, i].plot_date(kepler_transits[i].times.plot_date,
                                   scale_factor*kepler_transits[i].fluxes,
                              'k.', label='Kepler')
                ax[0, i].plot_date(model_transits[i].times.plot_date,
                                   scale_factor*model_transits[i].fluxes,
                              'r', label='STSP')
                ax[0, i].set(yticks=[])
                ax[1, i].axhline(0, color='r', lw=2)

                ax[1, i].plot_date(kepler_transits[i].times.plot_date,
                                   scale_factor*(kepler_transits[i].fluxes -
                                            model_transits[i].fluxes), 'k.')
                xtick = Time(kepler_transits[i].times.jd.mean(), format='jd')
                #ax[1, i].set(xticks=[xtick.plot_date], xticklabels=[xtick.iso.split('.')[0]],
                #             yticks=[])

                #ax[1, i].set(xlabel='Time')

            #ax[0, 0].legend(loc='lower left')
            ax[0, 0].set(ylabel=r'Flux')
            ax[1, 0].set(ylabel=r'Residuals')
            fig.tight_layout()
            return fig, ax
        else:
            return None, None
