import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.fft import fft2, fftfreq
from scipy.stats import binned_statistic
import html
from .utils import pt_bright

def integrated_intensity(intensity_vector,wavelengths, idx_range="all"):
    """
    A function to find the integrated intensity over a wavelength range of a spectral line.

    .. math::
       I = \int_{\Delta \lambda} I_{\lambda} ~d\lambda

    Parameters
    ----------
    intensity_vector : numpy.ndarray or crispy2.crisp.CRISP or crispy2.crisp.CRISPNonU
        The vector of spectral line intensities.
    wavelengths : numpy.ndarray
        The wavelengths to integrate over.
    idx_range : range, optional
        The range of indices to integrate over. Default is "all", integrates over the whole range.
    """

    if type(intensity_vector) != np.ndarray:
        intensity_vector = intensity_vector.file.data

    if idx_range == "all":
        return simps(intensity_vector, wavelengths)
    else:
        return simps(intensity_vector[idx_range], wavelengths[idx_range])

def intensity_ratio(I_1, I_2):
    """
    A function that calculates the intensity ratio of two previously integrated intensities.

    Parameters
    ----------
    I_1 : float
        The integrated intensity to be on the numerator of the ratio.
    I_2 : float
        The integrated intensity to be on the denominator of the ratio.
    """

    return I_1 / I_2

def doppler_vel(l, del_l=None):
    """
    A function to calculate the Doppler shift of a line core in km/s.

    .. math::
       v = \\frac{\Delta \lambda}{\lambda_{0}} c

    Parameters
    ----------
    l : float
        The rest wavelength of the transition.
    del_l : float, optional
        The difference between the observed wavelength and the rest wavelength of the transition.
    """

    return (del_l / l) * 3e5

def bar_lambda(intensity_vector, wavelengths):
    """
    Calculates the intensity-averaged line core.

    .. math::
       \\bar{\lambda} = \\frac{\int I(\lambda) ~\lambda ~d\lambda}{\int I (\lambda) ~ d\lambda}

    Parameters
    ----------
    intensity_vector : numpy.ndarray or crispy2.crisp.CRISP or crispy2.crisp.CRISPNonU
        The vector of spectral line intensities.
    wavelengths : numpy.ndarray
        The wavelengths to integrate over.
    """

    if type(intensity_vector) != np.ndarray:
        intensity_vector = intensity_vector.file.data

    num = simps(intensity_vector*wavelengths, wavelengths)
    den = simps(intensity_vector, wavelengths)

    return num / den

def variance(intensity_vector, wavelengths, bar_l=None):
    """
    Calculates the variance of the spectral line around the intensity-averaged line core.

    .. math::
       \sigma^{2} = \\frac{\int I (\lambda) ~ (\lambda - \\bar{\lambda})^{2} ~ d\lambda}{\int I (\lambda) ~ d\lambda}

    Parameters
    ----------
    intensity_vector : numpy.ndarray or crispy2.crisp.CRISP or crispy2.crisp.CRISPNonU
        The vector of spectral line intensities.
    wavelengths : numpy.ndarray
        The wavelengths to integrate over.
    bar_l : float or None, optional
        The intensity-averaged line core of the spectral line. Default is None will call ``crispy2.spectral.bar_lambda`` function on the ``intensity_vector`` and ``wavelengths`` argument to calculate.
    """

    if type(intensity_vector) != np.ndarray:
        intensity_vector = intensity_vector.file.data

    if bar_l == None:
        bar_l = bar_lambda(intensity_vector, wavelengths)

    num = simps(intensity_vector*(wavelengths-bar_l)**2, wavelengths)
    den = simps(intensity_vector, wavelengths)

    return num / den

def wing_idxs(intensity_vector, wavelengths, var=None, bar_l=None):
    """
    A function to work out the index range for the wings of the spectral line. This is working on the definition of wings that says the wings are defined as being one standard deviation away from the intensity-averaged line core.

    Parameters
    ----------
    intensity_vector : numpy.ndarray or crispy2.crisp.CRISP or crispy2.crisp.CRISPNonU
        The vector of spectral line intensities.
    wavelengths : numpy.ndarray
        The wavelengths to integrate over.
    var : float or None, optional
        The variance of the spectral line. Default is None will call ``crispy2.spectral.variance`` function on the ``intensity_vector`` and ``wavelengths`` argument to calculate.
    bar_l : float or None, optional
        The intensity-averaged line core of the spectral line. Default is None will call ``crispy2.spectral.bar_lambda`` function on the ``intensity_vector`` and ``wavelengths`` argument to calculate.
    """

    if type(intensity_vector) != np.ndarray:
        intensity_vector = intensity_vector.file.data

    if bar_l == None:
        bar_l = bar_lambda(intensity_vector, wavelengths)

    if var == None:
        var = variance(intensity_vector, wavelengths, bar_l=bar_l)

    blue_wing_start = 0 #blue wing starts at shortest wavelength
    red_wing_end = wavelengths.shape[0] - 1 #red wing ends at longest wavelength

    blue_end_wvl = bar_l - np.sqrt(var)
    red_start_wvl = bar_l + np.sqrt(var)

    blue_wing_end = np.argmin(np.abs(wavelengths - blue_end_wvl))
    red_wing_start = np.argmin(np.abs(wavelengths - red_start_wvl))

    return range(blue_wing_start, blue_wing_end+1), range(red_wing_start, red_wing_end+1)

def delta_lambda(wing_idxs, wavelengths):
    """
    Calculates the half-width in an intensity range i.e. there are N indices spanning a wing then the half-width is half of those indices multiplied by the change in wavelength between two entries. N.B. this is only the case when the space in the spectral axis is uniform.

    .. math::
       \delta \lambda = \\frac{N}{2} \\times (\Delta \lambda)

    where :math:`\Delta \lambda` is the difference in wavelength of two spectral points.

    Parameters
    ----------
    wing_idxs : range
        The indices of the spectral line corresponding to the wing of interest.
    wavelengths : numpy.ndarray
        The wavelengths to integrate over.
    """

    return len(wing_idxs) * (wavelengths[1] - wavelengths[0]) / 2

def lambda_0_wing(wing_idxs, wavelengths, d_lambda=None):
    """
    Calculates the centre wavelength of an intensity range.

    .. math::
       \lambda_{0, w} = \lambda_{\\text{end}, w} - \delta \lambda

    where :math:`\lambda_{\\text{end},w}` is the wavelength of the end of the wing.

    Parameters
    ----------
    wing_idxs : range
        The indices of the spectral line corresponding to the wing of interest.
    wavelengths : numpy.ndarray
        The wavelengths to integrate over.
    d_lambda : float, optional
        The half-width of the wing of the spectral line. Default is None will call ``crispy2.spectral.delta_lambda`` function on the ``wing_idxs`` and ``wavelengths`` arguments to calculate.
    """

    if d_lambda == None:
        d_lambda = delta_lambda(wing_idxs, wavelengths)

    return wavelengths[wing_idxs[-1]] - d_lambda

def interp_fine(spec_line):
    """
    Interpolates the spectral line onto a finer grid for more accurate calculations for the wing properties.

    Parameters
    ----------
    spec_line : tuple [numpy.ndarray]
        The spectral line to be interpolated onto a finer grid. This is a tuple of the intensities and the wavelengths in the order (wavelengths, intensities).
    """

    x, y = spec_line
    x_new = np.linspace(x[0], x[-1], num=1001)
    y_new = interp1d(x, y)(x_new)

    return np.array([x_new, y_new])

def power_spectrum(image, plot=True):
    """
    This function calculates the azimuthally-average power spectrum for an image.

    Parameters
    ----------
    image : numpy.ndarray
        The image to calculate the power spectrum for.
    plot : bool, optional
        Whether or not to plot the the power spectrum. Default is True.
    """

    nu = html.unescape("&nu;")
    h, w = image.shape

    #First calculate the Fourier transform of the image which represents the distribution of the image in frequency space
    ft = fft2(image)

    #Then the Fourier amplitudes for each pixel can be calculated
    fa = np.abs(ft)**2

    if h == w:
        nu_freq = fftfreq(h) * h
        nu_freq2D = np.meshgrid(nu_freq,nu_freq)

        nu_norm = np.sqrt(nu_freq2D[0]**2 + nu_freq2D[1]**2)

        nu_norm = nu_norm.flatten()
        fa = fa.flatten()

        nu_bins = np.arange(0.5, h//2, 1) #this is the limits of the spatial frequency bins
        nu_vals = 0.5 * (nu_bins[1:] + nu_bins[:-1]) #this calculates the midpoints of each spatial frequency bin

        Abins, _, _ = binned_statistic(nu_norm, fa, statistic="mean", bins=nu_bins)

        Abins *= 4*np.pi/3 * (nu_bins[1:]**3-nu_bins[:-1]**3) #the power in each bin is the average power from all cases of the spatial frequency multiplied by the volume of the bin (which in our case is just 1?)

        if plot:
            plt.figure()
            plt.loglog(nu_vals, Abins, c=pt_bright["blue"])
            plt.ylabel(f"P({nu})")
            plt.xlabel(fr"{nu} [px$^{-1}$]")
            plt.title("Power Spectrum of the Image")
            plt.show()

        return nu_vals, Abins
    else:
        nu_freqx = fftfreq(w) * w
        nu_freqy = fftfreq(h) * h
        nu_freq2D = np.meshgrid(nu_freqx,nu_freqy)

        nu_norm = np.sqrt(nu_freq2D[0]**2 + nu_freq2D[1]**2)

        nu_norm = nu_norm.flatten()
        fa = fa.flatten()

        if w < h:
            nu_bins = np.arange(0.5, w//2, 1)
        else:
            nu_bins = np.arange(0.5, h//2, 1)
        nu_vals = 0.5 * (nu_bins[1:] + nu_bins[:-1])

        Abins, _, _ = binned_statistic(nu_norm, fa, statistic="mean", bins=nu_bins)

        Abins *= 4*np.pi/3 * (nu_bins[1:]**3-nu_bins[:-1]**3)

        if plot:
            plt.figure()
            plt.loglog(nu_vals, Abins, c=pt_bright["blue"])
            plt.ylabel(f"P({nu})")
            plt.xlabel(fr"{nu} [px$^{-1}$]")
            plt.title("Power Spectrum of the Image")
            plt.show()

        return nu_vals, Abins