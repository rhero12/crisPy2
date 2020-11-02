import numpy as np
import os, yaml, h5py
from astropy.wcs import WCS
from scipy.io import readsav
from tqdm import tqdm

def memmap_crisp_cube(path):
    """
    This function memory maps a legacy La Palma data cube pulling metainformation from appropriate files. The function first looks for an ``assoc.pro`` file which contains important information about the first dimension. If this does not exist then the user will have to untangle the first dimension themselves. The first dimension is a multiplicative combination of time and wavelength (and Stokes) with the ``assoc.pro`` providing the necessary information to split these into distinct axes. When unable to access this information, the first dimension will be the multiplicative combination of time and wavelength (and Stokes) with the ordering of one time for all wavelengths sampled. N.B. when Stokes is present, the first axis takes the form of one time for wavelengths sampled of Stokes I, wavelengths sampled of Stokes Q etc.

    Parameters
    ----------
    path : str
        The path to the legacy La Palma cube.
    """

    if not os.path.exists(path):
        raise ValueError(f"No path found at {path}")

    folder, item = os.path.split(path)
    files = os.listdir(folder)
    assoc_name = ".".join(item.split[:-1]) + ".assoc.pro" #assoc file contains some meta information and the name usually follows similar convention to the La Palma cube without the file suffix

    if assoc_name in files:
        # Read out of the assoc file so no need to find meta information!!
        with open(folder+"/"+assoc_name) as f:
            assoc_pro = f.readlines() #read the file into variable
        dims = {}
        keys = ["nx", "ny", "nw", "nt"] #assoc file will contain at least the number of x points, y points, wavelength points ("nw") and time points

        for a in assoc_pro:
            if any(k in a for k in keys) and "=" in a and "assoc" not in a:
                key_val = a.split("=")
                dims[key_val[0]] = int(key_val[1])
        shape = [dims["nt"], dims["nw"], dims["ny"], dims["nx"]] #this is how the data is stored
        if "stokes" in item:
            shape.insert(1, 4)
    else:
        # Read directly from the cube (it is really good if there is an assoc pro file but this works *just* fine)
        header = np.memmap(path, np.uint8, mode="r")[:512].tostring() #this first 512 bytes of the La Palma cube are header information....
        entries = str(header).split()
        keys = ["nx", "ny", "nt"] #this meta information stores wavelength and time multiplied as the same axis which is why the assoc pro file is really useful
        dims = {}
        for e in entries:
            if any(k in e for k in keys) and "=" in e:
                key_val = e.stip(",").split("=")
                dims[key_val[0]] = key_val[1]

        shape = [dims["nt"], dims["ny"], dims["nx"]]

    if "icube" in item:
        dtype = np.int16
    elif "fcube" in item:
        dtype = np.float
    else:
        raise ValueError("Unknown cube type.")

    return np.memmap(path, offset=512, dtype=dtype, mode="r", shape=tuple(shape))

def hdf5_header_to_wcs(hdf5_header, nonu=False):
    """
    This function takes the hdf5 header information and converts it to a wcs object.

    Parameters
    ----------
    hdf5_header : dict or yaml
        The header information from the hdf5 observation file.
    nonu : bool, optional
        Whether or not the spectral points are sampled non-uniformly. Default is False.
    """

    if type(hdf5_header) is not dict:
        hdf5_header = yaml.load(hdf5_header[0], Loader=yaml.Loader)

    wcs_dict = {}

    wcs_dict["TELESCOP"] = hdf5_header["telescope"]
    wcs_dict["INSTRUME"] = hdf5_header["instrument"]
    wcs_dict["EXT_NAME"] = hdf5_header["element"]

    if len(hdf5_header["dimensions"]) == 4:
        # FITS axes are in the opposite order from numpy axes so the dimensions are read in reverse
        wcs_dict["NAXIS1"] = hdf5_header["dimensions"][-1]
        wcs_dict["NAXIS2"] = hdf5_header["dimensions"][-2]
        wcs_dict["NAXIS3"] = hdf5_header["dimensions"][-3]
        wcs_dict["NAXIS4"] = hdf5_header["dimensions"][-4]

        wcs_dict["CTYPE1"] = "HPLN-TAN"
        wcs_dict["CTYPE2"] = "HPLT-TAN"
        wcs_dict["CTYPE3"] = "WAVE"
        wcs_dict["CTYPE4"] = "STOKES"

        wcs_dict["CUNIT1"] = "arcsec"
        wcs_dict["CUNIT2"] = "arcsec"
        wcs_dict["CUNIT3"] = "Angstrom"

        wcs_dict["CRPIX1"] = hdf5_header["crpix"][-1]
        wcs_dict["CRPIX2"] = hdf5_header["crpix"][-2]
        wcs_dict["CRPIX3"] = hdf5_header["crpix"][-3]
        wcs_dict["CRPIX4"] = hdf5_header["crpix"][-4]

        wcs_dict["CRVAL1"] = hdf5_header["crval"][-1]
        wcs_dict["CRVAL2"] = hdf5_header["crval"][-2]
        wcs_dict["CRVAL3"] = hdf5_header["crval"][-3]
        wcs_dict["CRVAL4"] = 1.0

        wcs_dict["CDELT1"] = hdf5_header["pixel_scale"]
        wcs_dict["CDELT2"] = hdf5_header["pixel_scale"]
        if nonu:
            wcs_dict["CDELT3"] = 0.1
        else:
            wcs_dict["CDELT3"] = hdf5_header["wavel_scale"]
        wcs_dict["CDELT4"] = 1.0
    elif len(hdf5_header["dimensions"]) == 3:
        wcs_dict["NAXIS1"] = hdf5_header["dimensions"][-1]
        wcs_dict["NAXIS2"] = hdf5_header["dimensions"][-2]
        wcs_dict["NAXIS3"] = hdf5_header["dimensions"][-3]

        wcs_dict["CTYPE1"] = "HPLN-TAN"
        wcs_dict["CTYPE2"] = "HPLT-TAN"
        wcs_dict["CTYPE3"] = "WAVE"

        wcs_dict["CUNIT1"] = "arcsec"
        wcs_dict["CUNIT2"] = "arcsec"
        wcs_dict["CUNIT3"] = "Angstrom"

        wcs_dict["CRPIX1"] = hdf5_header["crpix"][-1]
        wcs_dict["CRPIX2"] = hdf5_header["crpix"][-2]
        wcs_dict["CRPIX3"] = hdf5_header["crpix"][-3]

        wcs_dict["CRVAL1"] = hdf5_header["crval"][-1]
        wcs_dict["CRVAL2"] = hdf5_header["crval"][-2]
        wcs_dict["CRVAL3"] = hdf5_header["crval"][-3]

        wcs_dict["CDELT1"] = hdf5_header["pixel_scale"]
        wcs_dict["CDELT2"] = hdf5_header["pixel_scale"]
        if nonu:
            wcs_dict["CDELT3"] = 0.1
        else:
            wcs_dict["CDELT3"] = hdf5_header["wavel_scale"]
    elif len(hdf5_header["dimensions"]) == 2:
        wcs_dict["NAXIS1"] = hdf5_header["dimensions"][-1]
        wcs_dict["NAXIS2"] = hdf5_header["dimensions"][-2]

        wcs_dict["CTYPE1"] = "HPLN-TAN"
        wcs_dict["CTYPE2"] = "HPLT-TAN"

        wcs_dict["CUNIT1"] = "arcsec"
        wcs_dict["CUNIT2"] = "arcsec"

        wcs_dict["CRPIX1"] = hdf5_header["crpix"][-1]
        wcs_dict["CRPIX2"] = hdf5_header["crpix"][-2]

        wcs_dict["CRVAL1"] = hdf5_header["crval"][-1]
        wcs_dict["CRVAL2"] = hdf5_header["crval"][-2]

        wcs_dict["CDELT1"] = hdf5_header["pixel_scale"]
        wcs_dict["CDELT2"] = hdf5_header["pixel_scale"]

    return WCS(wcs_dict)

def la_palma_cube_to_hdf5(cube_path, tseries_path, spectfile, date_obs, telescope, instrument, pixel_scale, cadence, element, pointing, mu=None, start_idx=0, save_dir="./"):
    """
    This is a function to save a La Palma legacy cube as hdf5 files.

    Parameters
    ----------
    cube_path : str
        The path to the legacy La Palma data cube.
    tseries_path : str
        The path to the ``tseries`` file containing the information on the times of the observations.
    spectfile : str
        The path to the file containing the spectral positions sampled.
    data_obs : str
        The observation date.
    telescope : str
        The telescope used.
    instrument : str
        The instrument used.
    pixel_scale : float
        The size of a detector pixel in arcseconds.
    cadence : float
        The time sampling of the observations.
    element : str
        The transition observed.
    pointing : tuple[float]
        The pointing of the centre of the images in the Helioprojective plane.
    mu : float or None, optional
        The direction cosine of the observations. Default is None.
    start_idx : int, optional
        The time index to start making files from. Default is 0.
    save_dir : str, optional
        The directory to save the hdf5 files in. Default is the current directory.
    """

    header = {}
    header["date-obs"] = date_obs
    header["telescope"] = telescope
    header["instrument"] = instrument
    header["pixel_scale"] = pixel_scale
    header["cadence"] = cadence
    header["element"] = element
    if mu is not None:
        header["mu"] = mu

    mem_cube = memmmap_crisp_cube(cube_path)
    times = readsav(tseries_path)["time"]

    if "wb" not in cube_path and mem_cube.ndim == 5:
        #this means there is Stokes
        header["ctype"] = ("STOKES", "WAVE", "HPLT-TAN", "HPLN-TAN")
        header["cunit"] = ("Angstrom", "arcsec", "arcsec")
        header["spect_pos"] = readsav(spectfile)["spect_pos"]
    elif "wb" not in cube_path and mem_cube.ndim == 4:
        header["ctype"] = ("WAVE", "HPLT-TAN", "HPLN-TAN")
        header["cunit"] = ("Angstrom", "arcsec", "arcsec")
        header["spect_pos"] = readsav(spectfile)["spect_pos"]
    else:
        header["ctype"] = ("HPLT-TAN", "HPLN-TAN")
        header["cunit"] = ("arcsec", "arcsec")

    for jj, frame in enumerate(tqdm(mem_cube)):
        header["time-obs"] = times[jj].deconde("utf-8")
        if "wb" not in cube_path and frame.ndim == 4:
            header["crpix"] = (1, np.median(np.arange(frame.shape[-3])), np.median(np.arange(frame.shape[-2])), np.median(np.arange(frame.shape[-1])))
            header["crval"] = (1, np.median(header["spect_pos"]), poitning[-2], pointing[-1])
            f = h5py.File(save_dir+str(start_idx).zfill(5)+".h5", "a")
            f["data"] = frame
            f.create_dataset("header", (1,), dtype=h5py.special_dtype(vlen=str))
            f["header"][0] = yaml.dump(header)
            f.close()
            start_idx += 1
        elif "wb" not in cube_path and frame.ndim == 3:
            header["crpix"] = (np.median(np.arange(frame.shape[-3])), np.median(np.arange(frame.shape[-2])), np.median(np.arange(frame.shape[-1])))
            header["crval"] = (np.median(header["spect_pos"]), poitning[-2], pointing[-1])
            f = h5py.File(save_dir+str(start_idx).zfill(5)+".h5", "a")
            f["data"] = frame
            f.create_dataset("header", (1,), dtype=h5py.special_dtype(vlen=str))
            f["header"][0] = yaml.dump(header)
            f.close()
            start_idx += 1
        else:
            header["crpix"] = (np.median(np.arange(frame.shape[-2])), np.median(np.arange(frame.shape[-1])))
            header["crval"] = (poitning[-2], pointing[-1])
            f = h5py.File(save_dir+str(start_idx).zfill(5)+".h5", "a")
            f["data"] = frame
            f.create_dataset("header", (1,), dtype=h5py.special_dtype(vlen=str))
            f["header"][0] = yaml.dump(header)
            f.close()
            start_idx += 1