import numpy as np
import matplotlib.pyplot as plt
import h5py, yaml
from matplotlib.colors import SymLogNorm
from astropy.wcs import WCS
import astropy.units as u
from .mixin import InversionSlicingMixin
from .utils import ObjDict

class Inversion(InversionSlicingMixin):
    """
    Class for transporting and using the inversions obtained from RADYNVERSION.

    :param filename: The file of the inversion. Can be either an hdf5 file path or an ObjDict.
    :type filename: str or ObjDict
    :param z: The height grid that the atmospheric parameters are calculated at. This can be either an hdf5 file path or a numpy.ndarray.
    :type z: str or numpy.ndarray
    :param header: The header information of the associated observation.
    :type header: dict or None

    :cvar ne: The electron number density estimated by RADYNVERSION. This is the median solution for a certain number of draws from the latent space.
    :cvar temp: The electron temperature estimated by RADYNVERSION. This is the median solution for a certain number of draws from the latent space.
    :cvar vel: The bulk velocity flow estimated by RADYNVERSION. This is the median solution for a certain number of draws from the latent space.
    :cvar err: This contains the median absolute deviation (MAD, standard error on the median) for each estimated parameter giving a sense of confidence intervals.
    :cvar wcs: The WCS from the observartion associated with the inversion.
    :cvar z: The height grid the inversions are estimated on.
    :cvar header: The header information from the observation associated with the inversion.
    """
    def __init__(self, filename, z, header, wcs=None):
        if type(filename) == str:
            self.f = h5py.File(filename, "r")
            if type(z) == str:
                self.z = h5py.File(z, "r").get("z")
            else:
                self.z = z
            if wcs == None:
                self.wcs = self._inversion_wcs(header)
            else:
                self.wcs = wcs
            self.header = header
        elif type(filename) == ObjDict:
            self.f = filename
            self.wcs = wcs
            self.z = z
            self.header = header

    @property
    def ne(self):
        if type(self.f) == ObjDict:
            return self.f["ne"]
        else:
            return self.f["/atmos/ne"]

    @property
    def temp(self):
        if type(self.f) == ObjDict:
            return self.f["temperature"]
        else:
            return self.f["/atmos/temperature"]

    @property
    def vel(self):
        if type(self.f) == ObjDict:
            return self.f["vel"]
        else:
            return self.f["/atmos/vel"]

    @property
    def ne_err(self):
        if type(self.f) == ObjDict:
            return self.f["ne_err"]
        else:
            return self.f["/atmos/ne_err"]

    @property
    def temp_err(self):
        if type(self.f) == ObjDict:
            return self.f["temperature_err"]
        else:
            return self.f["/atmos/temperature_err"]

    @property
    def vel_err(self):
        if type(self.f) == ObjDict:
            return self.f["vel_err"]
        else:
            return self.f["/atmos/vel_err"]

    def __str__(self):
        try :       
            time = self.header["DATE-AVG"][-12:]
            date = self.header["DATE-AVG"][:-13]
            pointing_x = str(self.header["CRVAL1"])
            pointing_y = str(self.header["CRVAL2"])
        except KeyError:
            time = self.header["time-obs"]
            date = self.header["date-obs"]
            pointing_x = str(self.header["crval"][-1])
            pointing_y = str(self.header["crval"][-2])

        return f"""Inversion
        ------------------
        {date} {time}

        Pointing: ({pointing_x}, {pointing_y})"""

    def _inversion_wcs(self, header):

        wcs_dict = {}

        try:
            wcs_dict["NAXIS1"] = header["NAXIS1"]
            wcs_dict["NAXIS2"] = header["NAXIS2"]
            wcs_dict["NAXIS3"] = self.z.shape[0]

            wcs_dict["CTYPE1"] = "HPLN-TAN"
            wcs_dict["CTYPE2"] = "HPLT-TAN"
            wcs_dict["CTYPE3"] = "HEIGHT"

            wcs_dict["CUNIT1"] = "arcsec"
            wcs_dict["CUNIT2"] = "arcsec"
            wcs_dict["CUNIT3"] = "Mm"

            wcs_dict["CRPIX1"] = header["CRPIX1"]
            wcs_dict["CRPIX2"] = header["CRPIX2"]
            wcs_dict["CRPIX3"] = self.z.shape[0] // 2

            wcs_dict["CRVAL1"] = header["CRVAL1"]
            wcs_dict["CRVAL2"] = header["CRVAL2"]
            wcs_dict["CRVAL3"] = self.z[self.z.shape[0] // 2]

            wcs_dict["CDELT1"] = header["CDELT1"]
            wcs_dict["CDELT2"] = header["CDELT2"]
            wcs_dict["CDELT3"] = 1.0 # z is sampled non-uniformly
        except KeyError:
            wcs_dict["NAXIS1"] = header["dimensions"][-1]
            wcs_dict["NAXIS2"] = header["dimensions"][-2]
            wcs_dict["NAXIS3"] = self.z.shape[0]

            wcs_dict["CTYPE1"] = "HPLN-TAN"
            wcs_dict["CTYPE2"] = "HPLT-TAN"
            wcs_dict["CTYPE3"] = "HEIGHT"

            wcs_dict["CUNIT1"] = "arcsec"
            wcs_dict["CUNIT2"] = "arcsec"
            wcs_dict["CUNIT3"] = "Mm"

            wcs_dict["CRPIX1"] = header["crpix"][-1]
            wcs_dict["CRPIX2"] = header["crpix"][-2]
            wcs_dict["CRPIX3"] = self.z.shape[0] // 2

            wcs_dict["CRVAL1"] = header["crval"][-1]
            wcs_dict["CRVAL2"] = header["crval"][-2]
            wcs_dict["CRVAL3"] = self.z[self.z.shape[0] // 2]

            wcs_dict["CDELT1"] = header["pixel_scale"]
            wcs_dict["CDELT2"] = header["pixel_scale"]
            wcs_dict["CDELT3"] = 1.0 # z is sampled non-uniformly

        return WCS(wcs_dict)


    def plot_ne(self, eb=False):
        """
        Class method to plot the electron number density for a given location within the field-of-view. This works by slicing the ``Inversion`` object.

        Parameters
        ----------
        eb : bool, optional
            Whether or not to plot the median absolute deviation (MAD) for the electron number density as errorbars. Default is False.
        """
        plt.style.use("bmh")
        point = [np.round(x << u.arcsec, decimals=2).value for x in self.wcs.low_level_wcs._wcs[0].array_index_to_world(*self.ind[-2:])]
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]

            title = f"{datetime} ({point[0]},{point[1]})"
        else:
            title = f"({point[0]}, {point[1]})"
        fig = plt.figure()
        ax1 = fig.gca()
        if eb:
            ax1.errorbar(self.z, self.ne, yerr=self.mad[0], capsize=3)
        else:
            ax1.plot(self.z, self.ne)
        ax1.set_ylabel(r"log$_{10}$ n$_{\text{e}}$ \[cm$^{-3}$\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title(f"Electron Number Density {title}")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_temp(self, eb=False):
        """
        Class method to plot the electron temperature for a given point in the field-of-view. This is done by slicing the ``Inversion`` object.

        Parameters
        ----------
        eb : bool, optional
            Whether or not to plot the median absolute deviation (MAD) of the estimated electron temperatures as errorbars. Default is False.
        """
        plt.style.use("bmh")
        point = [np.round(x << u.arcsec, decimals=2).value for x in self.wcs.low_level_wcs._wcs[0].array_index_to_world(*self.ind[-2:])]
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]

            title = f"{datetime} ({point[0]},{point[1]})"
        else:
            title = f"({point[0]}, {point[1]})"
        fig = plt.figure()
        ax1 = fig.gca()
        if eb:
            ax1.errorbar(self.z, self.temp, yerr=self.mad[1], capsize=3)
        else:
            ax1.plot(self.z, self.temp)
        ax1.set_ylabel(r"log$_{10}$ T \[K\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title(f"Electron Temperature {title}")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_vel(self, eb=False):
        """
        Class method to plot the bulk velocity for a certain point within the field-of-view. This is done using a slice of the ``Inversion`` instance.

        Parameters
        ----------
        eb : bool, optional
            Whether or not to plot the median absolute deviation (MAD) of the bulk velocity as errorbars. Default is False.
        """
        plt.style.use("bmh")
        point = [np.round(x << u.arcsec, decimals=2).value for x in self.wcs.low_level_wcs._wcs[0].array_index_to_world(*self.ind[-2:])]
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]

            title = f"{datetime} ({point[0]},{point[1]})"
        else:
            title = f"({point[0]}, {point[1]})"
        fig = plt.figure()
        ax1 = fig.gca()
        if eb:
            ax1.errorbar(self.z, self.vel, yerr=self.mad[2], capsize=3)
        else:
            ax1.plot(self.z, self.vel)
        ax1.set_ylabel(r"Bulk Plasma Flow \[km s$^{-1}$\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title(f"Bulk Plasma Flow {title}")
        ax1.tick_params(direction="in")
        fig.show()

    def plot_params(self, eb=False):
        """
        Class method to plot the electron number density, electron temperature, and bulk velocity for a certain point within the field-of-view. This is done using a slice of the ``Inversion`` instance.

        Parameters
        ----------
        eb : bool, optional
            Whether or not to plot the median absolute deviation (MAD) for each estimated quantity as errorbars. Default is False.
        """
        plt.style.use("bmh")
        point = [np.round(x << u.arcsec, decimals=2).value for x in self.wcs.low_level_wcs._wcs[0].array_index_to_world(*self.ind[-2:])]
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]

            title = f"{datetime} ({point[0]},{point[1]})"
        else:
            title = f"({point[0]}, {point[1]})"
        fig = plt.figure()
        fig.suptitle(title)
        ax1 = fig.add_subplot(1, 3, 1)
        if self.eb:
            ax1.errorbar(self.z, self.ne, yerr=self.mad[0], capsize=3)
        else:
            ax1.plot(self.z, self.ne)
        ax1.set_ylabel(r"log$_{10}$ n$_{e}$ \[cm$^{-3}$\]")
        ax1.set_xlabel("z [Mm]")
        ax1.set_title("Electron Number Density")
        ax1.tick_params(direction="in")

        ax2 = fig.add_subplot(1, 3, 2)
        if self.eb:
            ax2.errorbar(self.z, self.temp, yerr=self.mad[1], capsize=3)
        else:
            ax2.plot(self.z, self.temp)
        ax2.set_ylabel(r"log$_{10}$ T \[K\]")
        ax2.set_xlabel("z [Mm]")
        ax2.set_title("Electron Temperature")
        ax2.tick_params(direction="in")

        ax3 = fig.add_subplot(1, 3, 3)
        if self.eb:
            ax3.errorbar(self.z, self.vel, yerr=self.mad[2], capsize=3)
        else:
            ax3.plot(self.z, self.vel)
        ax3.set_ylabel(r"Bulk Plasma Flow \[km s$^{-1}\]")
        ax3.set_xlabel("z [Mm]")
        ax3.set_title("Bulk Plasma Flow")
        ax3.tick_params(direction="in")
        fig.show()

    def ne_map(self, frame=None):
        """
        Creates an electron density map at a specified height denoted in the ``Inversion`` slice.

        Parameters
        ----------
        frame : str, optional
            The frame to plot the map in. Default is None therefore uses the WCS frame. Other option is "pix" to plot in the pixel frame.
        """
        plt.style.use("bmh")
        if type(self.ind) == int:
            idx = self.ind
        else:
            idx = self.ind[-1]
        height = np.round(self.z[idx], decimals=4)
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]
        else:
            datetime = ""

        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
            im1 = ax1.imshow(self.ne, cmap="cividis")
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(f"Electron Number Density {datetime} z={height}Mm")
            fig.colorbar(im1, ax=ax1, label=r"log$_{10}$n$_{e}$ [cm$^{-3}$]")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.gca()
            im1 = ax1.imshow(self.ne, cmap="cividis")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(f"Electron Number Density {datetime} z={height}Mm")
            fig.colorbar(im1, ax=ax1, label=r"log$_{10}$n$_{e}$ [cm$^{-3}$]")
            fig.show()

    def temp_map(self, frame=None):
        """
        Creates an electron temperature map at a specified height denoted in the ``Inversion`` slice.

        Parameters
        ----------
        frame : str, optional
            The frame to plot the map in. Default is None therefore uses the WCS frame. Other option is "pix" to plot in the pixel frame.
        """
        plt.style.use("bmh")
        if type(self.ind) == int:
            idx = self.ind
        else:
            idx = self.ind[-1]
        height = np.round(self.z[idx], decimals=4)
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]
        else:
           datetime = ""
        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
            im1 = ax1.imshow(self.temp, cmap="hot")
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(f"Electron Temperature {datetime} z={height}Mm")
            fig.colorbar(im1, ax=ax1, label=r"log$_{10}$T [K]")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.gca()
            im1 = ax1.imshow(self.temp, cmap="cividis")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(f"Electron Temperature {datetime} z={height}Mm")
            fig.colorbar(im1, ax=ax1, label=r"log$_{10}$T [K]")
            fig.show()

    def vel_map(self, frame=None):
        """
        Creates a bulk velocity map at a specified height denoted in the ``Inversion`` slice.

        Parameters
        ----------
        frame : str, optional
            The frame to plot the map in. Default is None therefore uses the WCS frame. Other option is "pix" to plot in the pixel frame.
        """
        plt.style.use("bmh")
        if type(self.ind) == int:
            idx = self.ind
        else:
            idx = self.ind[-1]
        height = np.round(self.z[idx], decimals=4)
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]
        else:
            datetime = ""
        if frame is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
            im1 = ax1.imshow(self.vel, cmap="RdBu", norm=SymLogNorm(1))
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title(f"Bulk Velocity Flow {datetime} z={height}Mm")
            fig.colorbar(im1, ax=ax1, label=r"v [kms$^{-1}$]")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.gca()
            im1 = ax1.imshow(self.vel, cmap="RdBu", norm=SymLogNorm(1))
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title(f"Bulk Velocity Flow {datetime} z={height}Mm")
            fig.colorbar(im1, ax=ax1, label=r"v [kms$^{-1}$]")
            fig.show()

    def params_map(self, frame=None):
        """
        Creates maps of electron number density, electron temperature, and bulk velocity at a specified height denoted in the ``Inversion`` slice.

        Parameters
        ----------
        frame : str, optional
            The frame to plot the map in. Default is None therefore uses the WCS frame. Other option is "pix" to plot in the pixel frame.
        """
        plt.style.use("bmh")
        if type(self.ind) == int:
            idx = self.ind
        else:
            idx = self.ind[-1]
        height = np.round(self.z[idx], decimals=4)
        if self.header is not None:
            try:
                datetime = self.header["DATE-AVG"]
            except KeyError:
                datetime = self.header["date-obs"] + "T" + self.header["time-obs"]
        else:
            datetime = ""
        if frame is None:
            fig = plt.figure()
            fig.suptitle(f"{datetime} z={np.round(height,3)}Mm")
            ax1 = fig.add_subplot(1, 3, 1, projection=self.wcs.low_level_wcs)
            im1 = ax1.imshow(self.ne, cmap="cividis")
            ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            ax1.set_title("Electron Number Density")
            fig.colorbar(im1, ax=ax1, orientation="horizontal", label=r"log$_{10}$n$_{e}$ [cm$^{-3}$]")

            ax2 = fig.add_subplot(1, 3, 2, projection=self.wcs.low_level_wcs)
            im2 = ax2.imshow(self.temp, cmap="hot")
            ax2.set_ylabel("Helioprojective Latitude [arcsec]")
            ax2.set_xlabel("Helioprojective Longitude [arcsec]")
            ax2.set_title("Electron Temperature")
            fig.colorbar(im2, ax=ax2, orientation="horizontal", label=r"log$_{10}$T [K]")

            ax3 = fig.add_subplot(1, 3, 3, projection=self.wcs.low_level_wcs)
            im3 = ax3.imshow(self.vel, cmap="RdBu", norm=SymLogNorm(1))
            ax3.set_ylabel("Helioprojective Latitude [arcsec]")
            ax3.set_xlabel("Helioprojective Longitude [arcsec]")
            ax3.set_title("Bulk Velocity Flow")
            fig.colorbar(im3, ax=ax3, orientation="horizontal", label=r"v [kms$^{-1}$]")
            fig.show()
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            im1 = ax1.imshow(self.ne, cmap="cividis")
            ax1.set_ylabel("y [pixels]")
            ax1.set_xlabel("x [pixels]")
            ax1.set_title("Electron Number Density")
            fig.colorbar(im1, ax=ax1, orientation="horizontal", label=r"log$_{10}$n$_{e}$ [cm$^{-3}$]")

            ax2 = fig.add_subplot(1, 3, 2)
            im2 = ax2.imshow(self.temp, cmap="hot")
            ax2.set_ylabel("y [pixels]")
            ax2.set_xlabel("x [pixels]")
            ax2.set_title("Electron Temperature")
            fig.colorbar(im2, ax=ax2, orientation="horizontal", label=r"log$_{10}$T [K]")

            ax3 = fig.add_subplot(1, 3, 3)
            im3 = ax3.imshow(self.vel, cmap="RdBu", norm=SymLogNorm(1))
            ax3.set_ylabel("y [pixels]")
            ax3.set_xlabel("x [pixels]")
            ax3.set_title("Bulk Velocity Flow")
            fig.colorbar(im3, ax=ax3, orientation="horizontal", label=r"v [kms$^{-1}$]")
            fig.show()


    def to_lonlat(self, y, x):
        """
        This function will take a y, x coordinate in pixel space and map it to Helioprojective Longitude, Helioprojective Latitude according to the transform in the WCS. This will return the Helioprojective coordinates in units of arcseconds. Note this function takes arguments in the order of numpy indexing (y,x) but returns a pair longitude/latitude which is Solar-X, Solar-Y.

        Parameters
        ----------
        y : int
            The y-index to be converted to Helioprojective Latitude.
        x : int
            The x-index to be converted to Helioprojective Longitude.
        """
        if len(self.wcs.low_level_wcs.array_shape) == 4:
            if hasattr(self, "ind"):
                if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2],self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2]].array_index_to_world(y,x) << u.arcsec
                elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,:,self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                else:
                    return self.wcs.low_level_wcs._wcs[0,0].array_index_to_world(y,x) << u.arcsec
            else:
                return self.wcs[0,0].array_index_to_world(y,x) << u.arcsec
        elif len(self.wcs.low_level_wcs.array_shape) == 3:
            if hasattr(self, "ind") and self.wcs.low_level_wcs._wcs.naxis == 4:
                if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2],self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2]].array_index_to_world(y,x) << u.arcsec
                elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,:,self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                else:
                    return self.wcs.low_level_wcs._wcs[0,0].array_index_to_world(y,x) << u.arcsec
            else:
                if hasattr(self, "ind"):
                    if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                        return self.wcs.low_level_wcs._wcs[0,self.ind[-2],self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                    elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                        return self.wcs.low_level_wcs._wcs[0,self.ind[-2]].array_index_to_world(y,x) << u.arcsec
                    elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                        return self.wcs.low_level_wcs._wcs[0,:,self.ind[-1]].array_index_to_world(y,x) << u.arcsec
                    else:
                        return self.wcs.low_level_wcs._wcs[0].array_index_to_world(y,x) << u.arcsec
                else:
                    return self.wcs[0].array_index_to_world(y,x) << u.arcsec
        elif len(self.wcs.low_level_wcs.array_shape) == 2:
            return self.wcs.array_index_to_world(y,x) << u.arcsec
        else:
            raise NotImplementedError("Too many or too little dimensions.")

    def from_lonlat(self,lon,lat):
        """
        This function takes a Helioprojective Longitude, Helioprojective Latitude pair and converts them to the y, x indices to index the object correctly. The function takes its arguments in the order Helioprojective Longitude, Helioprojective Latitude but returns the indices in the (y,x) format so that the output of this function can be used to directly index the object.

        Parameters
        ----------
        lon : float
            The Helioprojective Longitude in arcseconds.
        lat : float
            The Helioprojective Latitude in arcseconds.
        """
        lon, lat = lon << u.arcsec, lat << u.arcsec
        if len(self.wcs.low_level_wcs.array_shape) == 4:
            if hasattr(self, "ind"):
                if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2],self.ind[-1]].world_to_array_index(lon,lat)
                elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2]].world_to_array_index(lon,lat)
                elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,:,self.ind[-1]].world_to_array_index(lon,lat)
                else:
                    return self.wcs.low_level_wcs._wcs[0,0].world_to_array_index(lon,lat)
            else:
                return self.wcs[0,0].world_to_array_index(lon,lat)
        elif len(self.wcs.low_level_wcs.array_shape) == 3:
            if hasattr(self, "ind") and self.wcs.low_level_wcs._wcs.naxis == 4:
                if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2],self.ind[-1]].world_to_array_index(lon,lat)
                elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                    return self.wcs.low_level_wcs._wcs[0,0,self.ind[-2]].world_to_array_index(lon,lat)
                elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,0,:,self.ind[-1]].world_to_array_index(lon,lat)
                else:
                    return self.wcs.low_level_wcs._wcs[0,0].world_to_array_index(lon,lat)
            else:
                if hasattr(self, "ind"):
                    if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                        return self.wcs.low_level_wcs._wcs[0,self.ind[-2],self.ind[-1]].world_to_array_index(lon,lat)
                    elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                        return self.wcs.low_level_wcs._wcs[0,self.ind[-2]].world_to_array_index(lon,lat)
                    elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                        return self.wcs.low_level_wcs._wcs[0,:,self.ind[-1]].world_to_array_index(lon,lat)
                    else:
                        return self.wcs.low_level_wcs._wcs[0].world_to_array_index(lon,lat)
                else:
                    return self.wcs[0].world_to_array_index(lon,lat)
        elif len(self.wcs.low_level_wcs.array_shape) == 2:
            return self.wcs.world_to_array_index(lon,lat)
        else:
            raise NotImplementedError("Too many or too little dimensions.")