import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import h5py, yaml, html
import matplotlib.patches as patches
from matplotlib.colors import SymLogNorm
import astropy.units as u
from .crisp import CRISP, CRISPSequence, CRISPWideband, CRISPWidebandSequence, CRISPNonU, CRISPNonUSequence
from .inversions import Inversion
from .utils import CRISP_sequence_constructor
from matplotlib import ticker
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from .utils import pt_bright_cycler
from IPython.core.display import display
from matplotlib.dates import date2num, DateFormatter

class SpectralViewer:
    """
    Imaging spectroscopic viewer. SpectralViewer should be used when one wants to click on points of an image and have the spectrum displayed for that point. This works **exclusively** in Jupyter notebook but can be a nice data exploration tool. This viewer utilises the data structures defined in `crispy2.crisp` and has many variable options.

    :param data: The data to explore, this can be either one or two spectral lines (support for more than two can be added if required). This is the only required argument to view the data.
    :type data: str or list or CRISP or CRISPSequence or CRISPNonU or CRISPNonUSequence
    :param wcs: A prescribed world coordinate system. If None, the world coordinate system is derived from the data. Default is None.
    :type wcs: astropy.wcs.WCS or None, optional
    :param uncertainty: The uncertainty in the intensity values of the data. Default is None.
    :type uncertainty: numpy.ndarray or None, optional
    :param mask: A mask to be used on the data. Default is None.
    :type mask: numpy.ndarray or None, optional
    :param nonu: Whether or not the spectral axis is non-uniform. Default is False.
    :type nonu: bool, optional

    :cvar coords: The coordinates selected to produce spectra.
    :type coords: list[tuple]
    :cvar px_coords: The coordinates selected to produce spectra in pixel space. This is important for indexing the data later to get the correct spectra.
    :type px_coords: list[tuple]
    :cvar shape_type: The spectra can be selected for a single point or for a box with specified dimensions with top-left corner where the user clicks. This attribute tells the user which point is described by which shape.
    :type shape_type: list[str]
    """
    def __init__(self, data, wcs=None, uncertainty=None, mask=None, nonu=False):
        plt.style.use("bmh")
        self.aa = html.unescape("&#8491;")
        self.l = html.unescape("&lambda;")
        self.a = html.unescape("&alpha;")
        self.D = html.unescape("&Delta;")
        shape = widgets.Dropdown(options=["point", "box"], value="point", description="Shape: ")
        if not nonu:
            if type(data) == str:
                self.cube = CRISP(filename=data, wcs=wcs, uncertainty=uncertainty, mask=mask)
                if self.cube.file.data.ndim == 3:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[0])) << u.Angstrom
                elif self.cube.file.data.ndim == 4:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
            elif type(data) == list:
                data = CRISP_sequence_constructor(data, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)
                self.cube = CRISPSequence(data)
                if self.cube.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[0])) << u.Angstrom
                elif self.cube.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
                if self.cube.list[1].file.data.ndim == 3:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[0])) << u.Angstrom
                elif self.cube.list[1].file.data.ndim == 4:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[1])) << u.Angstrom
            elif type(data) == CRISP:
                self.cube = data
                if self.cube.file.data.ndim == 3:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[0])) << u.Angstrom
                elif self.cube.file.data.ndim == 4:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
            elif type(data) == CRISPSequence:
                self.cube = data
                if self.cube.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[0])) << u.Angstrom
                elif self.cube.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
                if self.cube.list[1].file.data.ndim == 3:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[0])) << u.Angstrom
                elif self.cube.list[1].file.data.ndim == 4:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[1])) << u.Angstrom
        else:
            if type(data) == str:
                self.cube = CRISPNonU(filename=data, wcs=wcs, uncertainty=uncertainty, mask=mask)
                if self.cube.file.data.ndim == 3:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[0])) << u.Angstrom
                elif self.cube.file.data.ndim == 4:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
            elif type(data) == list:
                data = CRISP_sequence_constructor(data, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)
                self.cube = CRISPNonUSequence(data)
                if self.cube.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[0])) << u.Angstrom
                elif self.cube.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
                if self.cube.list[1].file.data.ndim == 3:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[0])) << u.Angstrom
                elif self.cube.list[1].file.data.ndim == 4:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[1])) << u.Angstrom
            elif type(data) == CRISPNonU:
                self.cube = data
                if self.cube.file.data.ndim == 3:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[0])) << u.Angstrom
                elif self.cube.file.data.ndim == 4:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
            elif type(data) == CRISPNonUSequence:
                self.cube = data
                if self.cube.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[0])) << u.Angstrom
                elif self.cube.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
                if self.cube.list[1].file.data.ndim == 3:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[0])) << u.Angstrom
                elif self.cube.list[1].file.data.ndim == 4:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[1])) << u.Angstrom

        if type(self.cube) == CRISP or type(self.cube) == CRISPNonU:
            self.fig = plt.figure(figsize=(8,10))
            try:
                self.ax1 = self.fig.add_subplot(1, 2, 1, projection=self.cube.wcs.dropaxis(-1))
            except:
                self.ax1 = self.fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.cube[0].wcs.low_level_wcs, 0))
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax2 = self.fig.add_subplot(1, 2, 2)
            self.ax2.yaxis.set_label_position("right")
            self.ax2.yaxis.tick_right()
            self.ax2.set_ylabel("I [DNs]")
            self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax2.tick_params(direction="in")

            ll = widgets.SelectionSlider(options=[np.round(l - np.median(self.wvls), decimals=2).value for l in self.wvls], description = f"{self.D} {self.l} [{self.aa}]")

            out1 = widgets.interactive_output(self._img_plot1, {"ll" : ll})
            out2 = widgets.interactive_output(self._shape, {"opts" : shape})

            display(widgets.HBox([ll, shape]))
                
        elif type(self.cube) == CRISPSequence or type(self.cube) == CRISPNonUSequence:
            self.fig = plt.figure(figsize=(8,10))
            try:
                self.ax1 = self.fig.add_subplot(2, 2, 1, projection=self.cube.list[0].wcs.dropaxis(-1))
            except:
                self.ax1 = self.fig.add_subplot(2, 2, 1, projection=SlicedLowLevelWCS(self.cube.list[0][0].wcs.low_level_wcs, 0))
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax1.xaxis.set_label_position("top")
            self.ax1.xaxis.tick_top()

            try:
                self.ax2 = self.fig.add_subplot(2, 2, 3, projection=self.cube.list[0].wcs.dropaxis(-1))
            except:
                self.ax2 = self.fig.add_subplot(2, 2, 3, projection=SlicedLowLevelWCS(self.cube.list[0][0].wcs.low_level_wcs, 0))
            self.ax2.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax2.set_xlabel("Helioprojective Longitude [arcsec]")

            self.ax3 = self.fig.add_subplot(2, 2, 2)
            self.ax3.yaxis.set_label_position("right")
            self.ax3.yaxis.tick_right()
            self.ax3.set_ylabel("Intensity [DNs]")
            self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax3.xaxis.set_label_position("top")
            self.ax3.xaxis.tick_top()
            self.ax3.tick_params(direction="in")

            self.ax4 = self.fig.add_subplot(2, 2, 4)
            self.ax4.yaxis.set_label_position("right")
            self.ax4.yaxis.tick_right()
            self.ax4.set_ylabel("Intensity [DNs]")
            self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax4.tick_params(direction="in")

            ll1 = widgets.SelectionSlider(
                options=[np.round(l - np.median(self.wvls1), decimals=2).value for l in self.wvls1],
                description=fr"{self.D} {self.l}$_{1}$ [{self.aa}]",
                style={"description_width" : "initial"}
            )
            ll2 = widgets.SelectionSlider(
                options=[np.round(l - np.median(self.wvls2), decimals=2).value for l in self.wvls2],
                description=fr"{self.D} {self.l}$_{2}$ [{self.aa}]",
                style={"description_width" : "initial"}
            )

            out1 = widgets.interactive_output(self._img_plot2, {"ll1" : ll1, "ll2" : ll2})
            out2 = widgets.interactive_output(self._shape, {"opts" : shape})

            display(widgets.HBox([widgets.VBox([ll1, ll2]), shape]))

        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0

        self.receiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        try:
            x = widgets.IntText(value=1, min=1, max=self.cube.shape[-1], description="x [pix]")
            y = widgets.IntText(value=1, min=1, max=self.cube.shape[-2], description="y [pix]")
        except:
            x = widgets.IntText(value=1, min=1, max=self.cube.list[0].shape[-1], description="x [pix]")
            y = widgets.IntText(value=1, min=1, max=self.cube.list[0].shape[-2], description="y [pix]")
        outx = widgets.interactive_output(self._boxx, {"x" : x})
        outy = widgets.interactive_output(self._boxy, {"y" : y})
        display(widgets.HBox([x, y]))

        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description="Clear")
        clear_button.on_click(self._clear)
        save_button = widgets.Button(description="Save")
        save_button.on_click(self._save)
        display(widgets.HBox([done_button, clear_button, save_button]))
        widgets.interact(self._file_name, fn= widgets.Text(description="Filename to save as: ", style={"description_width" : "initial"}, layout=widgets.Layout(width="50%")))

    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode != "":
            return

        if type(self.cube) == CRISP or type(self.cube) == CRISPNonU:
            if self.shape == "point":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                centre_coord = int(event.ydata), int(event.xdata)
                self.px_coords.append(centre_coord)
                self.shape_type.append("point")
                circ = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
                self.ax1.add_patch(circ)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                px = self.cube.to_lonlat(*centre_coord) << u.arcsec
                if self.cube.file.data.ndim == 3:
                    self.ax2.plot(self.wvls, self.cube.file.data[:, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube.file.data.ndim == 4:
                    self.ax2.plot(self.wvls, self.cube.file.data[0, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax2.legend()
                self.coords.append(px)
                self.colour_idx += 1
                self.fig.canvas.draw()
            elif self.shape == "box":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                box_anchor = int(event.ydata), int(event.xdata)
                self.px_coords.append(box_anchor)
                self.shape_type.append("box")
                # obtain the coordinates of the box on a grid with pixels the size of the box to make sure there is not copies of the same box
                box_coord = box_anchor[0] // self.boxy, box_anchor[1] // self.boxx
                if box_coord in self.box_coords:
                    coords = [p.get_xy() for p in self.ax1.patches]
                    for p in self.ax1.patches:
                        if p.get_xy() == box_anchor:
                            p.remove()
                    
                    idx = self.box_coords.index(box_coord)
                    del self.box_coords[idx]
                    del self.px_coords[idx]
                    del self.shape_type[idx]
                    del self.coords[idx]
                    return
                
                self.coords.append(self.cube.to_lonlat(*box_anchor) << u.arcsec)
                rect = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
                rect.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                self.ax1.add_patch(rect)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt = self.ax1.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                if self.cube.file.data.ndim == 3:
                    self.ax2.plot(self.wvls, np.mean(self.cube.file.data[:,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube.file.data.ndim == 4:
                    self.ax2.plot(self.wvls, np.mean(self.cube.file.data[0, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax2.legend()
                self.colour_idx += 1
                self.fig.canvas.draw()
        elif type(self.cube) == CRISPSequence or type(self.cube) == CRISPNonUSequence:
            if self.shape == "point":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                centre_coord = int(event.ydata), int(event.xdata) #with WCS, the event data is returned in pixels so we don't need to do the conversion from real world but rather to real world later on
                self.px_coords.append(centre_coord)
                circ1 = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
                circ2 = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
                self.ax1.add_patch(circ1)
                self.ax2.add_patch(circ2)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt_1 = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt_2 = self.ax2.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt_1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                txt_2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                px = self.cube.list[0].to_lonlat(*centre_coord) << u.arcsec
                if self.cube.list[0].file.data.ndim == 3:
                    self.ax3.plot(self.wvls1, self.cube.list[0].file.data[:, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube.list[0].file.data.ndim == 4:
                    self.ax3.plot(self.wvls1, self.cube.list[0].file.data[0, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                if self.cube.list[1].file.data.ndim == 3:
                    self.ax4.plot(self.wvls2, self.cube.list[1].file.data[:, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube.list[1].file.data.ndim == 4:
                    self.ax4.plot(self.wvls2, self.cube.list[1].file.data[0, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax3.legend()
                self.ax4.legend()
                self.coords.append(px)
                self.colour_idx += 1
                self.fig.canvas.draw()
            elif self.shape == "box":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                box_anchor = int(event.ydata), int(event.xdata)
                self.px_coords.append(box_anchor)
                self.shape_type.append("box")
                # obtain the coordinates of the box on a grid with pixels the size of the box to make sure there is not copies of the same box
                box_coord = box_anchor[0] // self.boxy, box_anchor[1] // self.boxx
                if box_coord in self.box_coords:
                    coords = [p.get_xy() for p in self.ax.patches]
                    for p in self.ax.patches:
                        if p.get_xy() == box_anchor:
                            p.remove()
                    
                    idx = self.box_coords.index(box_coord)
                    del self.box_coords[idx]
                    del self.px_coords[idx]
                    del self.shape_type[idx]
                    del self.coords[idx]
                    return
                
                self.coords.append(self.cube.to_lonlat(*box_anchor) << u.arcsec)
                rect1 = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
                rect1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                rect2 = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
                rect2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                self.ax1.add_patch(rect1)
                self.ax2.add_patch(rect2)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt1 = self.ax1.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                txt2 = self.ax2.text(box_anchor[1]-50, box_anchor[0]-1, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt2.set_path_effect([PathEffects.withStroke(linewidth=3, foreground="k")])
                if self.cube.list[0].file.data.ndim == 3:
                    self.ax3.plot(self.wvls1, np.mean(self.cube.list[0].file.data[:,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube.list[0].file.data.ndim == 4:
                    self.ax3.plot(self.wvls1, np.mean(self.cube.list[0].file.data[0, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                if self.cube.list[1].file.data.ndim == 3:
                    self.ax4.plot(self.wvls2, np.mean(self.cube.list[1].file.data[:,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube.list[1].file.data.ndim == 4:
                    self.ax4.plot(self.wvls2, np.mean(self.cube.list[1].file.data[0, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax3.legend()
                self.ax4.legend()
                self.colour_idx += 1
                self.fig.canvas.draw()

    def _shape(self, opts):
        self.shape = opts

    def _boxx(self, x):
        self.boxx = x

    def _boxy(self, y):
        self.boxy = y

    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)

    def _clear(self, _):
        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0
        if type(self.cube) == CRISP:
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax1.texts) > 0:
                for t in self.ax1.texts:
                    t.remove()
            self.ax2.clear()
            self.ax2.set_ylabel("Intensity [DNs]")
            self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax2.patches) > 0:
                for p in self.ax2.patches:
                    p.remove()
            while len(self.ax1.texts) > 0:
                for t in self.ax1.texts:
                    t.remove()
            while len(self.ax2.texts) > 0:
                for t in self.ax2.texts:
                    t.remove()
            self.ax3.clear()
            self.ax3.set_ylabel("Intensity [DNs]")
            self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax4.clear()
            self.ax4.set_ylabel("Intensity [DNs]")
            self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _save(self, _):
        self.fig.savefig(self.filename, dpi=300)

    def _file_name(self, fn):
        self.filename = fn

    def _img_plot1(self, ll):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar != None:
            self.ax1.images[-1].colorbar.remove()

        ll_idx = int(np.where(np.round(self.wvls, decimals=2).value == np.round(np.median(self.wvls).value + ll, decimals=2))[0])

        try:
            data = self.cube.file.data[ll_idx].astype(np.float)
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        except:
            data = self.cube.file.data[0, ll_idx].astype(np.float)
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        try:
            el = self.cube.file.header["WDESC1"]
        except KeyError:
            el = self.cube.file.header["element"]
        self.ax1.set_title(fr"{el} {self.aa} {self.D} {self.l}$_{1}$ = {ll} {self.aa}")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="Intensity [DNs]")

    def _img_plot2(self, ll1, ll2):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar != None:
            self.ax1.images[-1].colorbar.remove()

        if self.ax2.images == []:
            pass
        elif self.ax2.images[-1].colorbar != None:
            self.ax2.images[-1].colorbar.remove()

        ll1_idx = int(np.where(np.round(self.wvls1, decimals=2).value == np.round(np.median(self.wvls1).value + ll1, decimals=2))[0])
        ll2_idx = int(np.where(np.round(self.wvls2, decimals=2).value == np.round(np.median(self.wvls2).value + ll2, decimals=2))[0])

        try:
            data = self.cube.list[0].file.data[ll1_idx].astype(np.float)
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        except:
            data = self.cube.list[0].file.data[0, ll1_idx].astype(np.float)
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        try:
            data = self.cube.list[1].file.data[ll2_idx].astype(np.float)
            data[data < 0] = np.nan
            im2 = self.ax2.imshow(data, cmap="Greys_r")
        except:
            data = self.cube.list[1].file.data[0, ll2_idx].astype(np.float)
            data[data < 0] = np.nan
            im2 = self.ax2.imshow(data, cmap="Greys_r")
        try:
            el1 = self.cube.list[0].file.header["WDESC1"]
            el2 = self.cube.list[1].file.header["WDESC1"]
        except KeyError:
            el1 = self.cube.list[0].file.header["element"]
            el2 = self.cube.list[1].file.header["element"]
        self.ax1.set_title(fr"{el1} {self.aa} {self.D} {self.l}$_{1}$ = {ll1} {self.aa}")
        self.ax2.set_title(fr"{el2} {self.aa} {self.D} {self.l}$_{2}$ = {ll2} {self.aa}")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="Intensity [DNs]")
        self.fig.colorbar(im2, ax=self.ax2, orientation="horizontal", label="Intensity [DNs]")

class WidebandViewer:
    """
    Wideband image viewer. This visualisation tool is useful for exploring the time series evolution of the wideband images.

    :param files: The files to explore the time series for.
    :type files: CRISPWidebandSequence or list

    :cvar coords: The coordinates selected to produce spectra.
    :type coords: list[tuple]
    :cvar px_coords: The coordinates selected to produce spectra in pixel space. This is important for indexing the data later to get the correct spectra.
    :type px_coords: list[tuple]
    :cvar shape_type: The spectra can be selected for a single point or for a box with specified dimensions with top-left corner where the user clicks. This attribute tells the user which point is described by which shape.
    :type shape_type: list[str]
    """
    def __init__(self, files):
        plt.style.use("bmh")
        shape = widgets.Dropdown(options=["point", "box"], value="point", description="Shape: ")
        if type(files) == CRISPWidebandSequence:
            self.cube = files
        elif type(files) == list and type(files[0]) == dict:
            self.cube = CRISPWidebandSequence(files)
        elif type(files) == list and type(files[0]) == str:
            files = [{"filename" : f} for f in files]
            self.cube = CRISPWidebandSequence(files)
        elif type(files) == list and type(files[0]) == CRISPWidebandSequence:
            self.cube = files
        if type(self.cube) is not list:
            try:
                self.time = [date2num(f.file.header["DATE-AVG"]) for f in self.cube.list]
            except KeyError:
                self.time = [date2num(f.file.header["date-obs"]+" "+f.file.header["time-obs"]) for f in self.cube.list]
            self.fig = plt.figure(figsize=(8,10))
            self.ax1 = self.fig.add_subplot(1, 2, 1, projection=self.cube.list[0].wcs)
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax2 = self.fig.add_subplot(1, 2, 2)
            self.ax2.yaxis.set_label_position("right")
            self.ax2.yaxis.tick_right()
            self.ax2.set_ylabel("I [DNs]")
            self.ax2.set_xlabel("Time [UTC]")
            self.ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.ax2.tick_params(direction="in")

            t = widgets.IntSlider(value=0, min=0, max=len(self.cube.list)-1, step=1, description="Time index: ", style={"description_width" : "initial"})

            widgets.interact(self._img_plot1, t = t)
        else:
            try:
                self.time1 = [date2num(f.file.header["DATE-AVG"]) for f in self.cube[0].list]
                self.time2 = [date2num(f.file.header["DATE-AVG"]) for f in self.cube[1].list]
            except KeyError:
                self.time1 = [date2num(f.file.header["date-obs"]+" "+f.file.header["time-obs"]) for f in self.cube[0].list]
                self.time2 = [date2num(f.file.header["date-obs"]+" "+f.file.header["time-obs"]) for f in self.cube[1].list]
            self.fig = plt.figure(figsize=(8,10))
            self.ax1 = self.fig.add_subplot(2, 2, 1, projection=self.cube[0].list[0].wcs)
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax1.xaxis.set_label_position("top")
            self.ax1.xaxis.tick_top()

            self.ax2 = self.fig.add_subplot(2, 2, 3, projection=self.cube[1].list[0].wcs)
            self.ax2.ylabel("Helioprojective Latitude [arcsec]")
            self.ax2.xlabel("Helioprojective Longitude [arcsec]")

            self.ax3 = self.fig.add_subplot(2, 2, 2)
            self.ax3.yaxis.set_label_position("right")
            self.ax3.yaxis.tick_right()
            self.ax3.set_ylabel("I [DNs]")
            self.ax3.set_xlabel("Time [UTC]")
            self.ax3.xaxis.set_label_position("top")
            self.ax3.xaxis.tick_top()
            self.ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.ax3.tick_params(direction="in")

            self.ax4 = self.fig.add_subplot(2, 2, 4)
            self.ax4.yaxis.set_label_position("right")
            self.ax4.yaxis.tick_right()
            self.ax4.set_ylabel("I [DNs]")
            self.ax4.set_xlabel("Time [UTC]")
            self.ax4.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.ax4.tick_params(direction="in")

            t1 = widgets.IntSlider(value=0, min=0, max=len(self.cube[0].list)-1, step=1, description="Time index: ", style={"description_width" : "initial"})
            t2 = widgets.IntSlider(value=0, min=0, max=len(self.cube[1].list)-1, step=1, description="Time index: ", style={"description_width" : "initial"})

            widgets.interact(self._img_plot2, t1=t1, t2=t2)

        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0

        self.reveiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        widgets.interact(self._shape, opts=shape)

        x = widgets.IntText(value=1, min=1, max=self.cube.list[0].shape[-1], description="x [pix]")
        y = widgets.IntText(value=1, min=1, max=self.cube.list[0].shape[-2], description="y [pix]")
        outx = widgets.interactive_output(self._boxx, {"x" : x})
        outy = widgets.interactive_output(self._boxy, {"y" : y})
        display(widgets.HBox([x, y]))

        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description="Clear")
        clear_button.on_click(self._clear)
        save_button = widgets.Button(description="Save")
        save_button.on_click(self._save)
        display(widgets.HBox([done_button, clear_button, save_button]))
        widgets.interact(self._file_name, fn= widgets.Text(description="Filename to save as: ", style={"description_width" : "initial"}, layout=widgets.Layout(width="50%")))

    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode != "":
            return

        if type(self.cube) == CRISPWidebandSequence:
            if self.shape == "point":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                centre_coord = int(event.ydata), int(event.xdata)
                self.px_coords.append(centre_coord)
                self.shape_type.append("point")
                circ = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
                self.ax1.add_patch(circ)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                px = self.cube.list[0].wcs.array_index_to_world(*centre_coord) << u.arcsec
                prof = [f.file.data[centre_coord[0], centre_coord[1]] for f in self.cube.list]
                self.ax2.plot(self.time, prof, marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax2.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
                for label in self.ax2.get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
                self.ax2.legend()
                self.coords.append(px)
                self.colour_idx += 1
                self.fig.canvas.draw()
            elif self.shape == "box":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                box_anchor = int(event.ydata), int(event.xdata)
                self.px_coords.append(box_anchor)
                self.shape_type.append("box")
                # obtain the coordinates of the box on a grid with pixels the size of the box to make sure there is not copies of the same box
                box_coord = box_anchor[0] // self.boxy, box_anchor[1] // self.boxx
                if box_coord in self.box_coords:
                    coords = [p.get_xy() for p in self.ax.patches]
                    for p in self.ax.patches:
                        if p.get_xy() == box_anchor:
                            p.remove()
                    
                    idx = self.box_coords.index(box_coord)
                    del self.box_coords[idx]
                    del self.px_coords[idx]
                    del self.shape_type[idx]
                    del self.coords[idx]
                    return
                
                self.coords.append(self.cube.to_lonlat(*box_anchor) << u.arcsec)
                rect = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
                rect.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                self.ax1.add_patch(rect)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt = self.ax1.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                prof = [np.mean(f.file.data[box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube.list]
                self.ax2.plot(self.time, prof, marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax2.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
                for label in self.ax2.get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
                self.ax2.legend()
                self.colour_idx += 1
                self.fig.canvas.draw()
        elif type(self.cube) == list:
            if self.shape == "point":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                centre_coord = int(event.ydata), int(event.xdata)
                self.px_coords.append(centre_coord)
                circ1 = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
                circ2 = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
                self.ax1.add_patch(circ1)
                self.ax2.add_patch(circ2)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt_1 = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt_2 = self.ax2.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt_1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                txt_2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                px = self.cube[0].list[0].wcs.array_index_to_world(*centre_coord) << u.arcsec
                prof_1 = [f.file.data[centre_coord[0], centre_coord[1]] for f in self.cube[0].list]
                prof_2 = [f.file.data[centre_coord[0], centre_coord[1]] for f in self.cube[1].list]
                self.ax3.plot(self.time1, prof_1, marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax4.plot(self.time2, prof_2, marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax3.legend()
                self.ax3.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
                for label in self.ax3.get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
                self.ax4.legend()
                self.ax4.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
                for label in self.ax4.get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
                self.coords.append(px)
                self.colour_idx += 1
                self.fig.canvas.draw()
            elif self.shape == "box":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                box_anchor = int(event.ydata), int(event.xdata)
                self.px_coords.append(box_anchor)
                self.shape_type.append("box")
                # obtain the coordinates of the box on a grid with pixels the size of the box to make sure there is not copies of the same box
                box_coord = box_anchor[0] // self.boxy, box_anchor[1] // self.boxx
                if box_coord in self.box_coords:
                    coords = [p.get_xy() for p in self.ax.patches]
                    for p in self.ax.patches:
                        if p.get_xy() == box_anchor:
                            p.remove()
                    
                    idx = self.box_coords.index(box_coord)
                    del self.box_coords[idx]
                    del self.px_coords[idx]
                    del self.shape_type[idx]
                    del self.coords[idx]
                    return
                
                self.coords.append(self.cube.to_lonlat(*box_anchor) << u.arcsec)
                rect1 = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
                rect1.set_path_effects([PathEffects(linewidth=3, foreground="k")])
                rect2 = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
                rect2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                self.ax1.add_patch(rect1)
                self.ax2.add_patch(rect2)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt1 = self.ax1.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt1.set_path_effects([PathEffects(linewidth=3, foreground="k")])
                txt2 = self.ax2.text(box_anchor[1]-50, box_anchor[0]-1, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt2.set_path_effect([PathEffects(linewidth=3, foreground="k")])
                prof_1 = [np.mean(f.file.data[box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube[0].list]
                prof_2 = [np.mean(f.file.data[box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube[1].list]
                self.ax3.plot(self.time1, prof_1, marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax4.plot(self.time2, prof_2, marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax3.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
                for label in self.ax3.get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
                self.ax4.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
                for label in self.ax4.get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
                self.ax3.legend()
                self.ax4.legend()
                self.colour_idx += 1
                self.fig.canvas.draW()

    def _shape(self, opts):
        self.shape = opts

    def _boxx(self, x):
        self.boxx = x

    def _boxy(self, y):
        self.boxy = y

    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)

    def _clear(self, _):
        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0
        if type(self.cube) == CRISPWidebandSequence:
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax1.texts) > 0:
                for t in self.ax1.texts:
                    t.remove()
            self.ax2.clear()
            self.ax2.set_ylabel("I [DNs]")
            self.ax2.set_xlabel("Time [UTC]")
            self.ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax2.patches) > 0:
                for p in self.ax2.patches:
                    p.remove()
            while len(self.ax1.texts) > 0:
                for t in self.ax1.patches:
                    t.remove()
            while len(self.ax2.patches) > 0:
                for t in self.ax2.patches:
                    t.remove()
            self.ax3.clear()
            self.ax3.set_ylabel("I [DNs]")
            self.ax3.set_xlabel("Time [UTC]")
            self.ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.ax4.clear()
            self.ax4.set_ylabel("I [DNs]")
            self.ax4.set_xlabel("Time [UTC]")
            self.ax4.xaxis.set_major_locator(plt.MaxNLocator(4))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _save(self, _):
        self.fig.savefig(self.filename, dpi=300)

    def _file_name(self, fn):
        self.filename = fn

    def _img_plot1(self, t):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()

        im1 = self.ax1.imshow(self.cube.list[t].file.data, cmap="Greys_r")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="I [DNs]")

    def _img_plot2(self, t1, t2):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()

        if self.ax2.images == []:
            pass
        elif self.ax2.images[-1].colorbar is not None:
            self.ax2.images[-1].colorbar.remove()

        im1 = self.ax1.imshow(self.cube[0].list[t].file.data, cmap="Greys_r")
        im2 = self.ax2.imshow(self.cube[1].list[t].file.data, cmap="Greys_r")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="I [DNs]")
        self.fig.colorbar(im2, ax=self.ax2, orientation="horizontal", label="I [DNs]")

class AtmosViewer:
    """
    This visualisation tool is for the investigation of atmospheric parameters found via inversion techniques. This makes use of the ``Inversion`` class. This assumes that there are three atmospheric parameters in the inversion: electron number density, electron temperature and bulk line-of-sight velocity. These are the estimated quantities by RADYNVERSION.

    :param filename: The inversion file to be used.
    :type filename: str or Inversion
    :param z: The physical height grid of the estimated atmospheric parameters in megametres. Can only be None if filename is already an ``Inversion`` instance. Default is None. (N.B. the RADYNVERSION height grid is available from ``crispy2.radynversion.utils``).
    :type z: numpy.ndarray or None, optional
    :param wcs: The world coordinate system that the inversion parameters are defined by. Can be None only if filename is already an ``Inversion`` instance. Default is None.
    :type wcs: astropy.wcs.WCS or None, optional
    :param header: The additional header information from the observations. Default is None.
    :type header: dict or None, optional
    :param eb: Whether or not to plot the errorbars on the parameter profiles. Default is False.
    :type eb: bool, optional

    :cvar coords: The coordinates selected to produce spectra.
    :type coords: list[tuple]
    :cvar px_coords: The coordinates selected to produce spectra in pixel space. This is important for indexing the data later to get the correct spectra.
    :type px_coords: list[tuple]
    :cvar shape_type: The spectra can be selected for a single point or for a box with specified dimensions with top-left corner where the user clicks. This attribute tells the user which point is described by which shape.
    :type shape_type: list[str]
    """
    def __init__(self, filename, z=None, wcs=None, header=None, eb=False):
        plt.style.use("bmh")
        shape = widgets.Dropdown(options=["point", "box"], value="point", description="Shape: ")
        if type(filename) == str:
            assert z is not None
            assert header is not None
            self.inv = Inversion(filename=filename, wcs=wcs, z=z, header=header)
        elif type(filename) == Inversion:
            self.inv = filename

        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0
        self.eb = eb

        self.fig = plt.figure(figsize=(8,10))
        self.gs = self.fig.add_gridspec(nrows=5, ncols=3)

        self.ax1 = self.fig.add_subplot(self.gs[:2, 0], projection=self.inv.wcs.dropaxis(-1))
        self.ax2 = self.fig.add_subplot(self.gs[:2, 1], projection=self.inv.wcs.dropaxis(-1))
        self.ax3 = self.fig.add_subplot(self.gs[:2, 2], projection=self.inv.wcs.dropaxis(-1))
        self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
        self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
        self.ax2.set_xlabel("Helioprojective Longitude [arcsec]")
        self.ax3.set_xlabel("Helioprojective Longitude [arcsec]")
        self.ax2.tick_params(axis="y", labelleft=False)
        self.ax3.tick_params(axis="y", labelleft=False)
        
        self.ax4 = self.fig.add_subplot(self.gs[2, :])
        self.ax4.set_ylabel(r"log $n_{e}$ [cm$^{-3}$]")
        self.ax4.yaxis.set_label_position("right")
        self.ax4.yaxis.tick_right()
        
        self.ax5 = self.fig.add_subplot(self.gs[3, :])
        self.ax5.set_ylabel(r"log T [K]")
        self.ax5.yaxis.set_label_position("right")
        self.ax5.yaxis.tick_right()
        
        self.ax6 = self.fig.add_subplot(self.gs[4, :])
        self.ax6.set_ylabel(r"v [km s$^{-1}$]")
        self.ax6.set_xlabel(r"z [Mm]")
        self.ax6.yaxis.set_label_position("right")
        self.ax6.yaxis.tick_right()
        
        self.ax4.tick_params(axis="x", labelbottom=False, direction="in")
        self.ax5.tick_params(axis="x", labelbottom=False, direction="in")
        self.ax6.tick_params(axis="both", direction="in")
        
        widgets.interact(self._img_plot,
                        z = widgets.SelectionSlider(options=np.round(self.inv.z, decimals=3), description="Image height [Mm]: ", style={"description_width" : "initial"}, layout=widgets.Layout(width="50%")))
        
        widgets.interact(self._shape, opts=shape)

        self.receiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        x = widgets.IntText(value=1, min=1, max=self.inv.ne.shape[-1], description="x [pix]")
        y = widgets.IntText(value=1, min=1, max=self.inv.ne.shape[-2], description="y [pix]")
        outx = widgets.interactive_output(self._boxx, {"x" : x})
        outy = widgets.interactive_output(self._boxy, {"y" : y})
        display(widgets.HBox([x, y]))
        
        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description='Clear')
        clear_button.on_click(self._clear)
        save_button = widgets.Button(description="Save")
        save_button.on_click(self._save)
        display(widgets.HBox([done_button, clear_button, save_button]))
        widgets.interact(self._file_name, fn = widgets.Text(description="Filename to save as: ", style={"description_width" : "initial"}), layout=widgets.Layout(width="50%"))
        
    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode != "":
            return

        if self.shape == "point":
            if self.colour_idx > len(pt_bright_cycler)-1:
                self.colour_idx = 0
                self.n += 1
            centre_coord = int(event.ydata), int(event.xdata)
            self.px_coords.append(centre_coord)
            circ1 = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
            circ2 = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
            circ3 = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
            self.ax1.add_patch(circ1)
            self.ax2.add_patch(circ2)
            self.ax3.add_patch(circ3)
            font = {
                "size" : 12,
                "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
            }
            txt_1 = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
            txt_2 = self.ax2.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
            txt_3 = self.ax3.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
            txt_1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            txt_2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            txt_3.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            if self.eb:
                self.ax4.errorbar(self.inv.z, self.inv.ne[:,centre_coord[0], centre_coord[1]], yerr=self.inv.err[:,centre_coord[0],centre_coord[1],0], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax5.errorbar(self.inv.z, self.inv.temp[:,centre_coord[0], centre_coord[1]], yerr=self.inv.err[:,centre_coord[0],centre_coord[1],1], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax6.errorbar(self.inv.z, self.inv.vel[:,centre_coord[0],centre_coord[1]], yerr=self.inv.err[:,centre_coord[0],centre_coord[1],2], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            else:
                self.ax4.plot(self.inv.z, self.inv.ne[:,centre_coord[0],centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax5.plot(self.inv.z, self.inv.temp[:,centre_coord[0],centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax6.plot(self.inv.z, self.inv.vel[:,centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax4.legend()
            self.ax5.legend()
            self.ax6.legend()
            px = self.inv.to_lonlat(*centre_coord) << u.arcsec
            self.colour_idx += 1
            self.coords.append(px)
            self.fig.canvas.draw()
        elif self.shape == "box":
            if self.colour_idx > len(pt_bright_cycler)-1:
                self.colour_idx = 0
                self.n += 1
            box_anchor = int(event.ydata), int(event.xdata)
            self.px_coords.append(box_anchor)
            self.shape_type.append("box")
            # obtain the coordinates of the box on a grid with pixels the size of the box to make sure there is not copies of the same box
            box_coord = box_anchor[0] // self.boxy, box_anchor[1] // self.boxx
            if box_coord in self.box_coords:
                coords = [p.get_xy() for p in self.ax1.patches]
                for p in self.ax1.patches:
                    if p.get_xy() == box_anchor:
                        p.remove()
                
                idx = self.box_coords.index(box_coord)
                del self.box_coords[idx]
                del self.px_coords[idx]
                del self.shape_type[idx]
                del self.coords[idx]
                return
            
            self.coords.append(self.inv.to_lonlat(*box_anchor) << u.arcsec)
            rect1 = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
            rect1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            rect2 = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
            rect2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            rect3 = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
            rect3.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            self.ax1.add_patch(rect1)
            self.ax2.add_patch(rect2)
            self.ax3.add_patch(rect3)
            font = {
                "size" : 12,
                "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
            }
            txt1 = self.ax1.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
            txt1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            txt2 = self.ax2.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
            txt2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            txt3 = self.ax3.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
            txt3.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            if self.eb:
                self.ax4.errorbar(self.inv.z, np.mean(self.inv.ne[:,box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx], axis=(0,1)), yerr=np.mean(self.inv.err[:,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx,0], axis=(0,1)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax5.errorbar(self.inv.z, np.mean(self.inv.temp[:,box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx], axis=(0,1)), yerr=np.mean(self.inv.err[box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx,1], axis=(0,1)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax6.errorbar(self.inv.z, np.mean(self.inv.vel[:,box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx], axis=(0,1)), yerr=np.mean(self.inv.err[:,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx,2], axis=(0,1)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            else:
                self.ax4.plot(self.inv.z, np.mean(self.inv.ne[:,box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx], axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax5.plot(self.inv.z, np.mean(self.inv.temp[:,box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx], axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax6.plot(self.inv.z, np.mean(self.inv.vel[:,box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx], axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax4.legend()
            self.ax5.legend()
            self.ax6.legend()
            self.colour_idx += 1
            self.fig.canvas.draw()
        
    def _shape(self, opts):
        self.shape = opts

    def _boxx(self, x):
        self.boxx = x

    def _boxy(self, y):
        self.boxy = y

    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)
        
    def _clear(self, _):
        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0
        while len(self.ax1.patches) > 0:
            for p in self.ax1.patches:
                p.remove()
        while len(self.ax2.patches) > 0:
            for p in self.ax2.patches:
                p.remove()
        while len(self.ax3.patches) > 0:
            for p in self.ax3.patches:
                p.remove()
        while len(self.ax1.texts) > 0:
            for t in self.ax1.texts:
                t.remove()
        while len(self.ax2.texts) > 0:
            for t in self.ax2.texts:
                t.remove()
        while len(self.ax3.texts) > 0:
            for t in self.ax3.texts:
                t.remove()
        self.ax4.clear()
        self.ax4.set_ylabel(r"log n$_{e}$ [cm$^{-3}$]")
        self.ax5.clear()
        self.ax5.set_ylabel(r"log T [K]")
        self.ax6.clear()
        self.ax6.set_ylabel(r"v [km s$^{-1}$]")
        self.ax6.set_xlabel(r"z [Mm]")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _save(self, _):
        self.fig.savefig(self.filename, dpi=300)

    def _file_name(self, fn):
        self.filename = fn
            
    def _img_plot(self, z):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar != None:
            self.ax1.images[-1].colorbar.remove()
        if self.ax2.images == []:
            pass
        elif self.ax2.images[-1].colorbar != None:
            self.ax2.images[-1].colorbar.remove()
        if self.ax3.images == []:
            pass
        elif self.ax3.images[-1].colorbar != None:
            self.ax3.images[-1].colorbar.remove()
        z_idx = int(np.where(np.round(self.inv.z, decimals=3) == np.round(z, decimals=3))[0])
        im1 = self.ax1.imshow(self.inv.ne[z_idx], cmap="cividis")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label=r"log $n_{e}$ [cm$^{-3}$]")

        im2 = self.ax2.imshow(self.inv.temp[z_idx], cmap="hot")
        self.fig.colorbar(im2, ax=self.ax2, orientation="horizontal", label=r"log T [K]")

        im3 = self.ax3.imshow(self.inv.vel[z_idx], cmap="RdBu", clim=(-np.max(self.inv.vel[z_idx]), np.max(self.inv.vel[z_idx])))
        self.fig.colorbar(im3, ax=self.ax3, orientation="horizontal", label=r"v [km s$^{-1}$]")

class ImageViewer:
    """
    This visualiser only views the images for data, not the spectra. For use when interested only in imaging data. Includes sliders to change the wavelength of the observation.

    :param data: The data to explore, this can be either one or two spectral lines (support for more than two can be added if required). This is the only required argument to view the data.
    :type data: str or list or CRISP or CRISPSequence or CRISPNonU or CRISPNonUSequence
    :param wcs: A prescribed world coordinate system. If None, the world coordinate system is derived from the data. Default is None.
    :type wcs: astropy.wcs.WCS or None, optional
    :param uncertainty: The uncertainty in the intensity values of the data. Default is None.
    :type uncertainty: numpy.ndarray or None, optional
    :param mask: A mask to be used on the data. Default is None.
    :type mask: numpy.ndarray or None, optional
    :param nonu: Whether or not the spectral axis is non-uniform. Default is False.
    :type nonu: bool, optional
    """

    def __init__(self, data, wcs=None, uncertainty=None, mask=None, nonu=False):
        plt.style.use("bmh")
        self.aa = html.unescape("&#8491;")
        self.l = html.unescape("&lambda;")
        self.a = html.unescape("&alpha;")
        self.D = html.unescape("&Delta;")
        if not nonu:
            if type(data) == str:
                self.cube = CRISP(filename=data, wcs=wcs, uncertainty=uncertainty, mask=mask)
                if self.cube.file.data.ndim == 3:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[0])) << u.Angstrom
                elif self.cube.file.data.ndim == 4:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
            elif type(data) == list:
                data = CRISP_sequence_constructor(data, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)
                self.cube = CRISPSequence(files=data)
                if self.cube.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[0])) << u.Angstrom
                elif self.cube.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
                if self.cube.list[1].file.data.ndim == 3:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[0])) << u.Angstrom
                elif self.cube.list[1].file.data.ndim == 4:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[1])) << u.Angstrom
            elif type(data) == CRISP:
                self.cube = data
                if self.cube.file.data.ndim == 3:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[0])) << u.Angstrom
                elif self.cube.file.data.ndim == 4:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
            elif type(data) == CRISPSequence:
                self.cube = data
                if self.cube.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[0])) << u.Angstrom
                elif self.cube.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
                if self.cube.list[1].file.data.ndim == 3:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[0])) << u.Angstrom
                elif self.cube.list[1].file.data.ndim == 4:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[1])) << u.Angstrom
        else:
            if type(data) == str:
                self.cube = CRISPNonU(filename=data, wcs=wcs, uncertainty=uncertainty, mask=mask)
                if self.cube.file.data.ndim == 3:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[0])) << u.Angstrom
                elif self.cube.file.data.ndim == 4:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
            elif type(data) == list:
                data = CRISP_sequence_constructor(data, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)
                self.cube = CRISPNonUSequence(files=data)
                if self.cube.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[0])) << u.Angstrom
                elif self.cube.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
                if self.cube.list[1].file.data.ndim == 3:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[0])) << u.Angstrom
                elif self.cube.list[1].file.data.ndim == 4:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[1])) << u.Angstrom
            elif type(data) == CRISPNonU:
                self.cube = data
                if self.cube.file.data.ndim == 3:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[0])) << u.Angstrom
                elif self.cube.file.data.ndim == 4:
                    self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
            elif type(data) == CRISPNonUSequence:
                self.cube = data
                if self.cube.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[0])) << u.Angstrom
                elif self.cube.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
                if self.cube.list[1].file.data.ndim == 3:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[0])) << u.Angstrom
                elif self.cube.list[1].file.data.ndim == 4:
                    self.wvls2 = self.cube.list[1].wave(np.arange(self.cube.list[1].shape[1])) << u.Angstrom

        if type(self.cube) == CRISP or type(self.cube) == CRISPNonU:
            self.fig = plt.figure(figsize=(8,10))
            try:
                self.ax1 = self.fig.add_subplot(1, 1, 1, projection=self.cube.wcs.dropaxis(-1))
            except:
                self.ax1 = self.fig.add_subplot(1, 1, 1, projection=SlicedLowLevelWCS(self.cube[0].wcs.low_level_wcs, 0))
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")

            ll = widgets.SelectionSlider(options=[np.round(l - np.median(self.wvls), decimals=2).value for l in self.wvls], description = f"{self.D} {self.l} [{self.aa}]")

            out1 = widgets.interactive_output(self._img_plot1, {"ll" : ll})

            display(widgets.HBox([ll]))
                
        elif type(self.cube) == CRISPSequence or type(self.cube) == CRISPNonUSequence:
            self.fig = plt.figure(figsize=(8,10))
            try:
                self.ax1 = self.fig.add_subplot(1, 2, 1, projection=self.cube.list[0].wcs.dropaxis(-1))
            except:
                self.ax1 = self.fig.add_subplot(1, 2, 1, projection=SlicedLowLevelWCS(self.cube.list[0][0].wcs.low_level_wcs, 0))
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")

            try:
                self.ax2 = self.fig.add_subplot(1, 2, 2, projection=self.cube.list[1].wcs.dropaxis(-1))
            except:
                self.ax2 = self.fig.add_subplot(1, 2, 2, projection=SlicedLowLevelWCS(self.cube.list[1][0].wcs.low_level_wcs, 0))
            self.ax2.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax2.set_xlabel("Helioprojective Longitude [arcsec]")

            ll1 = widgets.SelectionSlider(
                options=[np.round(l - np.median(self.wvls1), decimals=2).value for l in self.wvls1],
                description=fr"{self.D} {self.l}$_{1}$ [{self.aa}]",
                style={"description_width" : "initial"}
            )
            ll2 = widgets.SelectionSlider(
                options=[np.round(l - np.median(self.wvls2), decimals=2).value for l in self.wvls2],
                description=fr"{self.D} {self.l}$_{2}$ [{self.aa}]",
                style={"description_width" : "initial"}
            )

            out1 = widgets.interactive_output(self._img_plot2, {"ll1" : ll1, "ll2" : ll2})

            display(widgets.HBox([widgets.VBox([ll1, ll2])]))

        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        save_button = widgets.Button(description="Save")
        save_button.on_click(self._save)
        display(widgets.HBox([done_button, save_button]))
        widgets.interact(self._file_name, fn= widgets.Text(description="Filename to save as: ", style={"description_width" : "initial"}, layout=widgets.Layout(width="50%")))

    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)


    def _save(self, _):
        self.fig.savefig(self.filename, dpi=300)

    def _file_name(self, fn):
        self.filename = fn

    def _img_plot1(self, ll):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()

        ll_idx = int(np.where(np.round(self.wvls, decimals=2).value == np.round(np.median(self.wvls).value + ll, decimals=2))[0])

        try:
            data = self.cube.file.data[ll_idx].astype(np.float)
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        except:
            data = self.cube.file.data[0, ll_idx].astype(np.float)
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        try:
            el = self.cube.file.header["WDESC1"]
        except KeyError:
            el = self.cube.file.header["element"]
        self.ax1.set_title(fr"{el} {self.aa} {self.D} {self.l}$_{1}$ = {ll} {self.aa}")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="Intensity [DNs]")

    def _img_plot2(self, ll1, ll2):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar is not None:
            self.ax1.images[-1].colorbar.remove()

        if self.ax2.images == []:
            pass
        elif self.ax2.images[-1].colorbar is not None:
            self.ax2.images[-1].colorbar.remove()

        ll1_idx = int(np.where(np.round(self.wvls1, decimals=2).value == np.round(np.median(self.wvls1).value + ll1, decimals=2))[0])
        ll2_idx = int(np.where(np.round(self.wvls2, decimals=2).value == np.round(np.median(self.wvls2).value + ll2, decimals=2))[0])

        try:
            data = self.cube.list[0].file.data[ll1_idx].astype(np.float)
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        except:
            data = self.cube.list[0].file.data[0, ll1_idx].astype(np.float)
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        try:
            data = self.cube.list[1].file.data[ll2_idx].astype(np.float)
            data[data < 0] = np.nan
            im2 = self.ax2.imshow(data, cmap="Greys_r")
        except:
            data = self.cube.list[1].file.data[0, ll2_idx].astype(np.float)
            data[data < 0] = np.nan
            im2 = self.ax2.imshow(data, cmap="Greys_r")
        try:
            el1 = self.cube.list[0].file.header["WDESC1"]
            el2 = self.cube.list[1].file.header["WDESC1"]
        except KeyError:
            el1 = self.cube.list[0].file.header["element"]
            el2 = self.cube.list[1].file.header["element"]
        self.ax1.set_title(fr"{el1} {self.aa} {self.D} {self.l}$_{1}$ = {ll1} {self.aa}")
        self.ax2.set_title(fr"{el2} {self.aa} {self.D} {self.l}$_{2}$ = {ll2} {self.aa}")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="Intensity [DNs]")
        self.fig.colorbar(im2, ax=self.ax2, orientation="horizontal", label="Intensity [DNs]")

class SpectralTimeViewer:
    """
    Imaging spectroscopic viewer. SpectralTimeViewer should be used when one wants to click on points of an image and have the spectrum displayed for that point and the time series for a certain time range of observations. This works **exclusively** in Jupyter notebook but can be a nice data exploration tool. This viewer utilises the data structures defined in `crispy2.crisp` and has many variable options.

    :param data1: The data to explore, this is one spectral line. This is the only required argument to view the data.
    :type data1: list or CRISPSequence or CRISPNonUSequence
    :param data2: If there is a second set of data to explore.
    :type data2: list or CRISPSequence or CRISPNonUSequence
    :param wcs: A prescribed world coordinate system. If None, the world coordinate system is derived from the data. Default is None.
    :type wcs: astropy.wcs.WCS or None, optional
    :param uncertainty: The uncertainty in the intensity values of the data. Default is None.
    :type uncertainty: numpy.ndarray or None, optional
    :param mask: A mask to be used on the data. Default is None.
    :type mask: numpy.ndarray or None, optional
    :param nonu: Whether or not the spectral axis is non-uniform. Default is False.
    :type nonu: bool, optional

    :cvar coords: The coordinates selected to produce spectra.
    :type coords: list[tuple]
    :cvar px_coords: The coordinates selected to produce spectra in pixel space. This is important for indexing the data later to get the correct spectra.
    :type px_coords: list[tuple]
    :cvar shape_type: The spectra can be selected for a single point or for a box with specified dimensions with top-left corner where the user clicks. This attribute tells the user which point is described by which shape.
    :type shape_type: list[str]
    """
    def __init__(self, data1, data2=None, wcs=None, uncertainty=None, mask=None, nonu=False):
        plt.style.use("bmh")
        self.aa = html.unescape("&#8491;")
        self.l = html.unescape("&lambda;")
        self.a = html.unescape("&alpha;")
        self.D = html.unescape("&Delta;")
        shape = widgets.Dropdown(options=["point", "box"], value="point", description="Shape: ")
        if not nonu:
            if type(data1) == list:
                data1 = CRISP_sequence_constructor(data1, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)
                self.cube1 = CRISPSequence(files=data1)
                if self.cube1.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube1.list[0].wave(np.arange(self.cube1.list[0].shape[0])) << u.Angstrom
                elif self.cube1.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube1.list[0].wave(np.arange(self.cube1.list[0].shape[1])) << u.Angstrom
            elif type(data1) == CRISPSequence:
                self.cube1 = data1
                if self.cube1.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube1.list[0].wave(np.arange(self.cube1.list[0].shape[0]))
                elif self.cube1.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube1.list[0].wave(np.arange(self.cube1.list[0].shape[1]))
            if data2 == None:
                pass
            elif type(data2) == list:
                data2 = CRISP_sequence_constructor(data2, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)
                self.cube2 = CRISPSequence(files=data2)
                if self.cube2.list[0].file.data.ndim == 3:
                    self.wvls2 = self.cube2.list[0].wave(np.arange(self.cube2.list[0].shape[0]))
                elif self.cube2.list[0].file.data.ndim == 4:
                    self.wvls2 = self.cube2.list[0].wave(np.arange(self.cube2.list[0].shape[1]))
            elif type(data2) == CRISPSequence:
                self.cube2 = data2
                if self.cube2.list[0].file.data.ndim == 3:
                    self.wvls2 = self.cube2.list[0].wave(np.arange(self.cube2.list[0].shape[0]))
                elif self.cube2.list[0].file.data.ndim == 4:
                    self.wvls2 = self.cube2.list[0].wave(np.arange(self.cube2.list[0].shape[1]))
        else:
            if type(data1) == list:
                data1 = CRISP_sequence_constructor(data1, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)
                self.cube1 = CRISPNonUSequence(files=data1)
                if self.cube1.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube1.list[0].wave(np.arange(self.cube1.list[0].shape[0])) << u.Angstrom
                elif self.cube1.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube1.list[0].wave(np.arange(self.cube1.list[0].shape[1])) << u.Angstrom
            elif type(data1) == CRISPNonUSequence:
                self.cube1 = data
                if self.cube1.list[0].file.data.ndim == 3:
                    self.wvls1 = self.cube1.list[0].wave(np.arange(self.cube1.list[0].shape[0])) << u.Angstrom
                elif self.cube1.list[0].file.data.ndim == 4:
                    self.wvls1 = self.cube1.list[0].wave(np.arange(self.cube1.list[0].shape[1])) << u.Angstrom
            if data2 == None:
                pass
            elif type(data2) == list:
                data2 = CRISP_sequence_constructor(data2, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)
                self.cube2 = CRISPNonUSequence(files=data2)
                if self.cube2.list[0].file.data.ndim == 3:
                    self.wvls2 = self.cube2.list[0].wave(np.arange(self.cube2.list[0].shape[0]))
                elif self.cube2.list[0].file.data.ndim == 4:
                    self.wvls2 = self.cube2.list[0].wave(np.arange(self.cube2.list[0].shape[1]))
            elif type(data2) == CRISPNonUSequence:
                self.cube2 = data2
                if self.cube2.list[0].file.data.ndim == 3:
                    self.wvls2 = self.cube2.list[0].wave(np.arange(self.cube2.list[0].shape[0]))
                elif self.cube2.list[0].file.data.ndim == 4:
                    self.wvls2 = self.cube2.list[0].wave(np.arange(self.cube2.list[0].shape[1]))

        if data2 == None:
            self.fig = plt.figure(figsize=(8,10))
            self.gs = self.fig.add_gridspec(nrows=2, ncols=2)
            if self.cube1.list[0].file.data.ndim == 3:
                self.ax1 = self.fig.add_subplot(self.gs[0,0], projection=self.cube1.list[0].wcs.dropaxis(-1))
            elif self.cube1.list[0].file.data.ndim == 4:
                self.ax1 = self.fig.add_subplot(self.gs[0,0], projection=SlicedLowLevelWCS(self.cube1.list[0][0].wcs.low_level_wcs, 0))
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax2 = self.fig.add_subplot(self.gs[0,1])
            self.ax2.yaxis.set_label_position("right")
            self.ax2.yaxis.tick_right()
            self.ax2.set_ylabel("I [DNs]")
            self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax2.tick_params(direction="in")
            self.ax3 = self.fig.add_subplot(self.gs[1,:])
            self.ax3.set_ylabel("I [DNs]")
            self.ax3.set_xlabel("Time [UTC]")

            self.ll = widgets.SelectionSlider(options=[np.round(l - np.median(self.wvls1), decimals=2).value for l in self.wvls1], description = f"{self.D} {self.l} [{self.aa}]")

            self.t = widgets.IntSlider(value=0, min=0, max=len(self.cube1.list)-1, step=1, description="Time index: ", disabled=False)

            try:
                self.times1 = [date2num(f.file.header["DATE-AVG"]) for f in self.cube1.list]
            except KeyError:
                self.times1 = [date2num(f.file.header["date-obs"]+" "+f.file.header["time-obs"]) for f in self.cube1.list]
            
            out1 = widgets.interactive_output(self._img_plot1, {"ll" : self.ll, "t" : self.t})
            out2 = widgets.interactive_output(self._shape, {"opts" : shape})

            display(widgets.HBox([widgets.VBox([self.ll,self.t]), shape]))
                
        else:
            self.fig = plt.figure(figsize=(8,10))
            self.gs = self.fig.add_gridspec(nrows=3, ncols=2)
            try:
                self.ax1 = self.fig.add_subplot(self.gs[0,0], projection=self.cube1.list[0].wcs.dropaxis(-1))
            except:
                self.ax1 = self.fig.add_subplot(self.gs[0,0], projection=SlicedLowLevelWCS(self.cube1.list[0][0].wcs.low_level_wcs, 0))
            self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
            self.ax1.xaxis.set_label_position("top")
            self.ax1.xaxis.tick_top()

            try:
                self.ax2 = self.fig.add_subplot(self.gs[1,0], projection=self.cube2.list[0].wcs.dropaxis(-1))
            except:
                self.ax2 = self.fig.add_subplot(self.gs[1,0], projection=SlicedLowLevelWCS(self.cube2.list[0][0].wcs.low_level_wcs, 0))
            self.ax2.set_ylabel("Helioprojective Latitude [arcsec]")
            self.ax2.set_xlabel("Helioprojective Longitude [arcsec]")

            self.ax3 = self.fig.add_subplot(self.gs[0,1])
            self.ax3.yaxis.set_label_position("right")
            self.ax3.yaxis.tick_right()
            self.ax3.set_ylabel("Intensity [DNs]")
            self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax3.xaxis.set_label_position("top")
            self.ax3.xaxis.tick_top()
            self.ax3.tick_params(direction="in")

            self.ax4 = self.fig.add_subplot(self.gs[1,1])
            self.ax4.yaxis.set_label_position("right")
            self.ax4.yaxis.tick_right()
            self.ax4.set_ylabel("Intensity [DNs]")
            self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax4.tick_params(direction="in")
            
            self.ax5 = self.fig.add_subplot(self.gs[2,:])
            self.ax5.set_ylabel("Intensity [DNs]")
            self.ax5.set_xlabel("Time [UTC]")
            self.ax5b = self.ax5.twinx()
            self.ax5b.set_ylabel("Intensity [DNs]")

            self.ll1 = widgets.SelectionSlider(
                options=[np.round(l - np.median(self.wvls1), decimals=2).value for l in self.wvls1],
                description=fr"{self.aa} {self.D} {self.l}$_{1}$ [{self.aa}]",
                style={"description_width" : "initial"}
            )
            self.ll2 = widgets.SelectionSlider(
                options=[np.round(l - np.median(self.wvls2), decimals=2).value for l in self.wvls2],
                description=fr"{self.aa} {self.D} {self.l}$_{2}$ [{self.aa}]",
                style={"description_width" : "initial"}
            )

            self.t1 = widgets.IntSlider(value=0, min=0, max=len(self.cube1.list)-1, step=1, disabled=False, description=r"t$_{1}$ index: ")
            self.t2 = widgets.IntSlider(value=0, min=0, max=len(self.cube2.list)-1, step=1, disabled=False, description=r"t$_{2}$ index: ")
            
            try:
                self.times1 = [date2num(f.file.header["DATE-AVG"]) for f in self.cube1.list]
                self.times2 = [date2num(f.file.header["DATE-AVG"]) for f in self.cube2.list]
            except KeyError:
                self.times1 = [date2num(f.file.header["date-obs"]+" "+f.file.header["time-obs"]) for f in self.cube1.list]
                self.times2 = [date2num(f.file.header["date-obs"]+" "+f.file.header["time-obs"]) for f in self.cube2.list]
            
            out1 = widgets.interactive_output(self._img_plot2, {"ll1" : self.ll1, "ll2" : self.ll2, "t1" : self.t1, "t2" : self.t2})
            out2 = widgets.interactive_output(self._shape, {"opts" : shape})

            display(widgets.HBox([widgets.VBox([widgets.HBox([self.ll1, self.ll2]),widgets.HBox([self.t1, self.t2])]), shape]))

        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0

        self.receiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        x = widgets.IntText(value=1, min=1, max=self.cube1.list[0].shape[-1], description="x [pix]")
        y = widgets.IntText(value=1, min=1, max=self.cube1.list[0].shape[-2], description="y [pix]")
        outx = widgets.interactive_output(self._boxx, {"x" : x})
        outy = widgets.interactive_output(self._boxy, {"y" : y})
        display(widgets.HBox([x, y]))

        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description="Clear")
        clear_button.on_click(self._clear)
        save_button = widgets.Button(description="Save")
        save_button.on_click(self._save)
        display(widgets.HBox([done_button, clear_button, save_button]))
        widgets.interact(self._file_name, fn= widgets.Text(description="Filename to save as: ", style={"description_width" : "initial"}, layout=widgets.Layout(width="50%")))

    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode != "":
            return

        if not hasattr(self, "cube2"):
            if self.shape == "point":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                centre_coord = int(event.ydata), int(event.xdata)
                self.px_coords.append(centre_coord)
                self.shape_type.append("point")
                circ = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
                self.ax1.add_patch(circ)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                px = self.cube1.list[self.t.value].to_lonlat(*centre_coord) << u.arcsec
                if self.cube1.list[0].file.data.ndim == 3:
                    self.ax2.plot(self.wvls1, self.cube1.list[self.t.value].file.data[:, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube1.list[0].file.data.ndim == 4:
                    self.ax2.plot(self.wvls1, self.cube1.list[self.t.value].file.data[0, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax2.legend()
                ll_idx = int(np.where(np.round(self.wvls1, decimals=2).value == np.round(np.median(self.wvls1).value + self.ll.value, decimals=2))[0])
                if self.cube1.list[0].file.data.ndim == 3:
                    i_time1 = [f.file.data[ll_idx, centre_coord[0], centre_coord[1]] for f in self.cube1.list]
                    self.ax3.plot(self.times1, i_time1,  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube1.list[0].file.data.ndim == 4:
                    i_time1 = [f.file.data[0, ll_idx, centre_coord[0], centre_coord[1]] for f in self.cube1.list]
                    self.ax3.plot(self.times1, i_time1,  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax3.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
                for label in self.ax3.get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
                self.coords.append(px)
                self.colour_idx += 1
                self.fig.canvas.draw()
            elif self.shape == "box":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                box_anchor = int(event.ydata), int(event.xdata)
                self.px_coords.append(box_anchor)
                self.shape_type.append("box")
                # obtain the coordinates of the box on a grid with pixels the size of the box to make sure there is not copies of the same box
                box_coord = box_anchor[0] // self.boxy, box_anchor[1] // self.boxx
                if box_coord in self.box_coords:
                    coords = [p.get_xy() for p in self.ax1.patches]
                    for p in self.ax.patches:
                        if p.get_xy() == box_anchor:
                            p.remove()
                    
                    idx = self.box_coords.index(box_coord)
                    del self.box_coords[idx]
                    del self.px_coords[idx]
                    del self.shape_type[idx]
                    del self.coords[idx]
                    return
                
                self.coords.append(self.cube1.list[0].to_lonlat(*box_anchor) << u.arcsec)
                rect = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
                rect.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                self.ax1.add_patch(rect)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt = self.ax1.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                if self.cube1.list[0].file.data.ndim == 3:
                    self.ax2.plot(self.wvls1, np.mean(self.cube1.list[self.t.value].file.data[:,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube1.list[0].file.data.ndim == 4:
                    self.ax2.plot(self.wvls1, np.mean(self.cube1.list[self.t.value].file.data[0, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax2.legend()
                ll_idx = int(np.where(np.round(self.wvls1, decimals=2).value == np.round(np.median(self.wvls1).value + self.ll.value, decimals=2))[0])
                if self.cube1.list[0].file.data.ndim == 3:
                    i_time1 = [np.mean(f.file.data[ll_idx,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube1.list]
                    self.ax3.plot(self.times1, i_time1,  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube1.list[0].file.data.ndim == 4:
                    i_time1 = [np.mean(f.file.data[0, ll_idx,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube1.list]
                    self.ax3.plot(self.times1, i_time1,  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax3.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
                for label in self.ax3.get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
                self.colour_idx += 1
                self.fig.canvas.draw()
        else:
            if self.shape == "point":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                centre_coord = int(event.ydata), int(event.xdata) #with WCS, the event data is returned in pixels so we don't need to do the conversion from real world but rather to real world later on
                self.px_coords.append(centre_coord)
                circ1 = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
                circ2 = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
                self.ax1.add_patch(circ1)
                self.ax2.add_patch(circ2)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt_1 = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt_2 = self.ax2.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt_1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                txt_2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                px = self.cube1.list[0].to_lonlat(*centre_coord) << u.arcsec
                if self.cube1.list[0].file.data.ndim == 3:
                    self.ax3.plot(self.wvls1, self.cube1.list[self.t1.value].file.data[:, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube1.list[0].file.data.ndim == 4:
                    self.ax3.plot(self.wvls1, self.cube1.list[self.t1.value].file.data[0, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                if self.cube2.list[0].file.data.ndim == 3:
                    self.ax4.plot(self.wvls2, self.cube2.list[self.t2.value].file.data[:, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube2.list[0].file.data.ndim == 4:
                    self.ax4.plot(self.wvls2, self.cube2.list[self.t2.value].file.data[0, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax3.legend()
                self.ax4.legend()
                ll_idx1 = int(np.where(np.round(self.wvls1, decimals=2).value == np.round(np.median(self.wvls1).value + self.ll1.value, decimals=2))[0])
                ll_idx2 = int(np.where(np.round(self.wvls2, decimals=2).value == np.round(np.median(self.wvls2).value + self.ll2.value, decimals=2))[0])
                if self.cube1.list[0].file.data.ndim == 3:
                    i_time1 = [f.file.data[ll_idx1, centre_coord[0], centre_coord[1]] for f in self.cube1.list]
                elif self.cube1.list[0].file.data.ndim == 4:
                    i_time1 = [f.file.data[0, ll_idx1, centre_coord[0], centre_coord[1]] for f in self.cube1.list]
                if self.cube2.list[0].file.data.ndim == 3:
                    i_time2 = [f.file.data[ll_idx2, centre_coord[0], centre_coord[1]] for f in self.cube2.list]
                elif self.cube2.list[0].file.data.ndim == 4:
                    i_time2 = [f.file.data[0, ll_idx2, centre_coord[0], centre_coord[1]] for f in self.cube2.list]
                self.ax5.plot(self.times1, i_time1,  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax5b.plot(self.times2, i_time2, linestyle="--",  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax5.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
                for label in self.ax5.get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
                self.coords.append(px)
                self.colour_idx += 1
                self.fig.canvas.draw()
            elif self.shape == "box":
                if self.colour_idx > len(pt_bright_cycler)-1:
                    self.colour_idx = 0
                    self.n += 1
                box_anchor = int(event.ydata), int(event.xdata)
                self.px_coords.append(box_anchor)
                self.shape_type.append("box")
                # obtain the coordinates of the box on a grid with pixels the size of the box to make sure there is not copies of the same box
                box_coord = box_anchor[0] // self.boxy, box_anchor[1] // self.boxx
                if box_coord in self.box_coords:
                    coords = [p.get_xy() for p in self.ax1.patches]
                    for p in self.ax.patches:
                        if p.get_xy() == box_anchor:
                            p.remove()
                    
                    idx = self.box_coords.index(box_coord)
                    del self.box_coords[idx]
                    del self.px_coords[idx]
                    del self.shape_type[idx]
                    del self.coords[idx]
                    return
                
                self.coords.append(self.cube1.list[0].to_lonlat(*box_anchor) << u.arcsec)
                rect1 = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
                rect1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                rect2 = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
                rect2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                self.ax1.add_patch(rect1)
                self.ax2.add_patch(rect2)
                font = {
                    "size" : 12,
                    "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
                }
                txt1 = self.ax1.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                txt2 = self.ax2.text(box_anchor[1]-50, box_anchor[0]-1, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
                txt2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
                if self.cube1.list[0].file.data.ndim == 3:
                    self.ax3.plot(self.wvls1, np.mean(self.cube1.list[self.t1.value].file.data[:, box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx], axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube1.list[0].file.data.ndim == 4:
                    self.ax3.plot(self.wvls1, np.mean(self.cube1.list[self.t1.value].file.data[0, :, box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx], axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                if self.cube2.list[0].file.data.ndim == 3:
                    self.ax4.plot(self.wvls2, np.mean(self.cube2.list[self.t2.value].file.data[:, box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx], axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                elif self.cube2.list[0].file.data.ndim == 4:
                    self.ax4.plot(self.wvls2, np.mean(self.cube2.list[self.t2.value].file.data[0, :, box_anchor[0]:box_anchor[0]+self.boxy, box_anchor[1]:box_anchor[1]+self.boxx], axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax3.legend()
                self.ax4.legend()
                ll_idx1 = int(np.where(np.round(self.wvls1, decimals=2).value == np.round(np.median(self.wvls1).value + self.ll1.value, decimals=2))[0])
                ll_idx2 = int(np.where(np.round(self.wvls2, decimals=2).value == np.round(np.median(self.wvls2).value + self.ll2.value, decimals=2))[0])
                if self.cube1.list[0].file.data.ndim == 3:
                    i_time1 = [np.mean(f.file.data[ll_idx1,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube1.list]
                elif self.cube1.list[0].file.data.ndim == 4:
                    i_time1 = [np.mean(f.file.data[0, ll_idx1,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube1.list]
                if self.cube2.list[0].file.data.ndim == 3:
                    i_time2 = [np.mean(f.file.data[ll_idx2,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube2.list]
                elif self.cube2.list[0].file.data.ndim == 4:
                    i_time2 = [np.mean(f.file.data[0, ll_idx2,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube2.list]
                self.ax5.plot(self.times1, i_time1,  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax5b.plot(self.times2, i_time2, linestyle="--",  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
                self.ax5.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
                for label in self.ax5.get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
                self.colour_idx += 1
                self.fig.canvas.draW()

    def _shape(self, opts):
        self.shape = opts

    def _boxx(self, x):
        self.boxx = x

    def _boxy(self, y):
        self.boxy = y

    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)

    def _clear(self, _):
        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0
        if not hasattr(self, "cube2"):
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax1.texts) > 0:
                for t in self.ax1.texts:
                    t.remove()
            self.ax2.clear()
            self.ax2.set_ylabel("Intensity [DNs]")
            self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax3.clear()
            self.ax3.set_ylabel("I [DNs]")
            self.ax3.set_xlabel("Time [UTC]")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            while len(self.ax1.patches) > 0:
                for p in self.ax1.patches:
                    p.remove()
            while len(self.ax2.patches) > 0:
                for p in self.ax2.patches:
                    p.remove()
            while len(self.ax1.texts) > 0:
                for t in self.ax1.texts:
                    t.remove()
            while len(self.ax2.texts) > 0:
                for t in self.ax2.texts:
                    t.remove()
            self.ax3.clear()
            self.ax3.set_ylabel("Intensity [DNs]")
            self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax4.clear()
            self.ax4.set_ylabel("Intensity [DNs]")
            self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
            self.ax5.clear()
            self.ax5.set_ylabel("I [DNs]")
            self.ax5.set_xlabel("Time [UTC]")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _save(self, _):
        self.fig.savefig(self.filename, dpi=300)

    def _file_name(self, fn):
        self.filename = fn

    def _img_plot1(self, ll, t):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar != None:
            self.ax1.images[-1].colorbar.remove()

        ll_idx = int(np.where(np.round(self.wvls1, decimals=2).value == np.round(np.median(self.wvls1).value + ll, decimals=2))[0])

        try:
            data = self.cube1.list[t].file.data[ll_idx].astype(np.float)
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        except:
            data = self.cube1.list[t].file.data[0, ll_idx].astype(np.float)
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        try:
            el = self.cube1.list[0].file.header["WDESC1"]
        except KeyError:
            el = self.cube1.list[0].file.header["element"]
        self.ax1.set_title(fr"{el} {self.aa} {self.D} {self.l}$_{1}$ = {ll} {self.aa}")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="Intensity [DNs]")

    def _img_plot2(self, ll1, ll2, t1, t2):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar != None:
            self.ax1.images[-1].colorbar.remove()

        if self.ax2.images == []:
            pass
        elif self.ax2.images[-1].colorbar != None:
            self.ax2.images[-1].colorbar.remove()

        ll1_idx = int(np.where(np.round(self.wvls1, decimals=2).value == np.round(np.median(self.wvls1).value + ll1, decimals=2))[0])
        ll2_idx = int(np.where(np.round(self.wvls2, decimals=2).value == np.round(np.median(self.wvls2).value + ll2, decimals=2))[0])

        try:
            data1 = self.cube1.list[t1].file.data[ll1_idx].astype(np.float)
            data1[data1 < 0] = np.nan
            im1 = self.ax1.imshow(data1, cmap="Greys_r")
        except:
            data1 = self.cube1.list[t1].file.data[0, ll1_idx].astype(np.float)
            data1[data1 < 0] = np.nan
            im1 = self.ax1.imshow(data1, cmap="Greys_r")
        try:
            data2 = self.cube2.list[t2].file.data[ll2_idx].astype(np.float)
            data2[data2 < 0] = np.nan
            im2 = self.ax2.imshow(data2, cmap="Greys_r")
        except:
            data2 = self.cube2.list[t2].file.data[0, ll2_idx].astype(np.float)
            data2[data2 < 0] = np.nan
            im2 = self.ax2.imshow(data2, cmap="Greys_r")
        try:
            el1 = self.cube1.list[0].file.header["WDESC1"]
            el2 = self.cube2.list[0].file.header["WDESC1"]
        except KeyError:
            el1 = self.cube1.list[0].file.header["element"]
            el2 = self.cube2.list[0].file.header["element"]
        self.ax1.set_title(fr"{el1} {self.aa} {self.D} {self.l}$_{1}$ = {ll1} {self.aa}")
        self.ax2.set_title(fr"{el2} {self.aa} {self.D} {self.l}$_{2}$ = {ll2} {self.aa}")
        self.fig.colorbar(im1, ax=self.ax1, orientation="vertical", label="Intensity [DNs]")
        self.fig.colorbar(im2, ax=self.ax2, orientation="vertical", label="Intensity [DNs]")

class PolarimetricViewer:
    """
    This class defines the visualisation tool for exploring narrowband imaging spectropolarimetric data. This currently is only developed to look at one spectral line at a time. The functionality is similar to the ``SpectralViewer`` defines above but with an added Stokes parameter that can be changed.

    :param data: The data to explore, this is one spectral line. This is the only required argument to view the data.
    :type data: str or CRISP or CRISPNonU
    :param wcs: A prescribed world coordinate system. If None, the world coordinate system is derived from the data. Default is None.
    :type wcs: astropy.wcs.WCS or None, optional
    :param uncertainty: The uncertainty in the intensity values of the data. Default is None.
    :type uncertainty: numpy.ndarray or None, optional
    :param mask: A mask to be used on the data. Default is None.
    :type mask: numpy.ndarray or None, optional
    :param nonu: Whether or not the spectral axis is non-uniform. Default is False.
    :type nonu: bool, optional

    :cvar coords: The coordinates selected to produce spectra.
    :type coords: list[tuple]
    :cvar px_coords: The coordinates selected to produce spectra in pixel space. This is important for indexing the data later to get the correct spectra.
    :type px_coords: list[tuple]
    :cvar shape_type: The spectra can be selected for a single point or for a box with specified dimensions with top-left corner where the user clicks. This attribute tells the user which point is described by which shape.
    :type shape_type: list[str]
    """

    def __init__(self, data, wcs=None, uncertainty=None, mask=None, nonu=False):
        plt.style.use("bmh")
        self.aa = html.unescape("&#8491;")
        self.l = html.unescape("&lambda;")
        self.a = html.unescape("&alpha;")
        self.D = html.unescape("&Delta;")
        shape = widgets.Dropdown(options=["point", "box"], value="point", description="Shape: ")
        if not nonu:
            if type(data) == str:
                self.cube = CRISP(filename=data, wcs=wcs, uncertainty=uncertainty, mask=mask)
                self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
            elif type(data) == CRISP:
                self.cube = data
                self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
        else:
            if type(data) == str:
                self.cube = CRISPNonU(filename=data, wcs=wcs, uncertainty=uncertainty, mask=mask)
                self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom
            elif type(data) == CRISPNonU:
                self.cube = data
                self.wvls = self.cube.wave(np.arange(self.cube.shape[1])) << u.Angstrom

        self.fig = plt.figure(figsize=(12,10))
        self.gs = self.fig.add_gridspec(nrows=2, ncols=6)
        self.ax1 = self.fig.add_subplot(self.gs[:,:2], projection=SlicedLowLevelWCS(self.cube[0].wcs.low_level_wcs,0))
        self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
        self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
        self.ax2 = self.fig.add_subplot(self.gs[0,2:4])
        self.ax2.set_ylabel("I [DNs]")
        self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax2.tick_params(direction="in")
        self.ax3 = self.fig.add_subplot(self.gs[0,4:])
        self.ax3.yaxis.set_label_position("right")
        self.ax3.yaxis.tick_right()
        self.ax3.set_ylabel("Q [DNs]")
        self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax4 = self.fig.add_subplot(self.gs[1,2:4])
        self.ax4.set_ylabel("U [DNs]")
        self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax5 = self.fig.add_subplot(self.gs[1,4:])
        self.ax5.yaxis.set_label_position("right")
        self.ax5.yaxis.tick_right()
        self.ax5.set_ylabel("V [DNs]")
        self.ax5.set_xlabel(f"{self.l} [{self.aa}]")

        ll = widgets.SelectionSlider(options=[np.round(l - np.median(self.wvls), decimals=2).value for l in self.wvls], description = f"{self.D} {self.l} [{self.aa}]")

        s = widgets.Dropdown(options=["I", "Q", "U", "V"], description="Stokes: ")

        out1 = widgets.interactive_output(self._img_plot1, {"ll" : ll, "s" : s})
        out2 = widgets.interactive_output(self._shape, {"opts" : shape})

        display(widgets.HBox([widgets.VBox([ll, s]), shape]))
                
        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0

        self.receiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        x = widgets.IntText(value=1, min=1, max=self.cube.shape[-1], description="x [pix]")
        y = widgets.IntText(value=1, min=1, max=self.cube.shape[-2], description="y [pix]")
        outx = widgets.interactive_output(self._boxx, {"x" : x})
        outy = widgets.interactive_output(self._boxy, {"y" : y})
        display(widgets.HBox([x, y]))

        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description="Clear")
        clear_button.on_click(self._clear)
        save_button = widgets.Button(description="Save")
        save_button.on_click(self._save)
        display(widgets.HBox([done_button, clear_button, save_button]))
        widgets.interact(self._file_name, fn= widgets.Text(description="Filename to save as: ", style={"description_width" : "initial"}, layout=widgets.Layout(width="50%")))

    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode != "":
            return

        if self.shape == "point":
            if self.colour_idx > len(pt_bright_cycler)-1:
                self.colour_idx = 0
                self.n += 1
            centre_coord = int(event.ydata), int(event.xdata)
            self.px_coords.append(centre_coord)
            self.shape_type.append("point")
            circ = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
            self.ax1.add_patch(circ)
            font = {
                "size" : 12,
                "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
            }
            txt = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            px = self.cube.to_lonlat(*centre_coord) << u.arcsec
            self.ax2.plot(self.wvls, self.cube.file.data[0, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax2.legend()
            self.ax3.plot(self.wvls, self.cube.file.data[1, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax4.plot(self.wvls, self.cube.file.data[2, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax5.plot(self.wvls, self.cube.file.data[3, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.coords.append(px)
            self.colour_idx += 1
            self.fig.canvas.draw()
        elif self.shape == "box":
            if self.colour_idx > len(pt_bright_cycler)-1:
                self.colour_idx = 0
                self.n += 1
            box_anchor = int(event.ydata), int(event.xdata)
            self.px_coords.append(box_anchor)
            self.shape_type.append("box")
            # obtain the coordinates of the box on a grid with pixels the size of the box to make sure there is not copies of the same box
            box_coord = box_anchor[0] // self.boxy, box_anchor[1] // self.boxx
            if box_coord in self.box_coords:
                coords = [p.get_xy() for p in self.ax.patches]
                for p in self.ax.patches:
                    if p.get_xy() == box_anchor:
                        p.remove()
                
                idx = self.box_coords.index(box_coord)
                del self.box_coords[idx]
                del self.px_coords[idx]
                del self.shape_type[idx]
                del self.coords[idx]
                return
            
            self.coords.append(self.cube.to_lonlat(*box_anchor) << u.arcsec)
            rect = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
            rect.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            self.ax1.add_patch(rect)
            font = {
                "size" : 12,
                "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
            }
            txt = self.ax1.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            self.ax2.plot(self.wvls, np.mean(self.cube.file.data[0, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax3.plot(self.wvls, np.mean(self.cube.file.data[1, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax4.plot(self.wvls, np.mean(self.cube.file.data[2, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax5.plot(self.wvls, np.mean(self.cube.file.data[3, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax2.legend()
            self.colour_idx += 1
            self.fig.canvas.draw()

    def _shape(self, opts):
        self.shape = opts

    def _boxx(self, x):
        self.boxx = x

    def _boxy(self, y):
        self.boxy = y

    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)

    def _clear(self, _):
        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0
        while len(self.ax1.patches) > 0:
            for p in self.ax1.patches:
                p.remove()
        while len(self.ax1.texts) > 0:
            for t in self.ax1.texts:
                t.remove()
        self.ax2.clear()
        self.ax2.set_ylabel("I [DNs]")
        self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax3.clear()
        self.ax3.set_ylabel("Q [DNs]")
        self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax4.clear()
        self.ax4.set_ylabel("U [DNs]")
        self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax5.clear()
        self.ax5.set_ylabel("V [DNs]")
        self.ax5.set_xlabel(f"{self.l} [{self.aa}]")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _save(self, _):
        self.fig.savefig(self.filename, dpi=300)

    def _file_name(self, fn):
        self.filename = fn

    def _img_plot1(self, ll, s):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar != None:
            self.ax1.images[-1].colorbar.remove()

        ll_idx = int(np.where(np.round(self.wvls, decimals=2).value == np.round(np.median(self.wvls).value + ll, decimals=2))[0])

        stokes_dict = {
            "I" : 0,
            "Q" : 1,
            "U" : 2,
            "V" : 3
        }
        
        s_idx = stokes_dict[s]

        if s_idx == 0:
            data = self.cube.file.data[s_idx, ll_idx]
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        elif s_idx == 1 or s_idx == 2:
            im1 = self.ax1.imshow(self.cube.file.data[s_idx, ll_idx], cmap="Greys_r", vmin=-10, vmax=10)
        else:
            im1 = self.ax1.imshow(self.cube.file.data[s_idx, ll_idx], cmap="Greys_r", vmin=-50, vmax=50)
        try:
            el = self.cube.file.header["WDESC1"]
        except KeyError:
            el = self.cube.file.header["element"]
        self.ax1.set_title(fr"{el} {self.aa} {self.D} {self.l}$_{1}$ = {ll} {self.aa}")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="Intensity [DNs]")

class PolarimetricTimeViewer:
    """
    This class defines the visualisation tool for exploring a time series narrowband imaging spectropolarimetric data. This currently is only developed to look at one spectral line at a time. The functionality is similar to the ``SpectralTimeViewer`` defines above but with an added Stokes parameter that can be changed.

    :param data: The data to explore, this is one spectral line. This is the only required argument to view the data.
    :type data: str or CRISPSequence or CRISPNonUSequence
    :param wcs: A prescribed world coordinate system. If None, the world coordinate system is derived from the data. Default is None.
    :type wcs: astropy.wcs.WCS or None, optional
    :param uncertainty: The uncertainty in the intensity values of the data. Default is None.
    :type uncertainty: numpy.ndarray or None, optional
    :param mask: A mask to be used on the data. Default is None.
    :type mask: numpy.ndarray or None, optional
    :param nonu: Whether or not the spectral axis is non-uniform. Default is False.
    :type nonu: bool, optional

    :cvar coords: The coordinates selected to produce spectra.
    :type coords: list[tuple]
    :cvar px_coords: The coordinates selected to produce spectra in pixel space. This is important for indexing the data later to get the correct spectra.
    :type px_coords: list[tuple]
    :cvar shape_type: The spectra can be selected for a single point or for a box with specified dimensions with top-left corner where the user clicks. This attribute tells the user which point is described by which shape.
    :type shape_type: list[str]
    """

    def __init__(self, data, wcs=None, uncertainty=None, mask=None, nonu=False):
        plt.style.use("bmh")
        self.aa = html.unescape("&#8491;")
        self.l = html.unescape("&lambda;")
        self.a = html.unescape("&alpha;")
        self.D = html.unescape("&Delta;")
        shape = widgets.Dropdown(options=["point", "box"], value="point", description="Shape: ")
        if not nonu:
            if type(data) == list:
                data = CRISP_sequence_constructor(data, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)
                self.cube = CRISPSequence(files=data)
                self.wvls = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
            elif type(data) == CRISPSequence:
                self.cube = data
                self.wvls = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
        else:
            if type(data) == list:
                data = CRISP_sequence_constructor(data, wcs=wcs, uncertainty=uncertainty, mask=mask, nonu=nonu)
                self.cube = CRISPNonUSequence(files=data)
                self.wvls = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom
            elif type(data) == CRISPNonUSequence:
                self.cube = data
                self.wvls = self.cube.list[0].wave(np.arange(self.cube.list[0].shape[1])) << u.Angstrom

        self.fig = plt.figure(figsize=(8,10))
        self.gs = self.fig.add_gridspec(nrows=4, ncols=4)
        self.ax1 = self.fig.add_subplot(self.gs[:2,:2], projection=SlicedLowLevelWCS(self.cube.list[0][0].wcs.low_level_wcs,0))
        self.ax1.set_ylabel("Helioprojective Latitude [arcsec]")
        self.ax1.set_xlabel("Helioprojective Longitude [arcsec]")
        self.ax2 = self.fig.add_subplot(self.gs[0,2])
        self.ax2.set_ylabel("I [DNs]")
        self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax2.tick_params(direction="in")
        self.ax3 = self.fig.add_subplot(self.gs[0,3])
        self.ax3.yaxis.set_label_position("right")
        self.ax3.yaxis.tick_right()
        self.ax3.set_ylabel("Q [DNs]")
        self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax4 = self.fig.add_subplot(self.gs[1,2])
        self.ax4.set_ylabel("U [DNs]")
        self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax5 = self.fig.add_subplot(self.gs[1,3])
        self.ax5.yaxis.set_label_position("right")
        self.ax5.yaxis.tick_right()
        self.ax5.set_ylabel("V [DNs]")
        self.ax5.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax6 = self.fig.add_subplot(self.gs[2:,:2])
        self.ax6.set_ylabel("I [DNs]")
        self.ax6.set_xlabel("Time [UTC]")
        self.ax7 = self.fig.add_subplot(self.gs[2:,2:])
        self.ax7.set_ylabel("Q & U [DNs]")
        self.ax7.set_xlabel("Time [UTC]")
        self.ax7b = self.ax7.twinx()
        self.ax7b.set_ylabel("V [DNs]")

        self.ll = widgets.SelectionSlider(options=[np.round(l - np.median(self.wvls), decimals=2).value for l in self.wvls], description = f"{self.D} {self.l} [{self.aa}]")

        s = widgets.Dropdown(options=["I", "Q", "U", "V"], description="Stokes: ")

        self.t = widgets.IntSlider(value=0, min=0, max=len(self.cube.list)-1, step=1, description="Time index: ", disabled=False)

        try:
            self.times1 = [date2num(f.file.header["DATE-AVG"]) for f in self.cube.list]
        except KeyError:
            self.times1 = [date2num(f.file.header["date-obs"]+" "+f.file.header["time-obs"]) for f in self.cube.list]

        out1 = widgets.interactive_output(self._img_plot1, {"ll" : self.ll, "s" : s, "t" : self.t})
        out2 = widgets.interactive_output(self._shape, {"opts" : shape})

        display(widgets.HBox([widgets.VBox([self.ll, s]), widgets.VBox([self.t, shape])]))
                
        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0

        self.receiver = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        x = widgets.IntText(value=1, min=1, max=self.cube.list[0].shape[-1], description="x [pix]")
        y = widgets.IntText(value=1, min=1, max=self.cube.list[0].shape[-2], description="y [pix]")
        outx = widgets.interactive_output(self._boxx, {"x" : x})
        outy = widgets.interactive_output(self._boxy, {"y" : y})
        display(widgets.HBox([x, y]))

        done_button = widgets.Button(description="Done")
        done_button.on_click(self._disconnect_matplotlib)
        clear_button = widgets.Button(description="Clear")
        clear_button.on_click(self._clear)
        save_button = widgets.Button(description="Save")
        save_button.on_click(self._save)
        display(widgets.HBox([done_button, clear_button, save_button]))
        widgets.interact(self._file_name, fn= widgets.Text(description="Filename to save as: ", style={"description_width" : "initial"}, layout=widgets.Layout(width="50%")))

    def _on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode != "":
            return

        if self.shape == "point":
            if self.colour_idx > len(pt_bright_cycler)-1:
                self.colour_idx = 0
                self.n += 1
            centre_coord = int(event.ydata), int(event.xdata)
            self.px_coords.append(centre_coord)
            self.shape_type.append("point")
            circ = patches.Circle(centre_coord[::-1], radius=10, facecolor=list(pt_bright_cycler)[self.colour_idx]["color"], edgecolor="k", linewidth=1)
            self.ax1.add_patch(circ)
            font = {
                "size" : 12,
                "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
            }
            txt = self.ax1.text(centre_coord[1]+20, centre_coord[0]+10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            px = self.cube.list[0].to_lonlat(*centre_coord) << u.arcsec
            self.ax2.plot(self.wvls, self.cube.list[self.t.value].file.data[0, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax2.legend()
            self.ax3.plot(self.wvls, self.cube.list[self.t.value].file.data[1, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax4.plot(self.wvls, self.cube.list[self.t.value].file.data[2, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax5.plot(self.wvls, self.cube.list[self.t.value].file.data[3, :, centre_coord[0], centre_coord[1]], marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            ll_idx = int(np.where(np.round(self.wvls, decimals=2).value == np.round(np.median(self.wvls).value + self.ll.value, decimals=2))[0])
            i_time = [f.file.data[0, ll_idx, centre_coord[0], centre_coord[1]] for f in self.cube.list]
            q_time = [f.file.data[1, ll_idx, centre_coord[0], centre_coord[1]] for f in self.cube.list]
            u_time = [f.file.data[2, ll_idx, centre_coord[0], centre_coord[1]] for f in self.cube.list]
            v_time = [f.file.data[3, ll_idx, centre_coord[0], centre_coord[1]] for f in self.cube.list]
            self.ax6.plot(self.times1, i_time,  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax7.plot(self.times1, q_time,  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax7.plot(self.times1, u_time,  linestyle="--", marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax7b.plot(self.times1, v_time,  linestyle="-.", marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.coords.append(px)
            self.colour_idx += 1
            self.fig.canvas.draw()
        elif self.shape == "box":
            if self.colour_idx > len(pt_bright_cycler)-1:
                self.colour_idx = 0
                self.n += 1
            box_anchor = int(event.ydata), int(event.xdata)
            self.px_coords.append(box_anchor)
            self.shape_type.append("box")
            # obtain the coordinates of the box on a grid with pixels the size of the box to make sure there is not copies of the same box
            box_coord = box_anchor[0] // self.boxy, box_anchor[1] // self.boxx
            if box_coord in self.box_coords:
                coords = [p.get_xy() for p in self.ax.patches]
                for p in self.ax.patches:
                    if p.get_xy() == box_anchor:
                        p.remove()
                
                idx = self.box_coords.index(box_coord)
                del self.box_coords[idx]
                del self.px_coords[idx]
                del self.shape_type[idx]
                del self.coords[idx]
                return
            
            self.coords.append(self.cube.to_lonlat(*box_anchor) << u.arcsec)
            rect = patches.Rectangle(box_anchor[::-1], self.boxx, self.boxy, linewidth=2, edgecolor=list(pt_bright_cycler)[self.colour_idx]["color"], facecolor="none")
            rect.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            self.ax1.add_patch(rect)
            font = {
                "size" : 12,
                "color" : list(pt_bright_cycler)[self.colour_idx]["color"]
            }
            txt = self.ax1.text(box_anchor[1]-50, box_anchor[0]-10, s=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", fontdict=font)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
            self.ax2.plot(self.wvls, np.mean(self.cube.file.data[0, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax3.plot(self.wvls, np.mean(self.cube.file.data[1, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax4.plot(self.wvls, np.mean(self.cube.file.data[2, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax5.plot(self.wvls, np.mean(self.cube.file.data[3, :,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx],axis=(1,2)), marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            i_time = [np.mean(f.file.data[0, ll_idx2,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube.list]
            q_time = [np.mean(f.file.data[1, ll_idx2,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube.list]
            u_time = [np.mean(f.file.data[2, ll_idx2,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube.list]
            v_time = [np.mean(f.file.data[3, ll_idx2,box_anchor[0]:box_anchor[0]+self.boxy,box_anchor[1]:box_anchor[1]+self.boxx]) for f in self.cube.list]
            self.ax6.plot(self.times1, i_time,  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax7.plot(self.times1, q_time,  marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax7.plot(self.times1, u_time,  linestyle="--", marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax7b.plot(self.times1, v_time,  linestyle="-.", marker=Line2D.filled_markers[self.colour_idx+self.n*len(pt_bright_cycler)], label=f"{self.colour_idx+1+(self.n*len(pt_bright_cycler))}", c=list(pt_bright_cycler)[self.colour_idx]["color"])
            self.ax2.legend()
            self.colour_idx += 1
            self.fig.canvas.draw()

    def _shape(self, opts):
        self.shape = opts

    def _boxx(self, x):
        self.boxx = x

    def _boxy(self, y):
        self.boxy = y

    def _disconnect_matplotlib(self, _):
        self.fig.canvas.mpl_disconnect(self.receiver)

    def _clear(self, _):
        self.coords = []
        self.px_coords = []
        self.shape_type = []
        self.box_coords = []
        self.colour_idx = 0
        self.n = 0
        while len(self.ax1.patches) > 0:
            for p in self.ax1.patches:
                p.remove()
        while len(self.ax1.texts) > 0:
            for t in self.ax1.texts:
                t.remove()
        self.ax2.clear()
        self.ax2.set_ylabel("I [DNs]")
        self.ax2.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax3.clear()
        self.ax3.set_ylabel("Q [DNs]")
        self.ax3.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax4.clear()
        self.ax4.set_ylabel("U [DNs]")
        self.ax4.set_xlabel(f"{self.l} [{self.aa}]")
        self.ax5.clear()
        self.ax5.set_ylabel("V [DNs]")
        self.ax5.set_xlabel(f"{self.l} [{self.aa}]")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _save(self, _):
        self.fig.savefig(self.filename, dpi=300)

    def _file_name(self, fn):
        self.filename = fn

    def _img_plot1(self, ll, s, t):
        if self.ax1.images == []:
            pass
        elif self.ax1.images[-1].colorbar != None:
            self.ax1.images[-1].colorbar.remove()

        ll_idx = int(np.where(np.round(self.wvls, decimals=2).value == np.round(np.median(self.wvls).value + ll, decimals=2))[0])

        stokes_dict = {
            "I" : 0,
            "Q" : 1,
            "U" : 2,
            "V" : 3
        }
        
        s_idx = stokes_dict[s]

        if s_idx == 0:
            data = self.cube.list[t].file.data[s_idx, ll_idx]
            data[data < 0] = np.nan
            im1 = self.ax1.imshow(data, cmap="Greys_r")
        elif s_idx == 1 or s_idx == 2:
            im1 = self.ax1.imshow(self.cube.list[t].file.data[s_idx, ll_idx], cmap="Greys_r", vmin=-10, vmax=10)
        else:
            im1 = self.ax1.imshow(self.cube.list[t].file.data[s_idx, ll_idx], cmap="Greys_r", vmin=-50, vmax=50)
        try:
            el = self.cube.file.header["WDESC1"]
        except KeyError:
            el = self.cube.file.header["element"]
        self.ax1.set_title(fr"{el} {self.aa} {self.D} {self.l}$_{1}$ = {ll} {self.aa}")
        self.fig.colorbar(im1, ax=self.ax1, orientation="horizontal", label="Intensity [DNs]")