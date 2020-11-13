from astropy.nddata.mixins.ndslicing import NDSlicingMixin
from .utils import ObjDict

class CRISPSlicingMixin(NDSlicingMixin):
    '''
    This is the parent class that will allow the CRISP objects to be sliced without having to create new objects.
    '''

    def __getitem__(self, item):
        kwargs = self._slice(item)
        cl = self.__class__(**kwargs)
        cl.ind = item
        return cl

    def _slice(self, item):
        kwargs = {}
        kwargs["filename"] = ObjDict({})
        kwargs["filename"]["data"] = self.data[item]
        kwargs["uncertainty"] = self._slice_uncertainty(item)
        kwargs["mask"] = self._slice_mask(item)
        kwargs["wcs"] = self._slice_wcs(item)
        kwargs["filename"]["header"] = self.header
        kwargs["nonu"] = self.nonu

        return kwargs

class CRISPSequenceSlicingMixin(CRISPSlicingMixin):
    '''
    This is the parent class that will allow the CRISPSequence objects to be sliced without having to create new objects.
    '''

    def __getitem__(self, item):
        args = self._slice(item)
        return self.__class__(args)

    def _slice(self, item):
        args = [f._slice(item) for f in self.list]
        return args

class InversionSlicingMixin(NDSlicingMixin):
    """
    This is the parent class that will allow the Inversion objects to be sliced without having to create new objects.
    """

    def __getitem__(self, item):
        kwargs = self._slice(item)
        cl = self.__class__(**kwargs)
        cl.ind = item
        return cl

    def _slice(self, item):
        kwargs = {}
        # if type(item) == tuple:
        #     err_item = list(item)
        #     err_item.insert(1, slice(None))
        #     err_item = tuple(err_item)
        # else:
        #     err_item = item
        kwargs["filename"] = ObjDict({})
        kwargs["filename"]["ne"] = self.ne[item]
        kwargs["filename"]["temperature"] = self.temp[item]
        kwargs["filename"]["vel"] = self.vel[item]
        kwargs["filename"]["ne_err"] = self.ne_err[item]
        kwargs["filename"]["temperature_err"] = self.temp_err[item]
        kwargs["filename"]["vel_err"] = self.vel_err[item]
        kwargs["wcs"] = self._slice_wcs(item)
        kwargs["z"] = self.z
        kwargs["header"] = self.header

        return kwargs
