import numpy as np
from scipy.ndimage import rotate
from cycler import cycler
from tqdm import tqdm
from numba import njit

class ObjDict(dict):
    '''
    This is an abstract class for allowing the keys of a dictionary to be accessed like class attributes.
    '''

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: "+name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: "+name)

## The following is the Paul Tol colourblind colour schemes for use in plots from https://personal.sron.nl/~pault/ ##

pt_bright = {
    "blue" : "#4477AA",
    "cyan" : "#66CCEE",
    "green" : "#228833",
    "yellow" : "#CCBB44",
    "red" : "#EE6677",
    "purple" : "#AA3377",
    "grey" : "#BBBBBB"
}

pt_hic = {
    "white" : "#FFFFFF",
    "yellow" : "#DDAA33",
    "red" : "#BB5566",
    "blue" : "#004488",
    "black" : "#000000"
}

pt_vibrant = {
    "blue" : "#0077BB",
    "cyan" : "#33BBEE",
    "teal" : "#009988",
    "orange" : "#EE7733",
    "red" : "#CC3311",
    "magenta" : "#EE3377",
    "grey" : "#BBBBBB"
}

pt_muted = {
    "indigo" : "#332288",
    "cyan" : "#88CCEE",
    "teal" : "#44AA99",
    "green" : "#117733",
    "olive" : "#999933",
    "sand" : "#DDCC77",
    "rose" : "#CC6677",
    "wine" : "#882255",
    "purple" : "#AA4499",
    "pale grey" : "#DDDDDD"
}

pt_bright_cycler = cycler(color=list(pt_bright.values()))

pt_hic_cycler = cycler(color=list(pt_hic.values()))

pt_vibrant_cycler = cycler(color=list(pt_vibrant.values()))

pt_muted_cycler = cycler(color=list(pt_muted.values()))

def CRISP_sequence_constructor(files, wcs=None, uncertainty=None, mask=None, nonu=False):
    """
    This is a helper function for constructing the kind of dictionary needed to make a CRISPSequence or CRISPWidebandSequence or CRISPNonUSequence object.

    Parameters
    ----------
    files : list
        A list of paths to the files.
    wcs : list [astropy.wcs.WCS] or None, optional
        A list of WCS of each of the files. Default is None, the data classes will work out the WCS for the observations.
    uncertainty : list [float] or None, optional
        A list of the uncertainty associated with each observation. Default is None.
    mask : list [numpy.ndarray] or None, optional
        A list of the mask to use for each observation. Default is None.
    nonu : bool, optional
        Whether or not the spectral axis is sampled non-uniformly. Default is False.
    """
    file_dict_list = []

    for j, f in enumerate(files):
        d = {}
        d["filename"] = f
        if wcs is not None:
            d["wcs"] = wcs[j]
        if uncertainty is not None:
            d["uncertainty"] = uncertainty[j]
        if mask is not None:
            d["mask"] = mask[j]
        d["nonu"] = nonu

        file_dict_list.append(d)

    return file_dict_list


@njit
def scanline_search_corners(im, size=10):
    '''
    Finds the corners of the embedded image by scanning through lines
    (rows/columns) and looking for a run of pixels different to each other
    and the background.

    Parameters
    ----------
    im : numpy.ndarray
        2D array in which to find the sub-image corners. Best results usually
        obtained from line-core observations.
    size : int, optional
        The number of consecutive pixels in a "run" for the sub-image to be
        detected (default: 10).

    Returns
    -------
    corners : numpy.ndarray
        2D array containing the (x,y) coordinates of the corners in order
        [top, left, right, bottom].
    '''
    bgRef = im[0,0]
    corners = []
    def top():
        for y in range(im.shape[0]):
            for x in range(im.shape[1] - size):
                chunk = im[y, x:x+size]
                if np.all(chunk != bgRef) and not np.all(chunk == chunk[0]):
                    return (x+size//2, y)
        return (0,0)

    def bottom():
        for y in range(im.shape[0]-1, -1, -1):
            for x in range(im.shape[1] - size):
                chunk = im[y, x:x+size]
                if np.all(chunk != bgRef) and not np.all(chunk == chunk[0]):
                    return (x+size//2, y)
        return (0,0)

    def left():
        for x in range(im.shape[1]):
            for y in range(im.shape[0] - size):
                chunk = im[y:y+size, x]
                if np.all(chunk != bgRef) and not np.all(chunk == chunk[0]):
                    return (x, y+size//2)
        return (0,0)

    def right():
        for x in range(im.shape[1]-1, -1, -1):
            for y in range(im.shape[0] - size):
                chunk = im[y:y+size, x]
                if np.all(chunk != bgRef) and not np.all(chunk == chunk[0]):
                    return (x, y+size//2)
        return (0,0)


    corners = np.array((top(), left(), right(), bottom()))
    return corners


def towards_centroid(vec, centroid, dist=2):
    '''
    Moves a point towards another point by a certain distance

    Parameters
    ----------
    vec : numpy.ndarray
        1D, 2 element vector representing the floating point coordinate to
        move. Will be modified in place.
    centroid : numpy.ndarray
        1D, 2 element vector representing the floating point coordinate to
        move towards.
    dist : float, optional
        The distance to move vec along the (vec, centroid) line (default: 2).
    '''
    dvec = centroid - vec
    dvec /= np.linalg.norm(dvec)
    vec += dist * dvec


def refine_corners(corners):
    '''
    Refine the corners found by the scanline algorithm so that the edges of
    the quadrilateral defined are perpendicular and guaranteed to remain
    within the image region.

    Parameters
    ----------
    corners : List[Tuple[int]]
        The corners found by the scanline algorithm.

    Returns
    -------
    refinedCorners: List[numpy.ndarray]
        The locations of the refined floating point corners (order: top,
        left, right, bottom)
    '''
    corners = [c.astype('<f8') for c in corners]
    top, left, right, bottom = corners
    v = right - bottom
    v /= np.linalg.norm(v)

    slopebr = v[1] / v[0]
    interceptbr = bottom[1] - slopebr * bottom[0]

    perpSlopelb = -v[0] / v[1]
    perpIntercept = bottom[1] - perpSlopelb * bottom[0]

    # NOTE(cmo): Adjust if the lb line leaves the region
    if perpSlopelb * left[0] + perpIntercept > left[1]:
        perpIntercept = left[1] - perpSlopelb * left[0]
        bottomX = (interceptbr - perpIntercept) / (perpSlopelb - slopebr)
        bottomY = slopebr * bottomX + interceptbr
        bottom = np.array([bottomX, bottomY])
    else:
        leftX = (left[1] - perpIntercept) / perpSlopelb
        left = np.array([leftX, left[1]])

    # NOTE(cmo): Adjust top point
    interceptlt = left[1] - slopebr * left[0]
    intercepttr = right[1] - perpSlopelb * right[0]
    topX = (intercepttr - interceptlt) / (slopebr - perpSlopelb)
    topY = slopebr * topX + interceptlt
    top = np.array([topX, topY])

    centroid = 0.25 * (top + bottom + left + right)

    # NOTE(cmo): Shift everything in a couple of units
    towards_centroid(top, centroid)
    towards_centroid(bottom, centroid)
    towards_centroid(left, centroid)
    towards_centroid(right, centroid)

    return [top, left, right, bottom]


def line_params(p1, p2):
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    return m, p1[1] - m * p1[0]


def unify_boxes(*cornerLists):
    '''
    Unify a set of boxes, defined by their corners so that the box with the
    smallest area is modified (if necessary) to be fully contained within the
    others.

    Parameters
    ----------
    *cornerLists : List[numpy.ndarray]
        Lists of corners returned from refine_corners to unify.

    Returns
    -------
    unifiedCorners : List[numpy.ndarray]
        List of unified corners found.

    Raises
    ------
    ValueError
        Raised if no valid set of corners could be found.
    '''
    def area(c):
        return np.linalg.norm(c[0] - c[1]) * np.linalg.norm(c[0] - c[2])

    areas = np.array([area(c) for c in cornerLists])
    minAreaIdx = areas.argmin()
    cornersToAdjust = [np.copy(c) for c in cornerLists[minAreaIdx]]
    centroid = 0.25 * (cornersToAdjust[0] + cornersToAdjust[1] + cornersToAdjust[2] + cornersToAdjust[3])
    otherCornerLists = [c for i, c in enumerate(cornerLists) if i != minAreaIdx]
    # [lb, br, lt, tr]
    params = [[line_params(c[1], c[3]), line_params(c[3], c[2]),
               line_params(c[1], c[0]), line_params(c[0], c[2])]
              for c in otherCornerLists]

    for p in params:
        for cIdx in range(len(cornersToAdjust)):
            for i in range(100):
                # TODO(cmo): This approach may fail for > 2 boxes, but that's
                # not a problem we have right now. May just need to swap loop
                # ordering.
                c = cornersToAdjust[cIdx]
                lbIn = c[1] <= p[0][0] * c[0] + p[0][1]
                brIn = c[1] <= p[1][0] * c[0] + p[1][1]
                ltIn = c[1] >= p[2][0] * c[0] + p[2][1]
                trIn = c[1] >= p[3][0] * c[0] + p[3][1]

                if not all((lbIn, brIn, ltIn, trIn)):
                    for cc in range(len(cornersToAdjust)):
                        towards_centroid(cornersToAdjust[cc], centroid)
                else:
                    break
            else:
                raise ValueError('No unified corners found')
    return cornersToAdjust


def find_unified_bb(imA, imB):
    '''
    Scans the provided images to find the corners of the rotated data and
    returns a unified bounding box that fits within the data on both frames.

    Parameters
    ----------
    imA : numpy.ndarray
        2D array of image data (float) containing the first image.

    imB : numpy.ndarray
        2D array of image data (float) containing the second image.

    Returns
    -------
    finalCorners : List[numpy.ndarray]
        Corners which are within the data region of both images.

    Raises
    ------
    ValueError
        This is propagated from unify_boxes if there is no valid corner
        solution for which the smaller box maintains its centroid and all
        corners are within the larger box.
    '''
    cornersA = scanline_search_corners(imA)
    refinedA = refine_corners(cornersA)
    cornersB = scanline_search_corners(imB)
    refinedB = refine_corners(cornersB)
    finalCorners = unify_boxes(refinedA, refinedB)
    return finalCorners


def rotate_crop_aligned_data(imA, imB):
    '''
    Rotate and crop a unified section of data from images in two different
    wavebands.

    Parameters
    ----------
    imA : numpy.ndarray
        3 or 4D array containing the first image cube.
    imB : numpy.ndarray
        3 or 4D array containing the first second cube.

    Returns
    -------
    aCrop : numpy.ndarray
        3 or 4D array containing the rotated and cropped data from the first
        image.
    bCrop : numpy.ndarray
        3 or 4D array containing the rotated and cropped data from the second
        image.
    cropData : dict
        Dictionary containing the metadata necessary to reconstruct these
        cropped images into their full-frame input using
        reconstruct_full_frame (excluding the border lost to the crop).

    Raises
    ------
    ValueError
        - Can be propagated from unify_boxes if no valid unified corner solution is found.
        - If the arrays are of incompatible dimension.
    '''
    imA = imA.astype('<f4')
    imB = imB.astype('<f4')
    stokes = False
    if len(imA.shape) == 4 and len(imB.shape) == 4:
        stokes = True
        aMidWvl = imA.shape[1] // 2
        bMidWvl = imB.shape[1] // 2
        imACore = np.copy(imA[aMidWvl])
        imBCore = np.copy(imB[bMidWvl])
        imA = imA.reshape(-1, *imA.shape[-2:])
        imB = imB.reshape(-1, *imB.shape[-2:])
    elif len(imA.shape) == 3 and len(imB.shape) == 3:
        aMidWvl = imA.shape[0] // 2
        bMidWvl = imB.shape[0] // 2
        imACore = np.copy(imA[aMidWvl])
        imBCore = np.copy(imB[bMidWvl])
    else:
        raise ValueError('Unexpected dimensionality of imA and imB, expected both 3 or 4, got %d and %d'
                           % (len(imA.shape), len(imB.shape)))

    if imA.shape[-2:] != imB.shape[-2:]:
        raise ValueError('x and y dimensions of image cubes seem incompatible.')

    corners = find_unified_bb(imACore, imBCore)
    top, left, right, bottom = corners
    br = right - bottom
    brUnit = br / np.linalg.norm(br)
    angle = np.arccos(brUnit @ np.array([1, 0]))
    imCentre = np.array([imACore.shape[1] // 2, imACore.shape[0] // 2])
    rotMat = np.array(((np.cos(-angle),  np.sin(-angle), imCentre[0]),
                       (-np.sin(-angle), np.cos(-angle), imCentre[1]),
                       (0,               0,              1)))
    bTrans = np.ones(3)
    bTrans[:2] = bottom - imCentre
    lTrans = np.ones(3)
    lTrans[:2] = left - imCentre
    rTrans = np.ones(3)
    rTrans[:2] = right - imCentre

    bRot = np.rint(rotMat @ bTrans)
    lRot = np.rint(rotMat @ lTrans)
    rRot = np.rint(rotMat @ rTrans)

    imARot = np.empty_like(imA)
    for wave in range(imA.shape[0]):
        imARot[wave] = rotate(imA[wave], -np.rad2deg(angle), cval=imA[wave,0,0], reshape=False)

    imBRot = np.empty_like(imB)
    for wave in range(imB.shape[0]):
        imBRot[wave] = rotate(imB[wave], -np.rad2deg(angle), cval=imB[wave,0,0], reshape=False)

    cropData = {'frameDims': (imA.shape[-2], imA.shape[-1]),
                'xMin': int(bRot[0]), 'xMax': int(rRot[0]), 'yMin': int(lRot[1]), 'yMax': int(bRot[1]),
                'angle': angle}

    aCrop = imARot[:, cropData['yMin']:cropData['yMax'],
                      cropData['xMin']:cropData['xMax']]
    bCrop = imBRot[:, cropData['yMin']:cropData['yMax'],
                      cropData['xMin']:cropData['xMax']]
    if stokes:
        aCrop = aCrop.reshape(4, -1, aCrop.shape[-2:])
        bCrop = bCrop.reshape(4, -1, bCrop.shape[-2:])
    return aCrop, bCrop, cropData


def rotate_crop_data(im):
    '''
    Rotate and crop the data from a rotated image.

    Parameters
    ----------
    im : numpy.ndarray
        3 or 4D array containing the image cube.

    Returns
    -------
    crop : numpy.ndarray
        3 or 4D array containing the rotated and cropped data from the image.
    cropData : dict
        Dictionary containing the metadata necessary to reconstruct these
        cropped images into their full-frame input using
        reconstruct_full_frame (excluding the border lost to the crop).

    Raises
    ------
    ValueError
        If array is of incorrect dimensionality.
    '''
    im = im.astype('<f4')
    stokes = False
    if len(im.shape) == 4:
        stokes = True
        midWvl = im.shape[1] // 2
        imCore = np.copy(im[midWvl])
        im = im.reshape(-1, *im.shape[-2:])
    elif len(im.shape) == 3:
        midWvl = im.shape[0] // 2
        imCore = np.copy(im[midWvl])
    else:
        raise ValueError('Unexpected dimensionality of im, expected 3 or 4, got %d'
                           % (len(im.shape)))

    scanCorners = scanline_search_corners(imCore)
    corners = refine_corners(scanCorners)
    top, left, right, bottom = corners
    br = right - bottom
    brUnit = br / np.linalg.norm(br)
    angle = np.arccos(brUnit @ np.array([1, 0]))
    imCentre = np.array([imCore.shape[1] // 2, imCore.shape[0] // 2])
    rotMat = np.array(((np.cos(-angle),  np.sin(-angle), imCentre[0]),
                       (-np.sin(-angle), np.cos(-angle), imCentre[1]),
                       (0,               0,              1)))
    bTrans = np.ones(3)
    bTrans[:2] = bottom - imCentre
    lTrans = np.ones(3)
    lTrans[:2] = left - imCentre
    rTrans = np.ones(3)
    rTrans[:2] = right - imCentre

    bRot = np.rint(rotMat @ bTrans)
    lRot = np.rint(rotMat @ lTrans)
    rRot = np.rint(rotMat @ rTrans)

    imRot = np.empty_like(im)
    for wave in range(im.shape[0]):
        imRot[wave] = rotate(im[wave], -np.rad2deg(angle), cval=im[wave,0,0], reshape=False)

    cropData = {'frameDims': (im.shape[-2], im.shape[-1]),
                'xMin': int(bRot[0]), 'xMax': int(rRot[0]), 'yMin': int(lRot[1]), 'yMax': int(bRot[1]),
                'angle': angle}

    crop = imRot[:, cropData['yMin']:cropData['yMax'],
                    cropData['xMin']:cropData['xMax']]
    if stokes:
        crop = crop.reshape(4, -1, crop.shape[-2:])
    return crop, cropData


def reconstruct_full_frame(cropData, im):
    '''
    Reconstruct the full-frame derotated data for a rotated and cropped image
    cube using the metadata.

    Parameters
    ----------
    cropData : dict
        The crop metadata returned from rotate_crop_data.
    im : numpy.ndarray
        The datacube to be restored to full frame

    Returns
    -------
    rotatedIm : numpy.ndarray
        A derotated, full-frame, copy of the input image cube.
    '''
    stokes = False
    if len(im.shape) == 4:
        stokes = True
        im = im.reshape(-1, *im.shape[-2:])
    wvls = im.shape[0]

    imFullFrame = np.zeros((wvls, *cropData['frameDims']), np.float32)
    imFullFrame[:, cropData['yMin']:cropData['yMax'],
                   cropData['xMin']:cropData['xMax']] = im

    imFFRot = np.empty_like(imFullFrame)
    for wave in range(wvls):
        imFFRot[wave] = rotate(imFullFrame[wave], np.rad2deg(cropData['angle']), reshape=False)

    if stokes:
        imFFRot = imFFRot.reshape(4, -1, *imFFRot.shape[-2:])

    return imFFRot


def segmentation(img, n):
    '''
    This is a preprocessing function that will segment the images into segments with dimensions n x n.
    Parameters
    ----------
    img : numpy.ndarray
        The image to be segmented.
    n : int
        The dimension of the segments.
    Returns
    -------
    segments : numpy.ndarray
        A numpy array of the segments of the image.
    '''

    N = img.shape[0] // n #the number of whole segments in the y-axis
    M = img.shape[1] // n #the number of whole segments in the x-axis

    ####
    # there are 4 cases
    #+------------+------------+------------+
    #| *n         | y segments | x segments |
    #+------------+------------+------------+
    #| N !=, M != | N+1        | M+1        |
    #+------------+------------+------------+
    #| N !=, M =  | N+1        | M          |
    #+------------+------------+------------+
    #| N =, M !=  | N          | M+1        |
    #+------------+------------+------------+
    #| N =, M =   | N          | M          |
    #+------------+------------+------------+
    ####
    if N*n != img.shape[0] and M*n != img.shape[1]:
        segments = np.zeros((N+1, M+1, n, n), dtype=np.float32)
    elif N*n != img.shape[0] and M*n == img.shape[1]:
        segments = np.zeros((N+1, M, n, n), dtype=np.float32)
    elif N*n == img.shape[0] and M*n != img.shape[1]:
        segments = np.zeros((N, M+1, n, n), dtype=np.float32)
    else:
        segments = np.zeros((N, M, n, n), dtype=np.float32)

    y_range = range(segments.shape[0])
    x_range = range(segments.shape[1])

    for j in y_range:
        for i in x_range:
            if i != x_range[-1] and j != y_range[-1]:
                segments[j, i] = img[j*n:(j+1)*n,i*n:(i+1)*n]
            elif i == x_range[-1] and j != y_range[-1]:
                segments[j, i] = img[j*n:(j+1)*n,-n:]
            elif i != x_range[-1] and j == y_range[-1]:
                segments[j, i] = img[-n:,i*n:(i+1)*n]
            elif i == x_range[-1] and j == y_range[-1]:
                segments[j, i] = img[-n:,-n:]

    segments = np.reshape(segments, newshape=((segments.shape[0]*segments.shape[1]), n, n))

    return segments

def segment_cube(img_cube, n):
    '''
    A function to segment a three-dimensional datacube.
    Parameters
    ----------
    img_cube : numpy.ndarray
        The image cube to be segmented.
    n : int
        The dimension of the segments.
    Returns
    -------
    segments : numpy.ndarray
        A numpy array of the segments of the image cube.
    '''

    for j, img in enumerate(tqdm(img_cube, desc="Segmenting image cube: ")):
        if j == 0:
            segments = segmentation(img, n=n)
            #we expand the segments arrays to be four-dimensional where one dimension will be the image positiion within the cube so it will be (lambda point, segments axis, y, x)
            segments = np.expand_dims(segments, axis=0)
        else:
            tmp_s = segmentation(img, n=n)
            tmp_s = np.expand_dims(tmp_s, axis=0)
            #we then add each subsequent segmented image along the wavelength axis
            segments = np.append(segments, tmp_s, axis=0)
    segments = np.swapaxes(segments, 0, 1) #this puts the segment dimension first, wavelength second to make it easier for data loaders

    return segments

def mosaic(segments, img_shape, n):
    '''
    A processing function to mosaic the segments back together.
    Parameters
    ----------
    segments : numpy.ndarray
        A numpy array of the segments of the image.
    img_shape : tuple
        The shape of the original image.
    n : int
        The dimension of the segments.
    Returns
    -------
    mosaic_img : numpy.ndarray
        The reconstructed image.
    '''

    N = img_shape[0] // n
    M = img_shape[1] // n
    if N*n != img_shape[0] and M*n != img_shape[1]:
        segments = np.reshape(segments, newshape=(N+1, M+1, *segments.shape[-2:]))
    elif N*n != img_shape[0] and M*n == img_shape[1]:
        segments = np.reshape(segments, newshape=(N+1, M, *segments.shape[-2:]))
    elif N*n == img_shape[0] and M*n != img_shape[1]:
        segments = np.reshape(segments, newshape=(N, M+1, *segments.shape[-2:]))
    else:
        segments = np.reshape(segments, newshape=(N, M, segments.shape[-2:]))

    mosaic_img = np.zeros(img_shape, dtype=np.float32)
    y_range = range(segments.shape[0])
    x_range = range(segments.shape[1])
    y_overlap = img_shape[0] - N*n
    x_overlap = img_shape[1] - M*n


    for j in y_range:
        for i in x_range:
            if i != x_range[-1] and j != y_range[-1]:
                mosaic_img[j*n:(j+1)*n,i*n:(i+1)*n] = segments[j,i]
            elif i == x_range[-1] and j != y_range[-1]:
                mosaic_img[j*n:(j+1)*n,-x_overlap:] = segments[j,i,:,-x_overlap:]
            elif i != x_range[-1] and j == y_range[-1]:
                mosaic_img[-y_overlap:,i*n:(i+1)*n] = segments[j,i,-y_overlap:]
            elif i == x_range[-1] and j == y_range[-1]:
                mosaic_img[-y_overlap:,-x_overlap:] = segments[j,i,-y_overlap:,-x_overlap:]
            else:
                raise IndexError("These indices are out of the bounds of the image. Check your ranges!")

    for j in y_range:
        for i in x_range:
            if ((j-1) >= 0) and ((i-1) >= 0) and ((j+1) <= y_range[-1]) and ((i+1) <= x_range[-1]):
                top = mosaic_img[((j+1)*n)-3:((j+1)*n)+3, i*n:(i+1)*n]
                bottom = mosaic_img[(j*n)-3:(j*n)+3, i*n:(i+1)*n]
                left = mosaic_img[j*n:(j+1)*n, (i*n)-3:(i*n)+3]
                right = mosaic_img[j*n:(j+1)*n, ((i+1)*n)-3:((i+1)*n)+3]

                top_new = np.zeros_like(top)
                bottom_new = np.zeros_like(bottom)
                left_new = np.zeros_like(left)
                right_new = np.zeros_like(right)
                for k in range(top.shape[-1]):
                    top_new[:,k] = np.interp(np.arange(top.shape[0]), np.array([0,top.shape[0]-1]), np.array([top[0,k],top[-1,k]]))
                    bottom_new[:,k] = np.interp(np.arange(bottom.shape[0]), np.array([0, bottom.shape[0]-1]), np.array([bottom[0,k],bottom[-1,k]]))
                    left_new[k,:] = np.interp(np.arange(left.shape[1]), np.array([0, left.shape[1]-1]), np.array([left[k,0],left[k,-1]]))
                    right_new[k,:] = np.interp(np.arange(right.shape[1]), np.array([0, right.shape[1]-1]), np.array([right[k,0],right[k,-1]]))

                mosaic_img[((j+1)*n)-3:((j+1)*n)+3, i*n:(i+1)*n] = top_new
                mosaic_img[(j*n)-3:(j*n)+3, i*n:(i+1)*n] = bottom_new
                mosaic_img[j*n:(j+1)*n, (i*n)-3:(i*n)+3] = left_new
                mosaic_img[j*n:(j+1)*n, ((i+1)*n)-3:((i+1)*n)+3] = right_new
            elif (j == 0) and ((i-1) >= 0) and ((i+1) <= x_range[-1]):
                left = mosaic_img[j*n:(j+1)*n, (i*n)-3:(i*n)+3]
                right = mosaic_img[j*n:(j+1)*n, ((i+1)*n)-3:((i+1)*n)+3]

                left_new = np.zeros_like(left)
                right_new = np.zeros_like(right)
                for k in range(left.shape[-2]):
                    left_new[k,:] = np.interp(np.arange(left.shape[1]), np.array([0, left.shape[1]-1]), np.array([left[k,0], left[k,-1]]))
                    right_new[k,:] = np.interp(np.arange(right.shape[1]), np.array([0, right.shape[1]-1]), np.array([right[k,0], right[k,-1]]))

                mosaic_img[j*n:(j+1)*n, (i*n)-3:(i*n)+3] = left_new
                mosaic_img[j*n:(j+1)*n, ((i+1)*n)-3:((i+1)*n)+3] = right_new

            elif (i == 0) and ((j-1) >= 0) and ((j+1) <= y_range[-1]):
                top = mosaic_img[((j+1)*n)-3:((j+1)*n)+3, i*n:(i+1)*n]
                bottom = mosaic_img[(j*n)-3:(j*n)+3, i*n:(i+1)*n]

                top_new = np.zeros_like(top)
                bottom_new = np.zeros_like(bottom)
                for k in range(top.shape[-1]):
                    top_new[:,k] = np.interp(np.arange(top.shape[0]), np.array([0, top.shape[0]-1]), np.array([top[0, k], top[-1, k]]))
                    bottom_new[:,k] = np.interp(np.arange(bottom.shape[0]), np.array([0, bottom.shape[0]-1]), np.array([bottom[0, k], bottom[-1, k]]))

                mosaic_img[((j+1)*n)-3:((j+1)*n)+3, i*n:(i+1)*n] = top_new
                mosaic_img[(j*n)-3:(j*n)+3, i*n:(i+1)*n] = bottom_new

            elif (i == x_range[-1]) and ((j-1) >= 0) and ((j+1) <= y_range[-1]):
                top = mosaic_img[((j+1)*n)-3:((j+1)*n)+3, -x_overlap:]
                bottom = mosaic_img[(j*n)-3:(j*n)+3, -x_overlap:]

                top_new = np.zeros_like(top)
                bottom_new = np.zeros_like(bottom)
                for k in range(top.shape[-1]):
                    top_new[:,k] = np.interp(np.arange(top.shape[0]), np.array([0, top.shape[0]-1]), np.array([top[0, k], top[-1, k]]))
                    bottom_new[:,k] = np.interp(np.arange(bottom.shape[0]), np.array([0, bottom.shape[0]-1]), np.array([bottom[0, k], bottom[-1, k]]))

                mosaic_img[((j+1)*n)-3:((j+1)*n)+3, -x_overlap:] = top_new
                mosaic_img[(j*n)-3:(j*n)+3, -x_overlap:] = bottom_new

            elif (j == y_range[-1]) and ((i-1) >= 0) and ((i+1) <= x_range[-1]):
                left = mosaic_img[-y_overlap:, (i*n)-3:(i*n)+3]
                right = mosaic_img[-y_overlap:, ((i+1)*n)-3:((i+1)*n)+3]

                left_new = np.zeros_like(left)
                right_new = np.zeros_like(right)
                for k in range(left.shape[-2]):
                    left_new[k,:] = np.interp(np.arange(left.shape[1]), np.array([0, left.shape[1]-1]), np.array([left[k,0], left[k,-1]]))
                    right_new[k,:] = np.interp(np.arange(right.shape[1]), np.array([0, right.shape[1]-1]), np.array([right[k,0], right[k,-1]]))

                mosaic_img[-y_overlap:, (i*n)-3:(i*n)+3] = left_new
                mosaic_img[-y_overlap:, ((i+1)*n)-3:((i+1)*n)+3] = right_new

    return mosaic_img

def mosaic_cube(segments, img_shape, n):
    '''
    A function to mosaic a segment list into an image cube.
    Parameters
    ----------
    segments : numpy.ndarray
        The segments to be mosaiced back into images.
    img_shape : tuple
        The dimensions of the images.
    n : int
        The dimensions of the segments. Default is 64 e.g. 64 x 64.
    Returns
    -------
    m_cube : numpy.ndarray
        The cube of mosaiced images.
    '''

    m_cube = np.zeros((segments.shape[1], *img_shape), dtype=np.float32)
    segments = np.swapaxes(segments, 0, 1) #swap the number of segments and wavelength channels back to make it easier to mosaic along the wavelength axis

    for j, img in enumerate(segments):
        m_cube[j] = mosaic(img, img_shape, n=n)

    return m_cube