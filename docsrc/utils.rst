Utilities
=========

Sequence Construction
---------------------

The way the input for the sequence data classes works is not the most opaque thing in the world. So we provide a function to generate a list of dictionaries to consturct the sequence data class.

.. autofunction:: crispy.utils.CRISP_sequence_constructor

Image Utilities
---------------

Oftentimes, the CRISP images need to be augmented in some way for their intended purpose. This could be rotating the images in the image plane to get rid of the background or chopping up the image into segments for work on smaller pieces of the field-of-view. Below is a list of functions to help do these things.

.. autofunction:: crispy.utils.scanline_search_corners

.. autofunction:: crispy.utils.refine_corners

.. autofunction:: crispy.utils.unify_boxes

.. autofunction:: crispy.utils.find_unified_bb

.. autofunction:: crispy.utils.rotate_crop_data

.. autofunction:: crispy.utils.rotate_crop_aligned_data

.. autofunction:: crispy.utils.reconstruct_full_frame

.. autofunction:: crispy.utils.segmentation

.. autofunction:: crispy.utils.segment_cube

.. autofunction:: crispy.utils.mosaic

.. autofunction:: crispy.utils.mosaic_cube