#!/usr/bin/env python
# coding: utf-8

# In[1]:


from crispy.inversions import Inversion
from crispy.Radynversion.utils import z
from crispy.crisp import CRISP

crisp = CRISP("../examples/2014/crisp_l2_20140906_152724_8542_r00447.fits")
inversion = Inversion("../examples/inversions/Inversion_0447.hdf5", z=z, header=crisp.file.header)
print(inversion)


# In[2]:


inversion[15].vel_map()


# In[3]:


inversion.from_lonlat(-755,-330)


# In[4]:


inversion[:,408,298].plot_vel()

