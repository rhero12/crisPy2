#!/usr/bin/env python
# coding: utf-8

# In[1]:


from crispy.crisp import CRISP, CRISPWideband, CRISPNonU


# In[2]:


crisp = CRISP("../examples/2014/crisp_l2_20140906_152724_6563_r00447.fits")
print(crisp)


# In[3]:


print(crisp[0])


# In[4]:


crisp[7].intensity_map()


# In[5]:


crisp.from_lonlat(-720, -310)


# In[6]:


crisp[:,759,912].plot_spectrum()


# In[7]:


crispnonu = CRISPNonU("../examples/2017/ca8542/00000.h5")
print(crispnonu)


# In[8]:


crispnonu[:,5].stokes_map(stokes="all")


# In[9]:


crispnonu[:,:,38,257].plot_stokes(stokes="all")


# In[10]:


crispwideband = CRISPWideband("../examples/2017/ca8542/wideband/00000.h5")
print(crispwideband)


# In[11]:


crispwideband.intensity_map()

