#!/usr/bin/env python
# coding: utf-8

# In[1]:


from crispy.crisp import CRISPSequence, CRISPNonUSequence, CRISPWidebandSequence
from crispy.utils import CRISP_sequence_constructor


# In[2]:


crisps = CRISPSequence(CRISP_sequence_constructor(["../examples/2014/crisp_l2_20140906_152724_8542_r00447.fits","../examples/2014/crisp_l2_20140906_152724_6563_r00447.fits"]))
print(crisps)


# In[3]:


print(crisps.list[0])


# In[4]:


crispsnonu = CRISPNonUSequence(CRISP_sequence_constructor(["../examples/2017/ca8542/00000.h5", "../examples/2017/Halpha/00000.h5"], nonu=True))
print(crispsnonu)


# In[5]:


crispswideband = CRISPWidebandSequence(CRISP_sequence_constructor(["../examples/2017/ca8542/wideband/00000.h5", "../examples/2017/ca8542/wideband/00002.h5"]))
print(crispswideband)

