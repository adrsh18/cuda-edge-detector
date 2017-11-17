
# coding: utf-8

# In[16]:


import matplotlib
import matplotlib.pyplot as plt


# In[17]:


eval_log = open('eval.log', 'r')
eval_lines = eval_log.readlines()
eval_log.close()


# In[10]:


times = [float(line.split()[-1]) for line in eval_lines if 'Time spent' in line]
main_times = [float(line.split()[-1]) for line in eval_lines if 'Time spent in main' in line]
read_times = [float(line.split()[-1]) for line in eval_lines if 'Time spent in reading' in line]
detect_times = [float(line.split()[-1]) for line in eval_lines if 'Time spent in detect_edges' in line]
gauss_times = [float(line.split()[-1]) for line in eval_lines if 'Time spent in gaussian_blur' in line]
sobel_times = [float(line.split()[-1]) for line in eval_lines if 'Time spent in sobel_operator' in line]
nms_times = [float(line.split()[-1]) for line in eval_lines if 'Time spent in non_maxima' in line]
hysteresis_times = [float(line.split()[-1]) for line in eval_lines if 'Time spent in hysteresis' in line]
write_times = [float(line.split()[-1]) for line in eval_lines if 'Time spent in writing' in line]
malloc_times = [float(line.split()[-1]) for line in eval_lines if 'alloc calls' in line]
free_times = [float(line.split()[-1]) for line in eval_lines if 'ree calls' in line]
cpy_times = [float(line.split()[-1]) for line in eval_lines if 'cpy calls' in line]


# In[15]:


serial = [main_times[-1], read_times[-1], detect_times[-1], gauss_times[-1], sobel_times[-1], nms_times[-1], hysteresis_times[-1], write_times[-1]]
parallel = [-main_times[-2], -read_times[-2], -detect_times[-2], -gauss_times[-2], -sobel_times[-2], -nms_times[-2], -hysteresis_times[-2], -write_times[-2]]

y = ['main', 'read input', 'detect_edges', 'gaussian_blur', 'sobel_operator', 'non_maxima_suppression', 'hysteresis', 'write']

plt.figure(figsize=(16,3))
plt.barh(y, serial)
plt.barh(y, parallel)
plt.savefig('../eval_plots/subroutine_details.png')

