
# coding: utf-8

# In[9]:


import matplotlib
import matplotlib.pyplot as plt


# In[10]:


eval_log = open('eval.log', 'r')
eval_lines = eval_log.readlines()
eval_log.close()


# In[11]:


pixel_sizes = [2.7, 5.4, 10.8, 21.6, 43.3, 86.6]


# In[13]:


main = [float(line.split()[-1]) for line in eval_lines if 'Time spent in main' in line]
serial_main = list(reversed(main[1::2]))
parallel_main = list(reversed(main[::2]))

matplotlib.rcParams.update({'font.size': 22})

plt.figure(figsize=(32,18))
plt.plot(pixel_sizes, serial_main, marker='o', label='Serial', color='crimson')
plt.plot(pixel_sizes, parallel_main, marker='o', label='Parallel', color='teal')
plt.xlabel('Pixels (millions)')
plt.ylabel('Runtime (seconds)')

for x, y in zip(pixel_sizes, serial_main):
    plt.annotate('%.2f' % y, xy=(x,y+1), textcoords='data', color='crimson')

for x, y in zip(pixel_sizes, parallel_main):
    plt.annotate('%.2f' % y, xy=(x,y-1), textcoords='data', color='teal')

plt.legend()
plt.grid()
plt.savefig('../eval_plots/serial_vs_parallel.png')


# In[9]:





# In[ ]:




