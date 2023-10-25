# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
plt.style.use('dark_background')


# %%
# run the script; variables will be available to the notebook afterwards
from test_cortex import *

# %% [markdown]
# For each frame in the animation, calculate an array of values to plot. 
# 
# That is, for each group of neurons:     
# 1. Spike times are given by Brian in a single 1D array. A second 1D array of the same length gives respective neuron indices. Replace these two 1D arrays with separate 1D spike time arrays, one for each neuron in the group.     
# 2. For each neuron in the group, find the indices where the bin edges fit into its array of spike times using `np.searchsorted`, then use `np.diff` to find the difference between the indices of subsequent bin edges, i.e. the number of spikes in each frame.    
# 3. Insert the per-frame counts into a single array. For example, if we're plotting with `imshow` we want an array of shape `(n_frames, rows, cols)` giving the grid of counts for each frame. 
# 4. Apply any filters to the array from 3. Here, convolution is used along the time dimension to add a fading effect to individual spikes. Without a fading effect, spikes flicker on and off on individual frames. 

# %%
## Parameters
framerate = 30  # frames/second
speed = 0.1  # 0.1 means a 10x slowdown; i.e. 10 s video for 1 s simulation time
filter_ = np.array([1, 0.66, 0.44, 0.29, 0.19, 0.13, 0.08, 0.05])  # for convolution/fade
# filter_ = np.array([1])  # for no fade
# filter_ = np.array([1, 0.5, 0.25])  # shorter fade


# %%
### Bin spike times into N frames.  
# Get spike time arrays for each neuron
# monitor provides pooled time array and neuron index array
spikes = {name: (monitor.t, monitor.i)
          for name, monitor in spike_monitors.items()}
spikes_by_neuron = {name: [times[idxs==idx] / second
                           for idx in range(len(net.pops[name]))]
                    for name, (times, idxs) in spikes.items()}
# Calculate N based on desired frame rate
n_frames = int(framerate * t_sim / speed)
# Divide T into N bins, and count spikes in each bin for each neuron. 
bins = np.linspace(0, float(t_sim), n_frames + 1)
frame_counts = {name: [np.diff(np.searchsorted(times, bins)) 
                       for times in pop_spikes]
                for name, pop_spikes in spikes_by_neuron.items()}
# For each pop, construct N matrices, one for each frame
frames = {name: np.zeros((n_frames, rows, cols))
          for name in frame_counts}
for name, counts_by_neuron in frame_counts.items():
    for idx, counts in enumerate(counts_by_neuron):
        # manual calculation of rows/cols is annoying due to non-standard traversal of grid
        # can't directly use numpy routines like `np.unravel_index`, I think
        # TODO: down the columns instead of up, `order='F'` would work
        row = rows - (idx % rows) - 1
        col = idx // rows 
        
        # turns all non-zero counts to 1
        # should remove the `.astype(bool)` if you want to keep the #spikes/frame for the neuron
        frames[name][:, row, col] = counts.astype(bool)
        

# convolve the spikes to add a fading effect
# this adds a few frames to the end; chop them off to keep 1-to-1 with `bins`. for now.
frames = {name: np.apply_along_axis(lambda n: np.convolve(n, filter_, mode='full'),
                                    axis=0, arr=frames[name])[:len(bins)]
          for name in frames}

# %% [markdown]
# Class for animating spikes on a grid using `plt.imshow`. 

# %%
class AnimateGrids:
    def __init__(self, frame_data, framerate=30, times=None, cmap=plt.cm.gray):
        self.frame_data = frame_data
        self.framerate = framerate
        self.times = times if times is not None else None
        self.cmap = cmap
        self.n_frames = [len(frame_data[name]) for name in frame_data][0]

        # TODO: generalize to arbitrary # of subplots
        self.fig, axs = plt.subplots(2,2, figsize=(8, 8))
        self.axs = {name: ax for name, ax in zip(frame_data, axs.flat)}
        self.anim = animation.FuncAnimation(self.fig, self.update, init_func=self.init, 
                                            frames=self.n_frames, interval=1./self.framerate,
                                            blit=False, save_count=self.n_frames, repeat=False)
    
    def init(self):
        for name, ax in self.axs.items():
            ax.set_title(name)
            ax.set_xticklabels([0, self.frame_data[name].shape[1] - 1])
            ax.set_yticklabels([0, self.frame_data[name].shape[2] - 1])
            ax.set_xticks([0, self.frame_data[name].shape[1] - 1])
            ax.set_yticks([0, self.frame_data[name].shape[2] - 1])

            self.ims = {name: ax.imshow(np.zeros_like(self.frame_data[name][0]), 
                                        norm=plt.Normalize(vmin=0, vmax=1),
                                        cmap=self.cmap, aspect='equal',
                                        interpolation='none') 
                        for name, ax in self.axs.items()}
            try: 
                self.time_txt = self.fig.suptitle("t = {:0.3f} s".format(self.times[0]))
            except TypeError:
                self.time_txt = None
        return self.artists
    
    def update(self, idx):
        for name, im in self.ims.items():
            try: 
                self.time_txt.set_text("t = {:0.3f} s".format(self.times[idx]))
            except TypeError:
                pass
            im.set_data(self.frame_data[name][idx])
        return self.artists      

    @property
    def artists(self):
        return tuple(self.ims.values()) + (self.time_txt,)


# %%
grid_anim = AnimateGrids(frames, framerate, times=bins)
# the next two lines are to keep matplotlib from overlapping plot elements
plt.tight_layout(h_pad=2)
plt.subplots_adjust(top=0.95)
grid_anim.anim.save('spikes_test.mp4', fps=framerate)


# %%



