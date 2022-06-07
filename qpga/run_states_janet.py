# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:58:13 2022

@author: Janet
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

import qutip
import h5py

from qpga import *
from qpga.model import QPGA
from qpga.circuits import QFT, QFT_layer_count, cluster_state_generator
from qpga.training import *
from qpga.fidelity_search import *
from qpga.linalg import *
from qpga.plotting import *
from qpga.state_preparation import *
from qpga.callbacks import *
from qpga.utils import *

from tqdm import tqdm_notebook as tqdm

import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
K.set_floatx('float64')

%matplotlib inline
mpl.rcParams['figure.dpi'] = 300

%config InlineBackend.figure_format = 'retina'
np.set_printoptions(precision=3, linewidth=300)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

N = 4
num_samples = 1

in_data = np_to_k_complex(np.array([zero_state(N)] * num_samples))
out_data = np.array([[[0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1/np.sqrt(2), 0., -1/np.sqrt(2)],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1/np.sqrt(2), 0., -1/np.sqrt(2)]]])

model = QPGA(N, 20).as_sequential()
model.compile(optimizer=Adam(lr=0.002), 
              loss=antifidelity, 
              metrics=[antifidelity])

callback = StatePreparationHistoryCallback(num_qubits=N, input_state = in_data[0:1], target_state = out_data[0:1])
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', 
                                                       factor = 0.25,
                                                       patience = 2,
                                                       verbose=1,
                                                       min_lr=1e-6)

history = model.fit(in_data, out_data, epochs=201, 
                    callbacks=[callback, reduce_lr_callback], verbose = 2)