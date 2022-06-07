# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:44:30 2022

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
from tensorflow.python.keras import backend as K
K.set_floatx('float64')

%matplotlib inline
mpl.rcParams['figure.dpi'] = 300

%config InlineBackend.figure_format = 'retina'
np.set_printoptions(precision=3, linewidth=300)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

N = 1

in_data, out_data = prepare_training_data(QFT, N, 1000)

loss_fn = keras.losses.SparseCategoricalCrossentropy()

model = QPGA(N, 200).as_sequential()
model.build(in_data.shape)
model.compile(optimizer = Adam(lr = 0.0001),
              loss=loss_fn,
              metrics = [antifidelity])

operator_vis = OperatorHistoryCallback(num_qubits=N, in_data=in_data, out_data=out_data)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                              factor = 0.5,
                                              cooldown = 4,
                                              patience = 2,
                                              verbose = 1,
                                              min_lr = 1e-6)
callbacks = [operator_vis, reduce_lr]

model.summary()

history = model.fit(in_data, out_data,
                    epochs = 200,
                    validation_split = 0.1,
                    callbacks = callbacks,
                    verbose=1 )

# def plot_qft_figure(filepath, t1 = 0, t2 = 25, tmax = 50, figscale = 12, savefig = False):

#     f = h5py.File(filepath, 'r')

#     fidelities_train = np.array(f['fidelities_train'])
#     fidelities_val = np.array(f['fidelities_val'])
#     operators = np.array(f['operators'])
#     fidelity_init = 1 - np.array(f['fidelity_initial'])[-1]
#     operator_init = np.array(f['operator_initial'])
    
#     fidelities_val = np.insert(fidelities_val, 0, fidelity_init, axis=0)
#     fidelities_train = np.insert(fidelities_train, 0, fidelity_init, axis=0)
#     operators = np.insert(operators, 0, operator_init, axis=0)
    
#     operator_targ = extract_operator_from_circuit(QFT, N)
#     kets, bras = computational_basis_labels(N, include_bras=True)

#     global_phase1 = np.mean(np.angle(operators[t1]/operator_targ))
#     operator1 = operators[t1] / np.exp(1j * global_phase1)

#     global_phase2 = np.mean(np.angle(operators[t2]/operator_targ))
#     operator2 = operators[t2] / np.exp(1j * global_phase2)
    
#     global_phase3 = np.mean(np.angle(operators[tmax]/operator_targ))
#     operator3 = operators[tmax] / np.exp(1j * global_phase3)

#     # Make figure and axis layout
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     fig = plt.figure(figsize=(figscale, figscale*(1/3 + 1/6)), tight_layout=True)
#     gs = mpl.gridspec.GridSpec(2, 3, height_ratios = [2, 1])

#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[0, 2])
    
#     # Plot antifidelity
#     fidel_val = fidelities_val[0:tmax+1]
#     fidel_train = fidelities_train[0:tmax+1]
#     ax_bot = fig.add_subplot(gs[1, :])
#     loss_plot(fidel_val, fidel_train, x_units='epochs', x_max = tmax+1, fig=fig, ax=ax_bot, log_fidelity=False)

#     # Plot operator visualizations
#     fidel1 = fidel_val[t1]
#     hinton(operator1, xlabels=kets, ylabels=bras, fig=fig, ax=ax1, title="$\\tilde{U}_{"+str(t1)+"}$")
    
#     fidel2 = fidel_val[t2]
#     hinton(operator2, xlabels=kets, ylabels=bras, fig=fig, ax=ax2, title="$\\tilde{U}_{"+str(t2)+"}$")
    
#     fidel_3 = fidel_val[tmax]
#     hinton(operator3, xlabels=kets, ylabels=bras, fig=fig, ax=ax3, title="$\\tilde{U}_{"+str(tmax)+"}$")

#     if savefig:
#         plt.savefig("assets/qft_3panel.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
#     else:
#         plt.show()
    
# plot_qft_figure('logs/operator_history_4_qubits_2019.08.27.11.03.16.h5', t1=10, t2=20, tmax=50, savefig=False)