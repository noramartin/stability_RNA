import numpy as np
import matplotlib
matplotlib.use('ps')
import matplotlib.pyplot as plt
from copy import deepcopy
from GPfunctions.general_functions import *
import pandas as pd
from os.path import isfile
import seaborn as sns
import GPfunctions.rna_structural_functions as cg
import networkx as nx
import GPfunctions.neutral_component as NC
import random
import GPfunctions.thermodynamic_functions as DG 
import GPfunctions.functions_for_sampling_with_bp_swap as sample
from GPfunctions.matplotlib_plot_functions import *
from multiprocessing import Pool
from functools import partial
from scipy.stats import spearmanr
import sys
import parameters as param

def is_decimal(string_test):
  try:
    a = float(string_test)
    return True
  except ValueError:
    return False

###############################################################################################
###############################################################################################
print  'setup'
###############################################################################################
###############################################################################################
thermodynamic_quantity = 'energygap' #'free_energy' 
test_RNAinverse = True
L = int(sys.argv[1])
num_sample_str = sys.argv[2]
number_measurements = 3 
number_walks, length_per_walk, every_Nth_seq = 5, 10**4, 100
max_no_trials_RNAinverse = 10
use_gsample = True # for sequence length > 40 only
###############################################################################################
###############################################################################################
if L <= 40:
   sample_filename = 'L' + str(L) + '_Nsamples' + num_sample_str
elif L > 40 and not use_gsample:
   sample_filename = 'fRNAL' + str(L)
elif L > 40 and use_gsample:
   sample_filename =  'gsampleL' + str(L) + '_Nsamples' + num_sample_str
filename_csv_file = './data_sampling/samplingdata'+thermodynamic_quantity+sample_filename+'_'+str(number_walks)+'_'+str(length_per_walk)+'_'+str(every_Nth_seq )+'_'+str(max_no_trials_RNAinverse)+'.csv'
filename_N_data = './data_sampling/neutral_set_size_results'+sample_filename+'_'+str(number_measurements)+'_repetition.txt'
###############################################################################################
###############################################################################################
print  'load data'
###############################################################################################
###############################################################################################
df = pd.read_csv(filename_csv_file)
try:
   neutral_network_size_list_uncorrected = np.loadtxt(filename_N_data, dtype='int')
except OverflowError:
   neutral_network_size_list_uncorrected = []
   with open(filename_N_data, 'r') as f:
      for l in f.readlines():
         neutral_network_size_list_uncorrected.append(float(l.strip()))
###
correction_factor_list = [(1-f) for  f in df['fraction of deltaG zero'].tolist()]
print 'correction factors for N, mean and standard deviation', np.mean(correction_factor_list), np.std(correction_factor_list)
###
neutral_network_size_list = [neutral_network_size_list_uncorrected[index] * (1-f) for index, f in enumerate(df['fraction of deltaG zero'].tolist())]
number_stacks_list = [cg.dotbracket_to_coarsegrained(structure).count('[') for structure in df['structure'].tolist()]
df = df.assign(N=pd.Series(neutral_network_size_list).values)
df = df.assign(stacks=pd.Series(number_stacks_list).values)
df.to_csv('./data_sampling/samplingplot_data'+thermodynamic_quantity+sample_filename+'.csv')

###############################################################################################
###############################################################################################
print  'plot in matplotlib: only deltaG, deltadeltaG and neutral component size - greyscale-compatible'
###############################################################################################
###############################################################################################
df = df.dropna()
neutral_network_size_list = df['N'].tolist()
deltadeltaG_list = df['lower quartile ddG incl inf'].tolist()
deltaG_list = df['deltaG'].tolist()
###############################################################################################
print  'simple plot'
###############################################################################################
df = df.dropna()
print df.columns
for layout in ['', 'uneven']:
  if layout == 'uneven':  
     f, ax = plt.subplots(ncols=2, figsize=(6,2.7), gridspec_kw={'width_ratios': [1, 0.6]})
  else:
     f, ax = plt.subplots(ncols=2, figsize=(7,3))
  deltadeltaG_list = df['lower quartile ddG incl inf'].tolist()
  deltaG_list = df['deltaG'].tolist()
  coarsegrained_length_list = df['length of coarse-grained structure including exterior'].tolist()
  neutral_network_size_list = df['N'].tolist()
  ###
  rounded_dG_list = [int(round(10* x)) for x in deltaG_list]
  rounded_ddG_list = [int(round(10* x)) for x in deltadeltaG_list]
  print 'Spearman correlation N-dG', spearmanr(neutral_network_size_list, deltaG_list)
  print 'Pearson correlation logN-dG', pearsonr(np.log10([N for N in neutral_network_size_list if N > 0]), [deltaG_list[i] for i, N in enumerate(neutral_network_size_list) if N > 0])
  ###
  cmap = plt.get_cmap('inferno')
  norm_ddG = matplotlib.colors.BoundaryNorm(np.arange(min(deltadeltaG_list) - 0.05, max(deltadeltaG_list) + 0.051, 0.1), cmap.N)                         
  ticks_colorbar = np.arange(0, max(deltadeltaG_list) + 0.1, 0.5)
  sc1 = ax[0].scatter(neutral_network_size_list, deltaG_list, c=deltadeltaG_list, s=1.3, linewidth=0, alpha=0.6, cmap=cmap, norm=norm_ddG)
  ax[0].set_xlabel('neutral set size')
  ax[0].set_xscale('log')
  ax[0].set_xlim(0.5*np.nanmin([N for N in neutral_network_size_list if N]), 2*np.nanmax(neutral_network_size_list))
  if layout == 'uneven':
    add_colorbar_above_plot(ax[0], sc1, r'lower quartile $|\Delta \Delta G|$ (kcal/mol)', ticks=ticks_colorbar, extra_space=True)
  else:
     add_colorbar_above_plot(ax[0], sc1, r'lower quartile $|\Delta \Delta G|$ (kcal/mol)', ticks=ticks_colorbar, extra_space=False)
  ax[0].set_ylabel(r'stability $\langle \Delta G \rangle$ (kcal/mol)')
  ##
  sns.boxplot(x='stacks', y='lower quartile ddG incl inf', data=df.dropna(), ax=ax[1], 
              order=sorted(list(set(number_stacks_list))), color='lightgrey', whis=(5,95), fliersize=2)
  ax[1].set_ylabel(r'lower quartile $|\Delta \Delta G|$ (kcal/mol)')
  ax[1].set_xlabel('number of stacks')
  for i in range(2):
      ax[i].annotate('ABCDEF'[i], xy=(0.06, 0.86), xycoords='axes fraction', fontsize='large', fontweight='bold', size=15)
  f.tight_layout()
  f.savefig('./plots_sampling/plt'+thermodynamic_quantity+sample_filename+'deltadeltaG_and_stacks'+layout+ '2.png', dpi=300, bbox_inches='tight')
  f.savefig('./plots_sampling/plt'+thermodynamic_quantity+sample_filename+'deltadeltaG_and_stacks'+layout+ '2.eps', dpi=300, bbox_inches='tight')
  plt.close('all')
  del f, ax

###############################################################################################
###############################################################################################
print  'plot in matplotlib: different ways of quantifying ddG and number of stacks'
###############################################################################################
###############################################################################################
df_nonan = df.dropna()
fractions_recorded = [column_title for column_title in df.columns if column_title.startswith('fraction of ddG above')] 
title_vs_fraction = {}
for fraction_recorded in fractions_recorded:
   if 'above' in fraction_recorded:
      fraction_as_str = fraction_recorded[-5:]
      while not is_decimal(fraction_as_str):
         fraction_as_str = fraction_as_str[1:] 
      title_vs_fraction[fraction_recorded] = float(fraction_as_str)
fractions_recorded_sorted = sorted(title_vs_fraction.keys(), key=title_vs_fraction.get)+ ['lower quartile ddG incl inf']
###
nrows = 2
ncols = (len(fractions_recorded_sorted) + nrows - 1)//nrows
f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3* ncols, 2.5 * nrows))
if len(fractions_recorded_sorted):
   for plotindex, fraction_recorded in enumerate(fractions_recorded_sorted):
      sns.boxplot(x='stacks', y=fraction_recorded, data=df_nonan.dropna(), ax=ax[plotindex//ncols, plotindex%ncols], 
                  order=sorted(list(set(number_stacks_list))), color='lightgrey', boxprops = {'linewidth': '0.3'})
      for lineindex, line in enumerate(ax[plotindex//ncols, plotindex%ncols].lines):
         if (lineindex - 4)%6 == 0:
             line.set_color('darkred')
             line.set_linewidth(1.4)
      ax[plotindex//ncols, plotindex%ncols].set_xlabel('number of stacks')
      ax[plotindex//ncols, plotindex%ncols].yaxis.set_tick_params(labelleft=True)
      if 'above' in fraction_recorded:
         fraction_as_str = title_vs_fraction[fraction_recorded]
         ax[plotindex//ncols, plotindex%ncols].set_ylabel('fraction\n'+r'$\Delta \Delta G  \geq$' + str(round(fraction_as_str, 1)) + ' kcal/mol')
      else:
         ax[plotindex//ncols, plotindex%ncols].set_ylabel('lower quartile\n' + r'$|\Delta \Delta G|$ (kcal/mol)')
   f.tight_layout()
   f.savefig('./plots_sampling/fraction_ddG_below_x_vs_quartile'+thermodynamic_quantity+sample_filename+'deltadeltaG_and_stacks_3.png', dpi=300, bbox_inches='tight')
   plt.close('all')
   del f, ax

