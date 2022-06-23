import numpy as np
import os 
os.environ["PATH"]+=os.pathsep+'/usr/bin'
import matplotlib
matplotlib.use('ps')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from GPfunctions.general_functions import *
import pandas as pd
import seaborn as sns
import GPfunctions.epistasis as epistasis
import GPfunctions.rna_structural_functions as cg
import GPfunctions.neutral_component as NC
import networkx as nx
import itertools
from GPfunctions.matplotlib_plot_functions import *
from functools import partial
from multiprocessing import Pool
import GPfunctions.thermodynamic_functions as DG
import parameters as param


L, K = 13, 4
thermodynamic_quantity =  'Boltzmann_based'  
stability_function, deltadeltaG_function, GPmapdef = param.set_parameters_functions_for_thermodynamic_quantity(thermodynamic_quantity)   
deltaGarray = np.load(param.thermodynamic_quantity_vs_filename_array[thermodynamic_quantity])
structure_invalid_test = isunfoldedstructure_or_isolated_bps 
mfearray = np.load(param.mfearray_filename)
GPmap = np.load(param.GPmapdef_to_GPmapfilename(GPmapdef))
seq_vs_NCindex_array, NCindex_vs_NCsize_all_ph = NC.find_all_NCs_parallel(GPmap, GPmapdef)
NCindex_vs_NCsize = {NCindex: NCsize for NCindex, NCsize in NCindex_vs_NCsize_all_ph.iteritems() if not structure_invalid_test(NC.get_structure_forNCindex(seq_vs_NCindex_array, GPmap, NCindex))}
NCindex_list = NCindex_vs_NCsize.keys()[:]

###############################################################################################
###############################################################################################
print  'get data: fraction of pairs of mutations with epistasis per NC'
###############################################################################################
###############################################################################################
for ignore_zero_ddG in [False, True]:
   if ignore_zero_ddG:
      description_saving = 'ignore_zero_ddG'
   else:
      description_saving = ''
   type_energy_vs_epistasis_fraction = {thermodynamic_quantity:[], 'mfe':[]}
   for energy_string, energyarray in [(thermodynamic_quantity, deltaGarray), ('mfe', mfearray), ('deltaG shuffled', deltaGarray)]:
      if energy_string != 'deltaG shuffled':
         fraction_epistasis_type_function = partial(epistasis.total_fraction_of_epistasis_in_NC, seq_vs_NCindex=seq_vs_NCindex_array, 
                                                 deltaGarray=energyarray, resolution=0.1, ignore_zero_ddG=ignore_zero_ddG, shuffle_landscape=False)                  
      elif energy_string == 'deltaG shuffled':
         fraction_epistasis_type_function = partial(epistasis.total_fraction_of_epistasis_in_NC, seq_vs_NCindex=seq_vs_NCindex_array, 
                                                 deltaGarray=energyarray, resolution=0.1, ignore_zero_ddG=ignore_zero_ddG, shuffle_landscape=True)     
      pool = Pool()
      type_energy_vs_epistasis_fraction[energy_string] = pool.map(fraction_epistasis_type_function, NCindex_list)
      pool.close()
      pool.join()
   f_list = type_energy_vs_epistasis_fraction[thermodynamic_quantity]+type_energy_vs_epistasis_fraction['mfe']+type_energy_vs_epistasis_fraction['deltaG shuffled']
   type_energy_list = [thermodynamic_quantity,]*len(type_energy_vs_epistasis_fraction[thermodynamic_quantity])+['mfe',]*len(type_energy_vs_epistasis_fraction['mfe'])+['deltaG shuffled',]*len(type_energy_vs_epistasis_fraction['deltaG shuffled'])
   df_epistasis_reformatted = pd.DataFrame.from_dict({'present in fraction of squares in neutral component': f_list, 'landscape': type_energy_list, 'NC index': NCindex_list+NCindex_list+NCindex_list})
   df_epistasis_reformatted.to_csv('./epistasis_data/alltypesepistasis'+description_saving+thermodynamic_quantity+'L'+str(L)+GPmapdef+'.csv')

###############################################################################################
###############################################################################################
print  'plot this with NCs of target size'
###############################################################################################
###############################################################################################
target_size, numbers_bps = 50000, np.arange(2, 5)  
number_bp_vs_NCindex_vs_size = {n_bp: {NCindex: abs(NCsize-target_size) for NCindex, NCsize in NCindex_vs_NCsize.iteritems() 
                              if get_dotbracket_from_int(NC.get_structure_forNCindex(seq_vs_NCindex_array, GPmap, NCindex)).count('(')==n_bp}
                              for n_bp in numbers_bps}  

list_NC_examples = [min(number_bp_vs_NCindex_vs_size[n_bp], key=number_bp_vs_NCindex_vs_size[n_bp].get) for n_bp in numbers_bps]
for ignore_zero_ddG in [False, True]:
   if ignore_zero_ddG:
      description_saving = 'ignore_zero_ddG'
   else:
      description_saving = ''
   df_epistasis_reformatted = pd.read_csv('./epistasis_data/alltypesepistasis'+description_saving+thermodynamic_quantity+'L'+str(L)+GPmapdef+'.csv')
   for NCindex in list_NC_examples:
      f = plt.figure(constrained_layout=True, figsize=(8, 3.1))
      gs = gridspec.GridSpec(ncols=5, nrows=4, figure=f)
      ax = f.add_subplot(gs[:3, :3])
      ax.axis('off')
      ax.annotate('A', xy=(0.01, 0.84), xycoords='axes fraction', fontsize='large', fontweight='bold', size=15)
      structure = get_dotbracket_from_int(NC.get_structure_forNCindex(seq_vs_NCindex_array, GPmap, NCindex))
      plot_NC_and_deltaG_discrete(NCindex, ax, f, seq_vs_NCindex_array, deltaGarray, cax=f.add_subplot(gs[3, :3]), string_saving=thermodynamic_quantity+GPmapdef, 
                        cbar_label=r'stability $\Delta G$ (kcal/mol)', structure=structure)      
      ax = f.add_subplot(gs[:, 3:5])
      sns.boxplot(data=df_epistasis_reformatted, x='landscape', y='present in fraction of squares in neutral component', color='lightgrey', ax=ax, 
                whis=(5,95), fliersize=2, order=[thermodynamic_quantity, 'deltaG shuffled', 'mfe'], boxprops = {'linewidth': '0.3'})
      column_label_to_plot_label = {thermodynamic_quantity: r'stability $\Delta G$', 'deltaG shuffled': 'shuffled stability', 'mfe': r'free energy, $G$'}
      labels = ax.get_xticklabels()
      ax.set_xticklabels([column_label_to_plot_label[l.get_text()] for l in labels], rotation=20.0, ha='right')
      ax.set_xlabel('')
      ax.set_ylabel('fraction of\npairs of substitutions\nwith epistasis')
      ax.annotate('B', xy=(0.01, 0.84), xycoords='axes fraction', fontsize='large', fontweight='bold', size=14)
      f.tight_layout()  
      seqlist_NC_array = np.argwhere(seq_vs_NCindex_array==NCindex)
      neutral_component_list_of_sequences = [tuple(seqlist_NC_array[i]) for i in range(len(seqlist_NC_array))]
      sample_seq = neutral_component_list_of_sequences[np.random.choice(len(neutral_component_list_of_sequences))]
      f.savefig('./figs_for_report/neutral_component_examples_and_epistasis'+thermodynamic_quantity+description_saving+'L'+str(L)+GPmapdef+str(NCindex)+DG.sequence_int_to_str(sample_seq)+'.png', dpi=300, bbox_inches='tight')
      f.savefig('./figs_for_report/neutral_component_examples_and_epistasis'+thermodynamic_quantity+description_saving+'L'+str(L)+GPmapdef+str(NCindex)+DG.sequence_int_to_str(sample_seq)+'.eps', dpi=300, bbox_inches='tight')
      plt.close('all')   
   f, ax = plt.subplots(figsize=(4, 3))
   sns.boxplot(data=df_epistasis_reformatted, x='landscape', y='present in fraction of squares in neutral component', color='lightgrey', ax=ax, 
             whis=(5,95), fliersize=1, order=[thermodynamic_quantity, 'deltaG shuffled', 'mfe'], boxprops = {'linewidth': '0.3'},
             medianprops= {'linewidth': 1.1, 'color': 'r'})
   column_label_to_plot_label = {thermodynamic_quantity: r'stability $\Delta G$', 'deltaG shuffled': 'shuffled stability', 'mfe': r'free energy, $G$'}
   labels = ax.get_xticklabels()
   ax.set_xticklabels([column_label_to_plot_label[l.get_text()] for l in labels], rotation=20.0, ha='right')
   ax.set_xlabel('')
   ax.set_ylabel('fraction of\npairs of substitutions\nwith epistasis')
   if ignore_zero_ddG:
      ax.set_title('only point mutations\nwith a non-zero change')
   else:
      ax.set_title(r'all point mutations')
   f.tight_layout()  
   f.savefig('./figs_for_report/epistasis'+thermodynamic_quantity+description_saving+'L'+str(L)+GPmapdef+'.png', dpi=300, bbox_inches='tight')
   f.savefig('./figs_for_report/epistasis'+thermodynamic_quantity+description_saving+'L'+str(L)+GPmapdef+'.eps', dpi=300, bbox_inches='tight')
   plt.close('all')   


 

