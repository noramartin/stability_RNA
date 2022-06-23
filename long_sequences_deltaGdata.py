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


###############################################################################################
###############################################################################################
print  'setup'
###############################################################################################
###############################################################################################
thermodynamic_quantity = 'energygap' #'free_energy' 
test_RNAinverse = True
L = int(sys.argv[1])
num_sample_str = sys.argv[2]
n_cpus = int(sys.argv[3])
if num_sample_str == 'all':
   num_sample = 'all'
else:
   num_sample = int(num_sample_str)
number_walks, length_per_walk, every_Nth_seq = 5, 10**4, 100
max_no_trials_RNAinverse = 10
print 'length=', L, num_sample
use_gsample = True # for sequence length > 40 only
use_existing_sample = True
###############################################################################################
###############################################################################################
if L <= 40:
   sample_filename = 'L' + str(L) + '_Nsamples' + str(num_sample)
elif L > 40 and not use_gsample:
   sample_filename = 'fRNAL' + str(L)
elif L > 40 and use_gsample:
   sample_filename =  'gsampleL' + str(L) + '_Nsamples' + str(num_sample)
###############################################################################################
###############################################################################################

filename_csv_file = './data_sampling/samplingdata'+thermodynamic_quantity+sample_filename+'_'+str(number_walks)+'_'+str(length_per_walk)+'_'+str(every_Nth_seq )+'_'+str(max_no_trials_RNAinverse)+'.csv'


if not isfile(filename_csv_file) or not use_existing_sample:
   stability_function, deltadeltaG_function, GPmapdef = param.set_parameters_functions_for_thermodynamic_quantity(thermodynamic_quantity)   
   ###############################################################################################
   ###############################################################################################
   print  'structure sample'
   ###############################################################################################
   ###############################################################################################
   filename_structure_list = './data_sampling/structure_sample'+sample_filename+'.txt'
   if isfile(filename_structure_list) and use_existing_sample:
      structure_sample = sample.read_structure_list(filename_structure_list)
   elif L > 40:
      print 'no method defined' 
   else:
      structure_sample = sample.generate_all_allowed_dotbracket(L, max_no_trials_RNAinverse, num_sample, allow_isolated_bps=False, test_RNAinverse=test_RNAinverse)
      sample.save_structure_list(structure_sample, filename_structure_list)
   print 'sample size:', len(structure_sample)
   ###############################################################################################
   ###############################################################################################
   print  'estimate stability'
   ###############################################################################################
   ###############################################################################################
   if thermodynamic_quantity == 'free_energy': # need stability in random walk to check for degenerate seq 
      site_scanning_function = partial(sample.site_scanning_random_walk_free_energy, stability_function=stability_function, thermo_function=DG.get_mfe, thermo_functionddG=deltadeltaG_function, 
                                number_walks=number_walks, length_per_walk=length_per_walk, every_Nth_seq=every_Nth_seq)
   else:
      site_scanning_function = partial(sample.site_scanning_random_walk_deltaG, thermo_function=stability_function, thermo_functionddG=deltadeltaG_function, 
                                number_walks=number_walks, length_per_walk=length_per_walk, every_Nth_seq=every_Nth_seq)
   pool = Pool(processes = n_cpus)
   stability_results_list = pool.map(site_scanning_function, structure_sample)
   pool.close()
   pool.join()
   deltaG_estimate_list = [e[0] for e in stability_results_list]
   lower_quartile_ddGwithinf_list = [e[1] for e in stability_results_list]
   deltadeltaG_median_list = [e[2] for e in stability_results_list]
   threshold_ddG_list = stability_results_list[0][3].keys()
   deltadeltaG_below_x_list = {x: [e[3][x] for e in stability_results_list] for x in threshold_ddG_list}
   threshold_fullddG_list = stability_results_list[0][4].keys()
   full_deltadeltaG_above_x_list = {x: [e[4][x] for e in stability_results_list] for x in threshold_fullddG_list}
   fraction_deltaG_zero_list = [e[5] for e in stability_results_list] 
   ###############################################################################################
   ###############################################################################################
   print  'prepare data'
   ###############################################################################################
   ###############################################################################################
   structure_sample2 = [structure_sample[i] for i in range(len(structure_sample)) if deltaG_estimate_list[i]]
   deltaG_estimate_list2 = [deltaG_estimate_list[i] for i in range(len(structure_sample)) if deltaG_estimate_list[i]]
   deltadeltaG_median_list2 = [deltadeltaG_median_list[i] for i in range(len(structure_sample)) if  deltaG_estimate_list[i]]
   lower_quartile_ddGwithinf_list2 = [lower_quartile_ddGwithinf_list[i] for i in range(len(structure_sample)) if  deltaG_estimate_list[i]]
   deltadeltaG_below_x_list2 = {x: [list_fraction[i] for i in range(len(list_fraction)) if  deltaG_estimate_list[i]] for x, list_fraction in deltadeltaG_below_x_list.iteritems()}
   fraction_deltaG_zero_list2 = [fraction_deltaG_zero_list[i] for i in range(len(structure_sample)) if  deltaG_estimate_list[i]]
   length_of_coarsegrained_structure_incl_ends = [len(cg.dotbracket_to_coarsegrained(structure)) for structure in structure_sample2]
   higherlevel_coarsegrained_structure = [cg.dotbracket_to_higherlevelcoarsegrained(structure) for structure in structure_sample2]
   length_of_higherlevel_coarsegrained_structure = [len(s) for s in higherlevel_coarsegrained_structure]
   number_basepairs_list = [structure.count('(') for structure in structure_sample2]
   ###############################################################################################
   ###############################################################################################
   print  'save data'
   ###############################################################################################
   ###############################################################################################
   dict_df = {'structure': structure_sample2, 
              'deltaG': deltaG_estimate_list2,
              'median deltadeltaG': deltadeltaG_median_list2, 
              'number of basepairs': number_basepairs_list, 
              'fraction of deltaG zero': fraction_deltaG_zero_list2, 
              'lower quartile ddG incl inf': lower_quartile_ddGwithinf_list2,
              'length of coarse-grained structure including exterior': length_of_coarsegrained_structure_incl_ends, 
              'higher-level coarse-grained structure': higherlevel_coarsegrained_structure, 
              'length of higher-level coarse-grained structure': length_of_higherlevel_coarsegrained_structure}
   for x, list_fraction in deltadeltaG_below_x_list.iteritems():
      dict_df['fraction of absolute ddG below '+str(x)] = list_fraction
   for x, list_fraction in full_deltadeltaG_above_x_list.iteritems():
      dict_df['fraction of ddG above '+str(x)] = list_fraction
   df = pd.DataFrame.from_dict(dict_df)
   df.to_csv(filename_csv_file)
   filename_final_structure_list = './data_sampling/final_structure_sample' + sample_filename + '.txt'
   sample.save_structure_list(structure_sample, filename_final_structure_list)
