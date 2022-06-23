import numpy as np
import thermodynamic_functions as DG
import subprocess
import sys
sys.path.insert(0,'/scratch/nsm36/ViennaRNA/lib/python2.7/site-packages') #for ViennaRNA in TCM
import RNA
import neutral_component as NC
import random
import networkx as nx
from general_functions import isunfoldedstructure
import rna_structural_functions as cg
from neutral_component import neighbours_g
from multiprocessing import Pool
import time
from os.path import isfile


K = 4

############################################################################################################
##sample x sequences per structure if full enumeration feasible
############################################################################################################
def find_x_sequences_per_structure(GPmap, number_sequences_per_structure, structure_invalid_test=isunfoldedstructure):
   """returns number_sequences_per_structure structures drawn for each strucutre that is present in the GPmap 
   and passes the structure_invalid_test function with False;
   sampling is performed with replacement"""
   structure_list, seq_list = list(set(GPmap.copy().flat)), []
   for structure_int in structure_list:
      if not structure_invalid_test(structure_int):
         seqlist_NN_array = np.argwhere(GPmap==structure_int)
         for seq_index in np.random.choice(len(seqlist_NN_array), number_sequences_per_structure, replace=True):
            seq_list.append(tuple(seqlist_NN_array[seq_index]))
            assert GPmap[tuple(seqlist_NN_array[seq_index])] == structure_int
   return seq_list

############################################################################################################
## get sample of structures
############################################################################################################

def return_all_allowed_structures_starting_with(structure, L, allow_isolated_bps=False):
   """recursively build up all possible structures of length L;
   already has some constraints of dot-bracket notations built in:
   only one closed bracket for each upstream open bracket, all brackets closed by the end, length of hairpin loops,
   optionally isolated base pairs"""
   if len(structure) == L:
      return [structure]
   assert len(structure) < L
   assert L-len(structure) <= 35
   structure_list = []
   if (len(structure) >= 2 and structure[-1] == '(' and structure[-2] != '(' and not allow_isolated_bps):
      structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps) #need to add ( to avoid isolated bp
   elif (len(structure) == 1 and structure[-1] == '(' and not allow_isolated_bps):
      structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps) #need to add ( to avoid isolated bp
   elif (len(structure) >= 2 and structure[-1] == ')' and structure[-2] != ')' and not allow_isolated_bps):
      structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps) #need to add ) to avoid isolated bp
   elif len(structure) <= 4: #at the beginning of a structure can only have opening brackets/loops
      structure_list += return_all_allowed_structures_starting_with(structure+'.', L, allow_isolated_bps) 
      structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps) 
   elif structure.count('(') > structure.count(')')+L-len(structure):
      pass  #cannot close all base pairs anymore, return empty list
   elif structure.count('(') == structure.count(')')+L-len(structure) and '(' not in structure[-3:]: #need to close base pairs
      structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps)    
   elif structure.count('(') > structure.count(')') and allow_isolated_bps and '(' not in structure[-3:]: #upstream bps are open, so closing also allowed
      structure_list += return_all_allowed_structures_starting_with(structure+'.', L, allow_isolated_bps)
      if L-len(structure) > 3:
         structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps)
      structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps)
   elif structure.count('(') > structure.count(')') and structure[-1] == ')' and not allow_isolated_bps: # everything allowed
      structure_list += return_all_allowed_structures_starting_with(structure+'.', L, allow_isolated_bps)
      if L-len(structure) > 3:
         structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps)
      structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps)
   elif structure.count('(') > structure.count(')')+1 and not allow_isolated_bps and '(' not in structure[-3:]: #at least two upstream bps are open, so closing also allowed with non-isolated bps
      structure_list += return_all_allowed_structures_starting_with(structure+'.', L, allow_isolated_bps)
      if L-len(structure) > 3:
         structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps)
      structure_list += return_all_allowed_structures_starting_with(structure+')', L, allow_isolated_bps)
   else: # can open new base pairs or introduce loop
      structure_list += return_all_allowed_structures_starting_with(structure+'.', L, allow_isolated_bps)
      if L-len(structure) > 3:
         structure_list += return_all_allowed_structures_starting_with(structure+'(', L, allow_isolated_bps)
   return structure_list

def generate_all_allowed_dotbracket(L, max_no_trials_RNAinverse=5, num_samples='all', allow_isolated_bps=False, test_RNAinverse=True):
   """recursively build up all possible structures of length L;
    - max_no_trials_RNAinverse: maximum number of times RNAinverse is performed to test if any sequences fold into the given structure
    - num_samples is either 'all' (return full list) or a number (return sample from list)
    - allow_isolated_bps (True/False) to control whether isolated base pairs are permitted
    - test_RNAinverse (True/False) to control whether RNAinverse is performed to test if any sequences fold into the given structure 
      or whether all allowed structures are returned"""
   assert L<31 # avoid overflow errors
   potential_structures = [s for s in return_all_allowed_structures_starting_with('', L, allow_isolated_bps=allow_isolated_bps) if cg.is_likely_to_be_valid_structure(s, allow_isolated_bps=allow_isolated_bps)]
   random.shuffle(potential_structures)
   if num_samples == 'all' and not test_RNAinverse:
      return potential_structures
   elif not test_RNAinverse:
      return potential_structures[:num_samples]
   final_structures = []
   for sindex, s in enumerate(potential_structures):
      if len(find_start_sequence(s, max_no_trials_RNAinverse)) > 0:
         final_structures.append(s)
         if num_samples != 'all' and len(final_structures) >= num_samples:
            return final_structures
   return final_structures

def save_structure_list(structure_sample, filename):
   """save list of structures to file, one structure per line"""
   with open(filename, 'w') as textfile:
      for structure in structure_sample:
         textfile.write(structure+'\n')

def read_structure_list(filename):
   """read list of structures from file"""
   with open(filename, 'r') as textfile:
      structure_sample = [str(line.strip()) for line in textfile.readlines()]
   for structure in structure_sample:
      validity_structure = cg.is_likely_to_be_valid_structure(structure, allow_isolated_bps=False)
      if not validity_structure:
         print structure, 'does not seem to be a valid structure'
      assert validity_structure
   return structure_sample


def g_sampling_structure_sample(required_sample_size, L, const_Nbp=np.nan):
   """adapted from free-energy-based neutral set size prediction code"""
   structure_sample = set([])
   while len(structure_sample) < required_sample_size * 10**3:
      sequence_sample_list = [tuple(np.random.choice([0, 1, 2, 3], replace=True, size=L)) for i in range(10**4)]     
      pool = Pool()
      pool_result = pool.map(DG.get_mfe_structure, sequence_sample_list)
      pool.close()
      pool.join()
      for structure in pool_result:
         nbp = structure.count('(')
         if nbp > 0 and np.isnan(const_Nbp) or nbp == const_Nbp:
            structure_sample.add(structure[:])
   structure_list = list(structure_sample)
   no_stacks_list = [cg.dotbracket_to_coarsegrained(s).count('[') for s in structure_list]
   weight_list = [1/float(no_stacks_list.count(n)) for n in no_stacks_list]
   weight_list_normalised = np.divide(weight_list, float(np.sum(weight_list)))
   return list(np.random.choice(structure_list, required_sample_size, p=weight_list_normalised, replace=False))

############################################################################################################
## run RNAinverse
############################################################################################################

def find_start_sequence(dotbracket_structure, max_no_trials=5):
   """run inverse_fold until a sequence whose minimum free energy is the given structure,
   with an energy gap to the next lowest-lying structure of > 0 (doe to ViennaRNA energy rounding to first decimal)
   if no result is found, run at most max_no_trials times"""
   L = len(dotbracket_structure)
   for trial in range(max_no_trials):
      random_startseq_int = tuple(np.random.randint(0, K, size=L))
      (sequence_str, distance_from_target_str) = RNA.inverse_fold(DG.sequence_int_to_str(random_startseq_int), dotbracket_structure)
      if distance_from_target_str < 0.01 and DG.get_DeltaG_energygap(DG.sequence_str_to_int(sequence_str)) > 0.01:
         assert DG.get_mfe_structure(DG.sequence_str_to_int(sequence_str)) == dotbracket_structure
         return DG.sequence_str_to_int(sequence_str)
   return tuple()



############################################################################################################
## site scanning method
############################################################################################################
## site scanning method adapted from the description Weiss, Marcel; Ahnert, Sebastian E. (2020): 
## Supplementary Information from Using small samples to estimate neutral component size and robustness in the genotype-phenotype map 
## of RNA secondary structure. The Royal Society. Journal contribution. https://doi.org/10.6084/m9.figshare.12200357.v2
## here base pair swaps are used as well as point mutations

def rw_sitescanning(startseq, length_per_walk, every_Nth_seq):
   """perform a site-scanning random walk of length length_per_walk starting from startseq (tuple of ints) and subsample every_Nth_seq;
   site-scanning method following Weiss and Ahnert (2020), Royal Society Interface
   neutrality is defined based on RNAfold alone (without checking if mfe level degenerate)"""
   seq_list_rw, current_seq, dotbracket_structure = [], tuple([g for g in startseq]), DG.get_mfe_structure(startseq)
   L, K, site_to_scan = len(startseq), 4, 0
   seq_list_rw.append(current_seq)
   neutral_neighbours_startseq = [g_mut for g_mut in neighbours_g(current_seq, K, L) ]
   if len(neutral_neighbours_startseq) == 0:
      return [current_seq,]
   while len(seq_list_rw) < length_per_walk:
      if dotbracket_structure[site_to_scan] == '.':
         neighbours_given_site = NC.neighbours_g_given_site(current_seq, K, L, site_to_scan)
      else:
         neighbours_given_site = NC.bp_swaps_g_given_site(current_seq, site_to_scan, dotbracket_structure)
      neighbours_given_site_shuffled = [tuple(neighbours_given_site[i]) for i in np.random.choice(len(neighbours_given_site), size=len(neighbours_given_site), replace=False)]
      for g in neighbours_given_site_shuffled:
         if dotbracket_structure == DG.get_mfe_structure(g):
            current_seq = tuple([c for c in g])
            seq_list_rw.append(current_seq)
            break
      site_to_scan = (site_to_scan+1)%L  
   assert len(seq_list_rw) == length_per_walk
   return [tuple(seq_list_rw[i]) for i in np.random.choice(length_per_walk, size=length_per_walk//every_Nth_seq, replace=False)]
      
def get_x_random_walks(dotbracket_structure, length_per_walk, every_Nth_seq, number_walks):
   """execute rw_sitescanning for dotbracket_structure number_walks times;
   length_per_walk and every_Nth_seq are passed directly to the site-scanning random walk function"""
   start_sequences, seq_list_rw = [], []
   while len(start_sequences) < number_walks:
      RNAinverse_result = find_start_sequence(dotbracket_structure)
      if len(RNAinverse_result): #empty tuple if RNAinverse fails
         start_sequences.append(RNAinverse_result)
   for startseq in start_sequences:
      seq_list_rw += rw_sitescanning(startseq, length_per_walk, every_Nth_seq) 
   return seq_list_rw

############################################################################################################
## run site scanning random walk and evaluate stability
############################################################################################################
def site_scanning_random_walk_free_energy(dotbracket_structure, stability_function, thermo_function, thermo_functionddG, number_walks, length_per_walk=1000, every_Nth_seq=1):
   """get site-scanning random walks for the dotbracket_structure:
   - stability_function determines which sequences are stable and non-degenerate
   - thermo_function, thermo_functionddG are the functions used to evaluate sequence free energy and mutational free energy change
   - length_per_walk, every_Nth_seq, number_walks are passed to get_x_random_walks
   - dG is calculated for every sequence in the random walk output and deltadeltaG for every stable one (deltaG >0)
   - return 1: mean stability of sequences with stability > 10.0**(-5) (no degenerate mfe level)
   - return 2: lower quartile of deltadeltaG distribution (including infinities)
   - return 3: median of deltadeltaG distribution (excluding infinities)
   - return 4: x vs fraction of deltadeltaG values below x as dictionary (including infinities)
   - return 5: fraction of sequences with stability < 10.0**(-5) (i.e. degenerate mfe level)
   """
   starttime = time.time()
   L, K = len(dotbracket_structure), 4
   assert length_per_walk > L
   seq_list_rw = get_x_random_walks(dotbracket_structure, length_per_walk, every_Nth_seq, number_walks)
   deltaG_list_rw = [stability_function(g) for g in seq_list_rw]
   landscape_value_list_rw = [thermo_function(g) for g in seq_list_rw]
   stable_sequences_rw, landscape_value_list_rw_stable_seq = zip(*[(seq, G) for seq, dG, G in zip(seq_list_rw, deltaG_list_rw, landscape_value_list_rw) if abs(dG) > 10.0**(-5)])
   stable_deltaG_list = [dG for dG in deltaG_list_rw if abs(dG) > 10.0**(-5)]
   deltadeltaG_list = []
   for gindex, g in enumerate(stable_sequences_rw):
      deltadeltaG_list += [thermo_functionddG(g, g_mut) for g_mut in neighbours_g(g, K, L)]
   noninf_abs_ddG_list = np.abs([ddG for ddG in deltadeltaG_list if not np.isinf(ddG)])
   print 'finished site-scanning for', dotbracket_structure, 'in', time.time() - starttime
   print  'mean energy', np.mean(landscape_value_list_rw)
   if len(stable_sequences_rw) >  0 and len(noninf_abs_ddG_list) >  0:
      lower_quartile_ddGwithinf = np.percentile(np.abs(deltadeltaG_list), 25)
      fraction_of_zero_deltaG = 1.0 -len(stable_sequences_rw)/float(len(seq_list_rw))
      fraction_of_ddG_below_x = {x: len([ddG for ddG in deltadeltaG_list if np.abs(ddG) <= x + 0.001])/float(len(deltadeltaG_list)) for x in [0.2, 0.4, 0.6, 0.8, 1.0]}
      fraction_of_ddG_above_x_including_sign = {x: len([ddG for ddG in deltadeltaG_list if ddG >= x - 0.001])/float(len(deltadeltaG_list)) for x in [-0.8, -0.6, -0.4, -0.2, 0]}
      return np.mean(landscape_value_list_rw), lower_quartile_ddGwithinf, np.median(noninf_abs_ddG_list), fraction_of_ddG_below_x, fraction_of_ddG_above_x_including_sign, fraction_of_zero_deltaG
   else:
      return None, None, None, {x: None for x in [0.2, 0.4, 0.6, 0.8, 1.0]}, {x: None for x in [-0.8, -0.6, -0.4, -0.2, 0]}, None



def site_scanning_random_walk_deltaG(dotbracket_structure, thermo_function, thermo_functionddG, number_walks, length_per_walk=1000, every_Nth_seq=1):
   """get site-scanning random walks for the dotbracket_structure:
   - thermo_function, thermo_functionddG are the functions used to evaluate sequence stability and mutational strability impact
   - length_per_walk, every_Nth_seq, number_walks are passed to get_x_random_walks
   - dG is calculated for every sequence in the random walk output and deltadeltaG for every stable one (deltaG >0)
   - return 1: mean stability of sequences with stability > 10.0**(-5) (no degenerate mfe level)
   - return 2: lower quartile of deltadeltaG distribution (including infinities)
   - return 3: median of deltadeltaG distribution (excluding infinities)
   - return 4: x vs fraction of deltadeltaG values below x as dictionary (including infinities)
   - return 5: fraction of sequences with stability < 10.0**(-5) (i.e. degenerate mfe level)
   """
   starttime = time.time()
   L, K = len(dotbracket_structure), 4
   assert length_per_walk > L
   seq_list_rw = get_x_random_walks(dotbracket_structure, length_per_walk, every_Nth_seq, number_walks)
   deltaG_list_rw = [thermo_function(g) for g in seq_list_rw]
   stable_sequences_rw = [seq for seq_i, seq in enumerate(seq_list_rw) if abs(deltaG_list_rw[seq_i]) > 10.0**(-5)]
   stable_deltaG_list = [dG for dG in deltaG_list_rw if abs(dG) > 10.0**(-5)]
   deltadeltaG_list = []
   for gindex, g in enumerate(stable_sequences_rw):
      deltadeltaG_list += [thermo_functionddG(g, g_mut) for g_mut in neighbours_g(g, K, L)]
   noninf_abs_ddG_list = np.abs([ddG for ddG in deltadeltaG_list if not np.isinf(ddG)])
   print 'finished site-scanning for', dotbracket_structure, 'in', time.time() - starttime
   if len(stable_deltaG_list) >  0 and len(noninf_abs_ddG_list) >  0:
      lower_quartile_ddGwithinf = np.percentile(np.abs(deltadeltaG_list), 25)
      fraction_of_zero_deltaG = 1.0 -len(stable_deltaG_list)/float(len(deltaG_list_rw))
      fraction_of_ddG_below_x = {x: len([ddG for ddG in deltadeltaG_list if np.abs(ddG) <= x + 0.001])/float(len(deltadeltaG_list)) for x in [0.2, 0.4, 0.6, 0.8, 1.0]}
      fraction_of_ddG_above_x_including_sign = {x: len([ddG for ddG in deltadeltaG_list if ddG >= x - 0.001])/float(len(deltadeltaG_list)) for x in [-0.8, -0.6, -0.4, -0.2, 0]}
      return np.mean(stable_deltaG_list), lower_quartile_ddGwithinf, np.median(noninf_abs_ddG_list), fraction_of_ddG_below_x, fraction_of_ddG_above_x_including_sign, fraction_of_zero_deltaG
   else:
      return None, None, None, {x: None for x in [0.2, 0.4, 0.6, 0.8, 1.0]}, {x: None for x in [-0.8, -0.6, -0.4, -0.2, 0]}, None

def site_scanning_random_walk_ddG_distribution(dotbracket_structure, thermo_function, thermo_functionddG, number_walks, length_per_walk=1000, every_Nth_seq=1):
   """get site-scanning random walks for the dotbracket_structure:
   - thermo_function, thermo_functionddG are the functions used to evaluate sequence stability and mutational strability impact
   - length_per_walk, every_Nth_seq, number_walks are passed to get_x_random_walks
   - dG is calculated for every sequence in the random walk output and deltadeltaG for every stable one (deltaG >0)
   - return 1: mean stability of sequences with stability > 10.0**(-5) (no degenerate mfe level)
   - return 2: lower quartile of deltadeltaG distribution (including infinities)
   - return 3: median of deltadeltaG distribution (excluding infinities)
   - return 4: x vs fraction of deltadeltaG values below x as dictionary (including infinities)
   - return 5: fraction of sequences with stability < 10.0**(-5) (i.e. degenerate mfe level)
   """
   starttime = time.time()
   L, K = len(dotbracket_structure), 4
   assert length_per_walk > L
   seq_list_rw = get_x_random_walks(dotbracket_structure, length_per_walk, every_Nth_seq, number_walks)
   deltaG_list_rw = [thermo_function(g) for g in seq_list_rw]
   stable_sequences_rw = [seq for seq_i, seq in enumerate(seq_list_rw) if abs(deltaG_list_rw[seq_i]) > 10.0**(-5)]
   deltadeltaG_list = []
   for gindex, g in enumerate(stable_sequences_rw):
      deltadeltaG_list += [thermo_functionddG(g, g_mut) for g_mut in neighbours_g(g, K, L)]
   return deltadeltaG_list
############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   print '\n\ntest functions_for_sampling_with_bp_swap'
   from general_functions import get_frequencies_in_array, get_dotbracket_from_int
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   ## plot true vs estimated NNS for L=12
   print 'start test'
   L,  GPmapdef = 13, 'mfe' 
   GPmap = np.load('./GPmapdata/GPmap_L'+str(L)+GPmapdef+'.npy')
   ph_vs_f = get_frequencies_in_array(GPmap)
   print 'frequency from full enumeration finished'
   P_int_list = list(ph_vs_f.keys())
   structure_list_fullenumeration = [get_dotbracket_from_int(p) for p in P_int_list]
   print '\n\n------\nall L=13 structures for quick check\n-------\n\n'
   list_structures_sampling_testinverse = generate_all_allowed_dotbracket(L, num_samples='all', allow_isolated_bps=False, test_RNAinverse=True, max_no_trials_RNAinverse= 20)
   list_structures_sampling = generate_all_allowed_dotbracket(L, num_samples='all', allow_isolated_bps=False, test_RNAinverse=False)
   for structure in structure_list_fullenumeration:
      assert structure in list_structures_sampling
      if structure not in list_structures_sampling_testinverse:
         print 'RNAinverse did not find any sequences for structure', structure, 'with phenotype frequency', ph_vs_f[DG.dotbracket_to_int(structure)]/float(4**L)
      
   print '\n\n------\nsite - scanning: verify that number correct and all on correct neutral set\n-------\n\n'
   L = 13
   list_structures_sampling_noisolatedbp = generate_all_allowed_dotbracket(L, num_samples='all', allow_isolated_bps=False, test_RNAinverse=True)
   length_per_walk, every_Nth_seq, number_walks = 15, 3, 7
   for dotbracket_structure in list_structures_sampling_noisolatedbp:
      if dotbracket_structure.count('(') > 0 :
         rw_seq_sample = get_x_random_walks(dotbracket_structure, length_per_walk, every_Nth_seq, number_walks)
         assert len(rw_seq_sample) == (length_per_walk // every_Nth_seq) * number_walks
         assert len([seq for seq in rw_seq_sample if DG.get_mfe_structure(seq) == dotbracket_structure or abs(get_G_arbitrary_sequence_structure_pair(seq, dotbracket_structure) - get_mfe(seq)) < 0.001])  == len(rw_seq_sample)
   print '\n\n-------------\n\n'
