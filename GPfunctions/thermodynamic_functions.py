import sys
sys.path.insert(0,'/scratch/nsm36/ViennaRNA/lib/python2.7/site-packages') #for ViennaRNA in TCM
import RNA
import numpy as np
from rna_structural_functions import sequence_compatible_with_basepairs
from os.path import isfile

index_to_base = {0: 'A', 1: 'C', 2: 'U', 3: 'G'}
base_to_number={'A':0, 'C':1, 'U':2, 'G':3, 'T':2}
db_to_bin = {'.': '00', '(': '10', ')': '01', '_': '00', '[': '10', ']': '01'}

RNA.cvar.uniq_ML = 1 # global switch for unique multiloop decomposition
model = RNA.md()
model.noLP = 1 # no isolate base pairs
model.pf_smooth = 0 # deactivate partition function smoothing
###############################################################################################
## Bloom et al. (2005) stability definition
###############################################################################################

def get_deltadeltaG_Boltzmann(sequence_indices, newsequence_indices):
   """calculate the mutational stability impact of a mutation, deltadeltaG,
    for the Boltzmann-based stability definition;
    mutational stability is the change in stability is the structure is kept constant"""
   wt = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
   (wt_structure, wt_G) = wt.mfe()
   if sequence_compatible_with_basepairs(sequence_int_to_str(newsequence_indices), wt_structure):
      wt.exp_params_rescale(wt_G)
      (prob_vector_wt, dG_wt) = wt.pf()
      wt_p = wt.pr_structure(wt_structure)
      mut = RNA.fold_compound(sequence_int_to_str(newsequence_indices), model)
      (mut_mfe_struct, mut_min_G) = mut.mfe()
      mut.exp_params_rescale(mut_min_G)
      (prob_vector_mut, dG_mut) = mut.pf()
      mut_p = mut.pr_structure(wt_structure)
      return delta_G_from_p(mut_p) - delta_G_from_p(wt_p)
   else:
      return - np.inf

def get_deltadeltaG_Boltzmann_fast(sequence_indices, newsequence_indices, GPmap, deltaGarray):
   """calculate the mutational stability impact of a mutation, deltadeltaG,
    for the Boltzmann-based stability definition when all deltaG values are known:
      this is faster than calculating deltadeltaG from scratch as deltadeltaG can be 
      computed quickly from substracting precomputed deltaG values if the structure does not change"""
   ph_old, ph_new = GPmap[tuple(sequence_indices)], GPmap[tuple(newsequence_indices)]
   if ph_old != ph_new:
      return get_deltadeltaG_Boltzmann(sequence_indices, newsequence_indices)
   else:
      return deltaGarray[tuple(newsequence_indices)] - deltaGarray[tuple(sequence_indices)]

def get_DeltaG_Boltzmann(sequence_indices):
   """calculate Boltzman-based stability"""
   a = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
   (mfe_structure, mfe) = a.mfe()
   a.exp_params_rescale(mfe)
   (prob_vector, dG) = a.pf()
   prob_mfe = a.pr_structure(mfe_structure)
   if abs(1.0-prob_mfe)<10.0**(-8): #happens in cases such as CAACAAACAAACAACAAC, which only forms a single structure (undefined) because no possible base pairs
      return np.inf
   return delta_G_from_p(prob_mfe)


def delta_G_from_p(p):
   """calculate Boltzman-based stability from Boltzmann probability"""
   kbT = RNA.exp_param().kT/1000.0 ## from https://github.com/ViennaRNA/ViennaRNA/issues/58
   assert 0 < p < 1.0
   K = p/(1-p)
   return kbT*np.log(K)  

###############################################################################################
## energy gap deltaG definition
###############################################################################################

def repeat_subopt_another_time(G_range, fold_compound_seq, mfe, mfe_structure, structure_vs_energy):
   """consider subopt calculation finished  if at least one structure other than the mfe structure is returned and 
   the subopt search range is at least 0.1 larger than the identified energy gap (to ensure structures whose energy is overestimated in the subopt energy model are captured)"""
   subopt_structure_vs_G = {alternativestructure: fold_compound_seq.eval_structure(alternativestructure) for alternativestructure in structure_vs_energy 
                               if mfe_structure != alternativestructure}
   if len(subopt_structure_vs_G) > 0 and G_range > 100.0*(0.1+abs(min(subopt_structure_vs_G.values()) - mfe)): 
      return False, subopt_structure_vs_G
   else:
      return True, subopt_structure_vs_G

def convert_result_to_dict(structure, energy, data):
   """ function needed for subopt from ViennaRNA documentation: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf"""
   if not structure == None:
      data[structure] = energy

def get_DeltaG_energygap(sequence_indices):
   """ for the sequence (in index notation) calculate the energy-gap-based stability;
   code adapted from ViennaRNA documentation: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf;
   subopt is run for a small energy range at first and then a larger one, until a suboptimal structure is identified"""
   RNA.cvar.uniq_ML, repeat_subopt, structure_vs_energy = 1, True, {}
   a = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
   (mfe_structure, mfe) = a.mfe()
   G_range0, G_range = 0.11*100.0, 0.1*100.0 # subopt input different unit (100 conversion); start with a small range of G and increase by 0.1kcal/mol
   while repeat_subopt:
      a.subopt_cb(int(G_range), convert_result_to_dict, structure_vs_energy)
      repeat_subopt, subopt_structure_vs_G = repeat_subopt_another_time(G_range, a, mfe, mfe_structure, structure_vs_energy)
      G_range += G_range0
   assert min(subopt_structure_vs_G.values()) >= mfe
   return min(subopt_structure_vs_G.values()) - mfe
   

def return_structures_in_energy_range(G_range_kcal, sequence_indices):
   """ return structures in energy range (kcal/mol) of G_range_kcal from the mfe of sequence sequence_indices;
   code adapted from ViennaRNA documentation: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf;
   subopt is run for a small energy range at first and then a larger one, until a suboptimal structure is identified"""
   RNA.cvar.uniq_ML, structure_vs_energy = 1, {} # Set global switch for unique multiloop decomposition
   fold_compound_seq = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
   (mfe_structure, mfe) = fold_compound_seq.mfe()
   fold_compound_seq.subopt_cb(int(G_range_kcal*100.0*1.1), convert_result_to_dict, structure_vs_energy)
   subopt_structure_vs_G = {alternativestructure: fold_compound_seq.eval_structure(alternativestructure) for alternativestructure in structure_vs_energy 
                               if abs(fold_compound_seq.eval_structure(alternativestructure)-mfe) <= G_range_kcal}
   subopt_structure_vs_G[mfe_structure] = mfe
   return subopt_structure_vs_G


def get_deltadeltaG_energygap(sequence_indices, newsequence_indices, delta_G_wt=-1):
   """ calculate mutational stability impact for energy-gap-based stability;
   mutational stability is the change in stability is the structure is kept constant"""
   wt_structure = get_mfe_structure(sequence_indices)
   mut_structure = get_mfe_structure(newsequence_indices)
   if wt_structure == mut_structure:
      return get_DeltaG_energygap(newsequence_indices) - get_DeltaG_energygap(sequence_indices)
   elif delta_G_wt < 0: #deltaG of wildtype is not given and has to be calculates
      delta_G_wt = get_DeltaG_energygap(sequence_indices) 
   deltaG_mut =  get_mfe(newsequence_indices) - get_G_arbitrary_sequence_structure_pair(newsequence_indices, wt_structure)
   assert deltaG_mut <= 0
   return deltaG_mut - delta_G_wt

def get_deltadeltaG_energygap_fast(sequence_indices, newsequence_indices, GPmap, deltaGarray):
   """ calculate mutational stability impact for energy-gap-based stability 
   based partly on pre-computed stability data"""
   ph_old, ph_new = GPmap[tuple(sequence_indices)], GPmap[tuple(newsequence_indices)]
   if ph_old != ph_new:
      return get_deltadeltaG_energygap(sequence_indices, newsequence_indices, delta_G_wt=deltaGarray[tuple(sequence_indices)])
   else:
      return deltaGarray[tuple(newsequence_indices)] - deltaGarray[tuple(sequence_indices)]



###############################################################################################
## focus on minimum-free energy structure and energy: 
## not tested whether there is an energy gap between top two structures
###############################################################################################

def get_mfe(sequence_indices):
   """get minimum free energy for the sequence in integer format"""
   a = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
   (mfe_structure, mfe) = a.mfe()
   return mfe

def get_deltamfe(sequence_indices, newsequence_indices):
   """get minimum free energy impact of a mutation, deltamfe: 
   change in stability when changing sequence, but keeping structure constant"""
   a = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
   (mfe_structure, mfe) = a.mfe()
   return get_G_arbitrary_sequence_structure_pair(newsequence_indices, mfe_structure)-mfe

def get_deltamfe_fast(sequence_indices, newsequence_indices, GPmap, mfearray):
   """calculate the mutational free energy impact of a mutation, deltadeltaG,
      when all free energy values are known:
      this is faster than calculating deltamfe from scratch as deltamfe can be 
      computed quickly from substracting precomputed mfe values if the structure does not change"""
   ph_old, ph_new = GPmap[tuple(sequence_indices)], GPmap[tuple(newsequence_indices)]
   if ph_old != ph_new:
      return get_deltamfe(sequence_indices, newsequence_indices)
   else:
      return mfearray[tuple(newsequence_indices)] - mfearray[tuple(sequence_indices)]   

def get_mfe_structure(sequence_indices):
   """get minimum free energy structure in dotbracket format for the sequence in integer format"""
   a = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
   (mfe_structure, mfe) = a.mfe()
   return mfe_structure

def get_mfe_structure_as_int(sequence_indices):
   """get minimum free energy structure in integer format for the sequence in integer format"""
   a = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
   (mfe_structure, mfe) = a.mfe()
   return dotbracket_to_int(mfe_structure)

###############################################################################################
## converting between sequence and structure representations
###############################################################################################
def dotbracket_to_binary(dotbracketstring):
   """translating each symbol in the dotbracket string a two-digit binary number;
   translation is defined such that the number of '1' digits is the number of paired positions;
   similar to Dingle, K., Camargo, C. Q. and Louis, A. A. Input-output maps are strongly biased towards simple outputs. Nature Communications 9, (2018)."""
   binstr = '1'
   for char in dotbracketstring:
      binstr = binstr + db_to_bin[char]
   return binstr

def dotbracket_to_int(dotbracketstring, check_size_limit=True):
   """convert dotbracket format to integer format:
   this is achieved by translating each symbol into a two-digit binary number
   and then converting this into a base-10 integer; a leading '1' is added, so that leading '0's in the binary string matter for the decimal
   if check_size_limit, the function tests whether the integer is within the range of the numpy uint32 datatype"""
   binstr = dotbracket_to_binary(dotbracketstring)
   if check_size_limit:
      assert len(binstr)<32
   integer_rep = int(binstr, 2)
   if check_size_limit:
      assert 0 < integer_rep < 4294967295 #limit of uint32
   return integer_rep

def sequence_str_to_int(sequence):
   """convert between sequence representations:
   from biological four-letter string to a tuple of integers from 0-3;
   motive: these integers can be used as array tuples"""
   return tuple([base_to_number[b] for b in sequence])

def sequence_int_to_str(sequence_indices_tuple):
   """convert between sequence representations:
   from tuple of integers from 0-3 to biological four-letter string;
   motive: these integers can be used as array tuples"""
   return ''.join([index_to_base[ind] for ind in sequence_indices_tuple])

###############################################################################################
## save deltadeltaG (both definitions) as array for full enumeration of sequences
###############################################################################################

def save_deltadeltaG_array_if_not_present(pos_and_new_nucl_index, L, K, GPmapdef, GPmap, energy_gap_array, deltaGBoltzarray, undefined_function):
   """function for saving full enumeration data;
   if an array is not present for deltadeltaG data for substitutions at a site and to a nucleotide given by pos_and_new_res_index;
   this is generated and saved"""
   (pos, new_nucl_type) = [(p, k) for p in range(L) for k in range(K)][pos_and_new_nucl_index]
   if not isfile('./thermodynamics_data/deltadeltaGarray_energy_gap_L'+str(L)+'pos'+str(pos)+'new'+str(new_nucl_type)+GPmapdef+'.npy'):
      deltadeltaGarray = np.zeros((K,)*L, dtype='float_')
      deltadeltaGarray_energygap = np.zeros((K,)*L, dtype='float_')
      for genotype, ph in np.ndenumerate(GPmap):
         if genotype[pos] != new_nucl_type and not undefined_function(ph):
            newsequence_indices = tuple([b if i != pos else new_nucl_type for i, b in enumerate(genotype)])
            deltadeltaGarray[tuple(genotype)] = get_deltadeltaG_Boltzmann_fast(tuple(genotype), tuple(newsequence_indices), GPmap, deltaGBoltzarray)
            deltadeltaGarray_energygap[tuple(genotype)] = get_deltadeltaG_energygap_fast(tuple(genotype), tuple(newsequence_indices), GPmap, energy_gap_array)
         else:
            deltadeltaGarray[tuple(genotype)] = np.nan
            deltadeltaGarray_energygap[tuple(genotype)] = np.nan
      np.save('./thermodynamics_data/deltadeltaGarrayBoltzmann_L'+str(L)+'pos'+str(pos)+'new'+str(new_nucl_type)+GPmapdef+'.npy', deltadeltaGarray, allow_pickle=False)
      np.save('./thermodynamics_data/deltadeltaGarray_energy_gap_L'+str(L)+'pos'+str(pos)+'new'+str(new_nucl_type)+GPmapdef+'.npy', deltadeltaGarray_energygap, allow_pickle=False)
   return None

def save_deltamfe_array_if_not_present(pos_and_new_nucl_index, L, K, GPmapdef, GPmap, mfearray, undefined_function):
   """function for saving full enumeration data;
   if an array is not present for deltamfe data for substitutions at a site and to a nucleotide given by pos_and_new_res_index;
   this is generated and saved"""
   (pos, new_nucl_type) = [(p, k) for p in range(L) for k in range(K)][pos_and_new_nucl_index]
   if not isfile('./thermodynamics_data/deltamfe_L'+str(L)+'pos'+str(pos)+'new'+str(new_nucl_type)+GPmapdef+'.npy'):
      deltamfearray = np.zeros((K,)*L, dtype='float_')
      for genotype, ph in np.ndenumerate(GPmap):
         if genotype[pos] != new_nucl_type and not undefined_function(ph):
            newsequence_indices = tuple([b if i != pos else new_nucl_type for i, b in enumerate(genotype)])
            deltamfearray[tuple(genotype)] = get_deltamfe_fast(tuple(genotype), tuple(newsequence_indices), GPmap, mfearray)
         else:
            deltamfearray[tuple(genotype)] = np.nan
      np.save('./thermodynamics_data/deltamfe_L'+str(L)+'pos'+str(pos)+'new'+str(new_nucl_type)+GPmapdef+'.npy', deltamfearray, allow_pickle=False)
   return None

###############################################################################################
## deltaG for an arbitrary (not necessarily mfe) structure given a sequence
###############################################################################################

def get_G_arbitrary_sequence_structure_pair(sequence, structure):
   """return free energy of folding the given sequence (integer representation)
   into the given structure dotbracket representation using fold_compound & eval_structure;
   returns np.inf if the sequence is not compatible with the structure's basepairing"""
   if sequence_compatible_with_basepairs(sequence_int_to_str(sequence), structure):
      a = RNA.fold_compound(sequence_int_to_str(sequence), model)
      G_in_given_structure = a.eval_structure(structure)
      return G_in_given_structure
   else:
      return np.inf

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   print 'test thermodynamic_functions.py'
   
   def get_Boltzmann_probabilities(sequence_indices, potential_structures_for_seq):
      """to ensure Boltzmann distribution is used correctly in ViennaRNA,
      we calculate the complete distribution for a full enumeration of secondary structures both manually, based on eval_structure
      and using ViennaRNA (this function)"""
      kbT = RNA.exp_param().kT/1000.0
      a = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
      (mfe_structure, mfe) = a.mfe()
      a.exp_params_rescale(mfe)
      (prob_vector, dG) = a.pf()
      Boltzmannp = []
      for s in potential_structures_for_seq: 
         if sequence_compatible_with_basepairs(sequence_int_to_str(sequence_indices), s):
            Boltzmannp.append(a.pr_structure(s))
         else:
            Boltzmannp.append(0.0)
      #print 'partition function computed ViennaRNA:', np.exp(-dG/kbT)
      return Boltzmannp

   def get_subopt_structures(sequence_indices):
      """to ensure Boltzmann distribution is used correctly in ViennaRNA,
      we calculate the complete distribution for a full enumeration of secondary structures both manually, based on eval_structure
      and using ViennaRNA (this function)"""
      a = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
      (mfe_structure, mfe) = a.mfe()
      a.exp_params_rescale(mfe)
      (prob_vector, dG) = a.pf()
      structure_vs_energy = {}
      a.subopt_cb(int(mfe * 2 * 100.0), convert_result_to_dict, structure_vs_energy)
      return structure_vs_energy.keys()

   def get_Boltzmann_probabilities_manually(sequence_indices, potential_structures_for_seq):
      """to ensure Boltzmann distribution is used correctly in ViennaRNA,
      we calculate the complete distribution for a full enumeration of secondary structures both manually (this function), based on eval_structure
      and using ViennaRNA"""
      a = RNA.fold_compound(sequence_int_to_str(sequence_indices), model)
      (mfe_structure, mfe) = a.mfe()
      a.exp_params_rescale(mfe)
      kbT = RNA.exp_param().kT/1000.0
      Boltzmann_factors = []
      for s in potential_structures_for_seq: 
         if sequence_compatible_with_basepairs(sequence_int_to_str(sequence_indices), s):
            Boltzmann_factors.append(np.exp(-a.eval_structure(s)/kbT))
         else:
            Boltzmann_factors.append(0.0)
      #print 'partition function computed manually:', np.sum(Boltzmann_factors)
      return np.divide(Boltzmann_factors, np.sum(Boltzmann_factors))

   ### setup   
   from rna_structural_functions import is_likely_to_be_valid_structure
   import functions_for_sampling_with_bp_swap as sample
   import neutral_component as NC
   from general_functions import get_dotbracket_from_int
   ### test energygap by brute force (enumerating all possible structures)
   seq_len = 13
   for test_no in range(10**3):
      seq = tuple(np.random.choice(np.arange(4), seq_len, replace=True))
      mfe_struct = get_mfe_structure(seq)
      if mfe_struct.count('(') > 0:
         energygap = get_DeltaG_energygap(seq)
         potential_structures = [s for s in sample.return_all_allowed_structures_starting_with('', seq_len, allow_isolated_bps=False) if is_likely_to_be_valid_structure(s, allow_isolated_bps=False)]
         potential_structures.append(''.join(['.',]*seq_len)) #unfolded structure
         energygap_by_full_structure_enumeration = min([get_G_arbitrary_sequence_structure_pair(seq, s) for s in potential_structures if s != mfe_struct]) - get_mfe(seq)
         assert abs(energygap_by_full_structure_enumeration - energygap) < 10.0**(-4)
         for seq2 in NC.neighbours_g(seq, K=3, L=seq_len):
            new_energygap_for_ddG = min([get_G_arbitrary_sequence_structure_pair(seq2, s) for s in potential_structures if s != mfe_struct]) - get_G_arbitrary_sequence_structure_pair(seq2, mfe_struct)
            deltadeltaG_by_full_enumeration = new_energygap_for_ddG - energygap_by_full_structure_enumeration
            deltadeltaG = get_deltadeltaG_energygap(seq, seq2)
            if not (np.isinf(deltadeltaG) and np.isinf(deltadeltaG_by_full_enumeration) and deltadeltaG*deltadeltaG_by_full_enumeration > 0) and not abs(deltadeltaG_by_full_enumeration - deltadeltaG) < 10.0**(-3):
               print seq, seq2, mfe_struct, get_mfe_structure(seq2), deltadeltaG, deltadeltaG_by_full_enumeration
            assert (np.isinf(deltadeltaG) and np.isinf(deltadeltaG_by_full_enumeration) and deltadeltaG*deltadeltaG_by_full_enumeration > 0) or abs(deltadeltaG_by_full_enumeration - deltadeltaG) < 10.0**(-3)
   ### test sequence and structure conversion
   seq_len = 13
   for test_no in range(10**4):
      seq = tuple(np.random.choice(np.arange(4), seq_len, replace=True))
      mfe_struct = get_mfe_structure(seq)   
      seq2 = sequence_str_to_int(sequence_int_to_str(seq))
      assert len([i for i in range(seq_len) if seq[i] == seq2[i]]) == seq_len
      assert get_dotbracket_from_int(dotbracket_to_int(mfe_struct)) == mfe_struct
   ### test get_G_arbitrary_sequence_structure_pair
   for test_no in range(100):
      seq = tuple(np.random.choice(np.arange(4), seq_len, replace=True))
      mfe_struct = get_mfe_structure(seq)
      if mfe_struct.count('(') > 0:
         assert get_G_arbitrary_sequence_structure_pair(seq, mfe_struct) - get_mfe(seq) < 10.0**(-3)
   ### test Boltzmann probability calculation
   for test_no in range(100):
      seq = tuple(np.random.choice(np.arange(4), seq_len, replace=True))
      mfe_struct = get_mfe_structure(seq)
      if mfe_struct.count('(') > 0:
         potential_structures = list(set([s for s in sample.return_all_allowed_structures_starting_with('', seq_len, allow_isolated_bps=True) if is_likely_to_be_valid_structure(s, allow_isolated_bps=True)]))
         potential_structures.append(''.join(['.']*seq_len))
         Boltzmann_prob_list_manual =  get_Boltzmann_probabilities_manually(seq, potential_structures)
         Boltzmann_prob_list_Vienna = get_Boltzmann_probabilities(seq, potential_structures)
         subopt_structures_ViennaRNA = get_subopt_structures(seq)
         #print '\ncheck ViennaRNA Boltzmann calculation for', sequence_int_to_str(seq)
         #print 'sum of manually computed Boltzmann-probabilities:', sum(Boltzmann_prob_list_manual)
         #print 'sum of ViennaRNA computed Boltzmann-probabilities:', sum(Boltzmann_prob_list_Vienna)
         for structureindex, structure in enumerate(potential_structures):
            assert abs(Boltzmann_prob_list_manual[structureindex]-Boltzmann_prob_list_Vienna[structureindex]) < 0.1
   print 'finished thermodynamic functions tests\n-------------\n\n'








