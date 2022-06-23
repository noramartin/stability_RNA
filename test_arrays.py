import numpy as np
import os 
os.environ["PATH"]+=os.pathsep+'/usr/bin'
from GPfunctions.general_functions import *
import GPfunctions.thermodynamic_functions as DG
import pandas as pd
import GPfunctions.neutral_component as NC
import itertools
import parameters as param
###############################################################################################
###############################################################################################
print  'test a few examples from saved stability data as a quick check'
###############################################################################################
###############################################################################################
for thermodynamic_quantity in ['energygap', 'Boltzmann_based']:
   deltaG_function, deltadeltaG_function, GPmapdef = param.set_parameters_functions_for_thermodynamic_quantity(thermodynamic_quantity)   
   if thermodynamic_quantity =='energygap':
      deltadeltaGdata = {(pos, new_residue_type): np.load('./thermodynamics_data/deltadeltaGarray_energy_gap_L'+str(param.L)+'pos'+str(pos)+'new'+str(new_residue_type)+GPmapdef.split('_')[0]+'.npy') for new_residue_type in range(param.K) for pos in range(param.L)}      
   elif thermodynamic_quantity == 'Boltzmann_based':
      deltadeltaGdata = {(pos, new_residue_type): np.load('./thermodynamics_data/deltadeltaGarrayBoltzmann_L'+str(param.L)+'pos'+str(pos)+'new'+str(new_residue_type)+GPmapdef.split('_')[0]+'.npy') for new_residue_type in range(param.K) for pos in range(param.L)}
   deltaGarray = np.load(param.thermodynamic_quantity_vs_filename_array[thermodynamic_quantity])
   structure_invalid_test = isunfoldedstructure_or_isolated_bps
   GPmap = np.load(param.GPmapdef_to_GPmapfilename(GPmapdef))
   seq_vs_NCindex_array, NCindex_vs_NCsize = NC.find_all_NCs_parallel(GPmap, GPmapdef)
   deltaGdata = pd.read_csv('./thermodynamics_data/'+thermodynamic_quantity+'dataL'+str(param.L)+GPmapdef+'.csv')
   ###############################################################################################
   ###############################################################################################
   print  'test: numpy arrays'
   ###############################################################################################
   ###############################################################################################
   for test_no in range(1000):
      seq = tuple(np.random.choice(np.arange(4), param.L, replace=True))
      mfe_struct = DG.get_mfe_structure(seq)
      if '(' in mfe_struct:
         deltaG = deltaG_function(seq)
         deltaG_from_array = deltaGarray[seq]
         assert (deltaG < 10.0**(-5) and structure_invalid_test(GPmap[seq])) or mfe_struct == get_dotbracket_from_int(GPmap[seq])
         assert abs(deltaG - deltaG_from_array) < 10.0**(-3) or structure_invalid_test(mfe_structure_from_array)
         for seq2 in NC.neighbours_g(seq, param.K, param.L):
            different_sites = [i for i, c in enumerate(seq2) if c != seq[i]]
            assert len(different_sites) == 1
            pos, new_residue_type = different_sites[0], seq2[different_sites[0]]
            deltadeltaG_from_array = deltadeltaGdata[(pos, new_residue_type)][seq]
            deltadeltaG_calculated = deltadeltaG_function(seq, seq2)
            assert (np.isinf(deltadeltaG_from_array) and np.isinf(deltadeltaG_calculated) and deltadeltaG_calculated*deltadeltaG_from_array > 0) or abs(deltadeltaG_from_array - deltadeltaG_calculated) < 10.0**(-3)
   ###############################################################################################
   ###############################################################################################
   print  'test: csv file'
   ###############################################################################################
   ###############################################################################################
   if thermodynamic_quantity != 'mfe':
      deltaGdata_sample = deltaGdata.sample(n=1000, replace=True, random_state=1) # pick out random rows and test
      for rowindex, row in deltaGdata_sample.iterrows():
         sequence, structure, stability = DG.sequence_str_to_int(row['sequence']), row['structure'], row['stability']
         calculated_structure = DG.get_mfe_structure(sequence)
         calculated_stability = deltaG_function(sequence)
         assert calculated_structure == structure 
         assert abs(calculated_stability - stability) < 10.0**(-3)
###############################################################################################
###############################################################################################
print  'test deltadeltamfe files'
###############################################################################################
###############################################################################################
mfearray = np.load(param.mfearray_filename)
deltadeltaMFEdata = {(pos, new_letter): np.load('./thermodynamics_data/deltamfe_L'+str(param.L)+'pos'+str(pos)+'new'+str(new_letter)+GPmapdef.split('_')[0]+'.npy') for new_letter in range(param.K) for pos in range(param.L)}      
for test_no in range(1000):
   seq = tuple(np.random.choice(np.arange(4), param.L, replace=True))
   mfe_struct = DG.get_mfe_structure(seq)
   if '(' in mfe_struct:
      mfe = DG.get_mfe(seq)
      mfe_from_array = mfearray[seq]
      assert abs(mfe_from_array - mfe) < 10.0**(-3)
      for seq2 in NC.neighbours_g(seq, param.K, param.L):
         different_sites = [i for i, c in enumerate(seq2) if c != seq[i]]
         assert len(different_sites) == 1
         pos, new_residue_type = different_sites[0], seq2[different_sites[0]]
         deltadeltaG_from_array = deltadeltaMFEdata[(pos, new_residue_type)][seq]
         deltadeltaG_calculated = DG.get_deltamfe(seq, seq2)
         assert (np.isinf(deltadeltaG_from_array) and np.isinf(deltadeltaG_calculated) and deltadeltaG_calculated*deltadeltaG_from_array > 0) or abs(deltadeltaG_from_array - deltadeltaG_calculated) < 10.0**(-3) 




