import numpy as np
from copy import deepcopy
from GPfunctions.general_functions import *
import GPfunctions.thermodynamic_functions as DG
from os.path import isfile
from multiprocessing import Pool
import GPfunctions.neutral_component as NC
from functools import partial
from GPfunctions.rna_structural_functions import has_length_one_stack
import pandas as pd
import parameters as param

###################################################################################################
L, K = 13, 4
np.random.seed(5)
###################################################################################################
###############################################################################################
print 'sequence to mfe structure' 
###############################################################################################
if not isfile(param.GPmap_filename):
   GPmap =  np.zeros((K,)*L, dtype='uint32')
   sequence_list_withundefined = [tuple(sequence) for sequence, ph in np.ndenumerate(GPmap)]
   pool = Pool()
   mfestructurelist = pool.map(DG.get_mfe_structure_as_int, sequence_list_withundefined)
   for gindex, g in enumerate(sequence_list_withundefined):
      GPmap[g] = mfestructurelist[gindex]
   pool.close()
   pool.join()
   np.save(param.GPmap_filename, GPmap, allow_pickle=False)
   del mfestructurelist, GPmap
###############################################################################################
print 'deltaG - Boltzmann-based' 
###############################################################################################
if not isfile(param.deltaGBoltzarray_filename):
   GPmap = np.load(GPmap_filename)
   deltaGBoltzarray = np.zeros((param.K,) * param.L, dtype='float_')
   sequence_list = [tuple(sequence) for sequence, ph in np.ndenumerate(GPmap) if not isunfoldedstructure(ph)]
   pool = Pool()
   deltaGlist = pool.map(DG.get_DeltaG_Boltzmann, sequence_list)
   for gindex, g in enumerate(sequence_list):
      deltaGBoltzarray[g] = deltaGlist[gindex]
   pool.close()
   pool.join()
   np.save(param.deltaGBoltzarray_filename, deltaGBoltzarray, allow_pickle=False)
   del deltaGlist, deltaGBoltzarray
###############################################################################################
print 'energy gap' 
###############################################################################################
if not isfile(param.energy_gap_array_filename):
   energygaparray = np.zeros((param.K,) * param.L, dtype='float_')
   GPmap = np.load(GPmap_filename)
   sequence_list = [tuple(sequence) for sequence, ph in np.ndenumerate(GPmap) if not isunfoldedstructure(ph)] #if unfolded structure is predicted mfe structure, energy gap cannot always be computed (might be only one structure)
   pool = Pool()
   energygaplist = pool.map(DG.get_DeltaG_energygap, sequence_list)
   for gindex, g in enumerate(sequence_list):
      energygaparray[g] = energygaplist[gindex]
   pool.close()
   pool.join()
   np.save(param.energy_gap_array_filename, energygaparray, allow_pickle=False)
   del energygaplist, energygaparray
###############################################################################################
print 'mfe' 
###############################################################################################
if not isfile(param.mfearray_filename):
   mfearray = np.zeros((param.K,) * param.L, dtype='float_')
   pool = Pool()
   mfelist = pool.map(DG.get_mfe, sequence_list)
   for gindex, g in enumerate(sequence_list):
      mfearray[g] = mfelist[gindex]
   pool.close()
   pool.join()
   np.save(param.mfearray_filename, mfearray, allow_pickle=False)
   del mfearray, mfelist
###############################################################################################
print 'deltadeltaG' 
###############################################################################################
deltaGBoltzarray = np.load(param.deltaGBoltzarray_filename)
GPmap = np.load(param.GPmap_filename)
energy_gap_array = np.load(param.energy_gap_array_filename)

ddG_function = partial(DG.save_deltadeltaG_array_if_not_present, L=param.L, K=param.K, GPmapdef='mfe', 
                                            GPmap=GPmap, energy_gap_array=energy_gap_array, deltaGBoltzarray=deltaGBoltzarray, undefined_function=isunfoldedstructure)    

pool = Pool()
ddGlist = pool.map(ddG_function, list(range(L*K)))
pool.close()
pool.join()

###############################################################################################
print 'deltamfe' 
###############################################################################################
GPmap = np.load(param.GPmap_filename)
mfearray = np.load(param.mfearray_filename)

dmfe_function = partial(DG.save_deltamfe_array_if_not_present, L=param.L, K=param.K, GPmapdef='mfe', 
                                            GPmap=GPmap, mfearray=mfearray, undefined_function=isunfoldedstructure)    
pool = Pool()
ddGlist = pool.map(dmfe_function, list(range(L*K)))
pool.close()
pool.join()
###############################################################################################
print 'GP map with energygap > 0'
###############################################################################################
energy_gap_array = np.load(param.energy_gap_array_filename)
GPmap = np.load(param.GPmap_filename)
unique_GPmap = np.zeros((param.K,) * param.L, dtype='uint32')
structure_int_for_unstable = 2
assert isunfoldedstructure(structure_int_for_unstable)
for sequence, ph in np.ndenumerate(GPmap):
   if abs(energy_gap_array[tuple(sequence)]) > 10.0**(-6):
      unique_GPmap[tuple(sequence)] = ph
   else:
      unique_GPmap[tuple(sequence)] = structure_int_for_unstable #basically an undefined structure
np.save(param.GPmap_unique_filename, unique_GPmap, allow_pickle=False)
###############################################################################################
print 'GP map with Boltzmann-deltaG > 0'
###############################################################################################
deltaGBoltzarray = np.load(param.deltaGBoltzarray_filename)
GPmap = np.load(param.GPmap_filename)
uniquefifty_GPmap = np.zeros((param.K,) * param.L, dtype='uint32')
structure_int_for_unstable = 2
assert isunfoldedstructure(structure_int_for_unstable)
for sequence, ph in np.ndenumerate(GPmap):
   if deltaGBoltzarray[tuple(sequence)] > 0:
      uniquefifty_GPmap[tuple(sequence)] = ph
   else:
      uniquefifty_GPmap[tuple(sequence)] = structure_int_for_unstable #basically an undefined structure
np.save(param.GPmap_uniquefifty_filename, uniquefifty_GPmap, allow_pickle=False)
uniquefifty_GPmap = np.load(param.GPmap_uniquefifty_filename)
###############################################################################################
print 'neutral components'
###############################################################################################
mfearray = np.load(param.mfearray_filename)
deltaGBoltzarray = np.load(param.deltaGBoltzarray_filename)
energy_gap_array = np.load(param.energy_gap_array_filename)
uniquefifty_GPmap = np.load(param.GPmap_uniquefifty_filename)
unique_GPmap = np.load(param.GPmap_unique_filename)
for GPmapdef, GPmap_forNC in [('mfe_unique', unique_GPmap), ('mfe', unique_GPmap), ('mfe_unique_fifty', uniquefifty_GPmap)]:
   sequence_vs_NCindex_array, NCindex_vs_NCsize = NC.find_all_NCs_parallel(GPmap_forNC, GPmapdef)
###############################################################################################
print 'save data as csv'
###############################################################################################
mfearray = np.load(param.mfearray_filename)
deltaGBoltzarray = np.load(param.deltaGBoltzarray_filename)
energy_gap_array = np.load(param.energy_gap_array_filename)
uniquefifty_GPmap = np.load(param.GPmap_uniquefifty_filename)
unique_GPmap = np.load(param.GPmap_unique_filename)
for GPmapdef, GPmap_foranalysis, type_thermo_function, stabilityarray in [('mfe_unique', unique_GPmap, 'energygap', energy_gap_array), ('mfe_unique_fifty', uniquefifty_GPmap, 'Boltzmann_based', deltaGBoltzarray)]:
   csv_filename_summary = './thermodynamics_data/'+type_thermo_function+'dataL'+str(param.L)+GPmapdef+'.csv'
   if not isfile(csv_filename_summary):
      sequence_vs_NCindex_array, NCindex_vs_NCsize = NC.find_all_NCs_parallel(GPmap_foranalysis, GPmapdef)
      ph_vs_f = get_frequencies_in_array(GPmap_foranalysis, ignore_undefined=True, structure_invalid_test=isunfoldedstructure)
      sequence_list_df, NCindex_list_df, ph_list_df, stability_list_df, neutralsetsize_list_df, NCsize_list_df, mfe_list_df = [], [], [], [], [], [], []
      for sequence, ph in np.ndenumerate(GPmap_foranalysis):
         if not isunfoldedstructure_or_isolated_bps(ph): #zero if unfolded structure
            sequence_list_df.append(DG.sequence_int_to_str(tuple(sequence)))
            ph_list_df.append(get_dotbracket_from_int(ph))
            NCindex_list_df.append(sequence_vs_NCindex_array[tuple(sequence)])
            NCsize_list_df.append(NCindex_vs_NCsize[sequence_vs_NCindex_array[tuple(sequence)]])
            stability_list_df.append(stabilityarray[tuple(sequence)])
            mfe_list_df.append(mfearray[tuple(sequence)])
            neutralsetsize_list_df.append(ph_vs_f[ph])
            assert not has_length_one_stack(get_dotbracket_from_int(ph))
      deltaGdata = pd.DataFrame.from_dict({'sequence': sequence_list_df, 'NC index': NCindex_list_df, 'structure': ph_list_df, 'stability': stability_list_df, 
                              'neutral set size': neutralsetsize_list_df, 'NC size': NCsize_list_df, 'mfe': mfe_list_df})
      deltaGdata.to_csv(csv_filename_summary)
      del sequence_list_df, NCindex_list_df, ph_list_df, stability_list_df, neutralsetsize_list_df, NCsize_list_df, mfe_list_df
