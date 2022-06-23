import numpy as np
from neutral_component import neighbours_g_given_site
import random


allowed_bps = [(0,2), (2,0), (3, 1), (1, 3), (2, 3), (3, 2)]
########################################################################################################################################################################
## find fraction of each type of epistasis
########################################################################################################################################################################
def Hamming_dist(g1, g2):
   """return Hamming distance between g1 and g2"""
   return len([site for site in range(len(g1)) if g1[site]!=g2[site]])

def epistasis_present(s_AB, s_ab, s_Ab, s_aB, resolution):
   """test if the four values for single and double mutants 
   show epistasis of at the given resolution (0.1 for ViennaRNA data)"""
   if abs(s_AB + s_ab - s_Ab - s_aB) > resolution*0.99:
      return True
   else:
      return False


def create_double_mutant(g1, g2, g3, site_mut1, site_mut2):
   """return double mutant for g1, where site_mut1 takes the same value as g2 and site_mut3 as g3
   g2 has to differ from g1 only at site_mut1 and g3 from g1 only at site_mut2"""
   assert Hamming_dist(g1, g2) == 1 and Hamming_dist(g1, g3) == 1 and Hamming_dist(g2, g3) == 2 and g1[site_mut1] != g2[site_mut1] and g1[site_mut2] != g3[site_mut2]
   g4 = [c for c in g1]
   g4[site_mut1] = g2[site_mut1]
   g4[site_mut2] = g3[site_mut2]
   assert Hamming_dist(g1, g4) == 2  and Hamming_dist(g2, g4) == 1 and Hamming_dist(g3, g4) == 1
   return tuple(g4)


def total_fraction_of_epistasis_in_NC(NCindex, seq_vs_NCindex, deltaGarray, ignore_zero_ddG, resolution=0.1, shuffle_landscape=False):
   """for the specified NC, find pairs of substitutions where the double mutant is also contained in the NC;
   for these substitution pairs, count the fraction for which there is pairwise epistasis
   - seq_vs_NCindex maps sequences to NCindex
   - deltaGarray maps sequences to energy values
   - ignore_zero_ddG controls if substitutions with zero energy effect are included
   - resolution of the energy values
   - shuffle_landscape if the energy values should be shuffled within the NC (null model)
   following the analysis in Aguilar-Rodriguez, J., Payne, J. L. & Wagner, A. A thousand empirical adaptive landscapes and their navigability. Nat Ecol Evol 1, 0045 (2017),
   but here different types of epistasis are not distinguished
   """
   if shuffle_landscape:
      energyarray_to_use = shuffle_energylandscape_NC(NCindex, seq_vs_NCindex, deltaGarray)
   else:
      energyarray_to_use = deltaGarray   
   count_epi, count_total = 0, 0
   for g1, g2, g3, g4 in generator_for_each_mutation_pair_in_NC(NCindex, seq_vs_NCindex):
      s_ab, s_Ab, s_aB, s_AB = energyarray_to_use[g1], energyarray_to_use[g2], energyarray_to_use[g3], energyarray_to_use[g4]
      if (abs(s_Ab - s_ab) > 0.99*resolution and abs(s_aB - s_ab) > 0.99*resolution) or not ignore_zero_ddG:
         count_total += 1.0
         if epistasis_present(s_AB, s_ab, s_Ab, s_aB, resolution):
            count_epi += 1
   if count_total > 0:
      return count_epi/count_total
   else:
      return np.nan

def shuffle_energylandscape_NC(NCindex, seq_vs_NCindex, energyarray):
   """shuffle the values in energyarray for all indices, for which seq_vs_NCindex equals NCindex"""
   K, L = seq_vs_NCindex.shape[1], seq_vs_NCindex.ndim
   seqlist_NC_array = np.argwhere(seq_vs_NCindex == NCindex)
   neutral_component_list_of_seq = [tuple(seqlist_NC_array[i2]) for i2 in range(len(seqlist_NC_array))]
   neutral_component_list_of_deltaG = [energyarray[g] for g in neutral_component_list_of_seq]
   shuffled_energyarray = np.zeros((K,)*L, dtype='float_')
   random.shuffle(neutral_component_list_of_deltaG)
   for g_index, g in enumerate(neutral_component_list_of_seq):
      shuffled_energyarray[g] = neutral_component_list_of_deltaG[g_index]
   return shuffled_energyarray



def generator_for_each_mutation_pair_in_NC(NCindex, seq_vs_NCindex):
   """generator for sequence, pair of substitutions and the corresponding double mutant
   if all four of these sequences are part of the specified NC"""
   K, L = seq_vs_NCindex.shape[1], seq_vs_NCindex.ndim
   seqlist_NC_array = np.argwhere(seq_vs_NCindex == NCindex)
   for i in range(len(seqlist_NC_array)):
      g1 = tuple(seqlist_NC_array[i])
      for site_mut1 in range(L):
         for site_mut2 in range(site_mut1): 
            for g2 in neighbours_g_given_site(g1, K, L, site_mut1):
               for g3 in neighbours_g_given_site(g1, K, L, site_mut2):
                  if seq_vs_NCindex[g2] == NCindex and seq_vs_NCindex[g3] == NCindex:
                     g4 = create_double_mutant(g1, g2, g3, site_mut1, site_mut2)
                     if seq_vs_NCindex[g4] == NCindex:
                        yield g1, g2, g3, g4


############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   print 'test: epistasis.py'
   ####
   print 'Hamming distance'
   ####
   g1 = (1, 0, 2, 0, 1 , 2, 3)
   g2 = (3, 0, 2, 0, 1 , 2, 3)
   g3 = (0, 1, 2, 0, 1 , 2, 2)
   assert Hamming_dist(g1, g3) == 3
   assert Hamming_dist(g1, g2) == 1
   assert Hamming_dist(g2, g3) == 3
   ####
   print 'Epistasis function'
   ####
   assert not epistasis_present(s_AB = 1.0, s_ab=0.5, s_Ab=0.8, s_aB=0.7, resolution=0.1) 
   assert epistasis_present(s_AB = 1.0, s_ab=0.5, s_Ab=0.9, s_aB=0.7, resolution=0.1) 
   assert not epistasis_present(s_AB = 2.501, s_ab=-0.2, s_Ab=0.0, s_aB=2.3, resolution=0.1)
   assert epistasis_present(s_AB = 2.501, s_ab=-0.2, s_Ab=0.0, s_aB=2.4, resolution=0.1) 
   assert epistasis_present(s_AB = 2.501, s_ab=-0.2, s_Ab=0.0, s_aB=2.2, resolution=0.1)
   assert epistasis_present(s_AB = 2.501, s_ab=-0.2, s_Ab=0.1, s_aB=2.3, resolution=0.1) 
   assert epistasis_present(s_AB = 2.501, s_ab=-0.2, s_Ab=-0.1, s_aB=2.3, resolution=0.1)
   ####
   print 'double mutant'
   ####
   g3 = (1, 0, 2, 0, 1 , 2, 2)
   g4 = create_double_mutant(g1, g2, g3, 0, 6)
   g4_test = (3, 0, 2, 0, 1 , 2, 2)
   assert len([i for i, g4_i in enumerate(g4) if g4_i != g4_test[i]]) == 0
   ####
   print  'epistasis: toy examples'
   ####
   def create_epistasis_testcase(seq_vs_deltaG_in_NC, K=4):
      """ build test case arrays to test epistasis calculation"""
      L = len(seq_vs_deltaG_in_NC.keys()[0])
      NCindex, seq_vs_NCindex, deltaGarray = 1, np.random.randint(low=2, high=40, size=(K,)*L, dtype='uint32'), 4*np.random.random_sample((K,)*L)
      for seq, deltaG in seq_vs_deltaG_in_NC.iteritems():
         seq_vs_NCindex[seq] = 1
         deltaGarray[seq] = deltaG
      return NCindex, seq_vs_NCindex, deltaGarray
   seq_vs_deltaG_in_NC = {(1, 0, 1): 2.0, (1, 1, 1): 2.5, (1, 0, 2): 1.8, (3, 1, 1): 2.5, (1, 1, 2): 2.3, (3, 1, 2): 2.3}
   NCindex, seq_vs_NCindex, deltaGarray = create_epistasis_testcase(seq_vs_deltaG_in_NC, K=4)
   assert abs(total_fraction_of_epistasis_in_NC(NCindex, seq_vs_NCindex, deltaGarray, ignore_zero_ddG=False, resolution=0.1, shuffle_landscape=False) - 0.0) < 0.0001
   assert abs(total_fraction_of_epistasis_in_NC(NCindex, seq_vs_NCindex, deltaGarray, ignore_zero_ddG=True, resolution=0.1, shuffle_landscape=False) - 0.0) < 0.0001
   seq_vs_deltaG_in_NC = {(1, 0, 1): 2.0, (1, 1, 1): 2.5, (1, 0, 2): 1.8, (3, 1, 1): 2.5, (1, 1, 2): 2.3, (3, 1, 2): 2.4}
   NCindex, seq_vs_NCindex, deltaGarray = create_epistasis_testcase(seq_vs_deltaG_in_NC, K=4)
   assert abs(total_fraction_of_epistasis_in_NC(NCindex, seq_vs_NCindex, deltaGarray, ignore_zero_ddG=False, resolution=0.1, shuffle_landscape=False) - 4.0/8) < 0.0001
   assert abs(total_fraction_of_epistasis_in_NC(NCindex, seq_vs_NCindex, deltaGarray, ignore_zero_ddG=True, resolution=0.1, shuffle_landscape=False) - 2.0/6) < 0.0001 
   seq_vs_deltaG_in_NC = {(1, 0, 1): 2.0, (1, 1, 1): 2.0, (1, 0, 2): 2.0, (3, 1, 1): 2.0, (1, 1, 2): 2.0, (3, 1, 2): 2.0}
   NCindex, seq_vs_NCindex, deltaGarray = create_epistasis_testcase(seq_vs_deltaG_in_NC, K=4)
   assert abs(total_fraction_of_epistasis_in_NC(NCindex, seq_vs_NCindex, deltaGarray, ignore_zero_ddG=False, resolution=0.1, shuffle_landscape=True) - 0.0) < 0.0001
   assert np.isnan(total_fraction_of_epistasis_in_NC(NCindex, seq_vs_NCindex, deltaGarray, ignore_zero_ddG=True, resolution=0.1, shuffle_landscape=True))





