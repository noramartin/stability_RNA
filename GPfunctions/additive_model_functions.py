import numpy as np
import pandas as pd
from scipy.special import comb

def set_size_in_additive_model(list_impact_nosites_K, deltaG):
   """rcalculate the number of sites in an additive model:
   - deltaG is th stability of the most stabile sequence in the NC
   - list_impact_nosites_K is a list of tuples: 
   (mutational stability impact deltadeltaG, number of sites with this impact, number of possibilities per site) 
   base pairs are treated as a single site with 2-6 different possibilities per site, depending on whether base pair swaps are allowed or only point mutations
   """
   impact, no_sites, K_eff = list_impact_nosites_K[0]
   if impact > 0:
      number_mutations_allowed1 = min(int(abs(deltaG)/abs(impact)), no_sites)
   else:
      number_mutations_allowed1 = no_sites
   list_N = []
   for number_mutations1 in range(0, number_mutations_allowed1+1):
      factor1 = comb(no_sites, number_mutations1)*(K_eff-1)**number_mutations1
      deltaG_left = deltaG - number_mutations1 * impact
      if len(list_impact_nosites_K)>1:
         list_N.append(factor1*set_size_in_additive_model(list_impact_nosites_K[1:], deltaG_left))
      else:
         list_N.append(factor1)
   return np.sum(list_N)

def look_up_value_for_model(df_parameters_additive_model, category, bp_swaps_included, quantity):
   """ look up the column of the reuired quantity ('lower quartile', 'median', 'upper quartile') with our without base pair swaps
   in the table df_parameters_additive_model for the required category"""
   assert quantity in ['lower quartile', 'median', 'upper quartile']
   if bp_swaps_included:
      return df_parameters_additive_model.loc[df_parameters_additive_model['category']==category]['ddG '+quantity+' with bp swaps'].tolist()[0]
   else:
      return df_parameters_additive_model.loc[df_parameters_additive_model['category']==category]['ddG '+quantity].tolist()[0]

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   print 'test: additive_model_functions.py'
   ## single mutational impact
   list_impact_nosites_K = [(0.8, 5, 4), (8.5, 10, 3)]
   assert 4**5 == int(set_size_in_additive_model(list_impact_nosites_K, deltaG=5))
   assert 1 + (4-1) * 5 + (4-1)**2 * 5 * 4 /2 == int(set_size_in_additive_model(list_impact_nosites_K, deltaG=1.8)) # single and double mutantions possible for lower impact
   assert 4**5 + (3-1) * 10 * (1 + (4-1) * 5) == int(set_size_in_additive_model(list_impact_nosites_K, deltaG=10)) # all mutations possible for first part + single mutations second part with one of first part
   print '\n\n-------------\n\n'