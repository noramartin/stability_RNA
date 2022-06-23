import GPfunctions.thermodynamic_functions as DG


def set_parameters_functions_for_thermodynamic_quantity(thermodynamic_quantity_input):
   assert thermodynamic_quantity_input in ['energygap', 'free_energy', 'Boltzmann_based']
   if thermodynamic_quantity_input == 'energygap':
      return DG.get_DeltaG_energygap, DG.get_deltadeltaG_energygap, 'mfe_unique'
   elif thermodynamic_quantity_input == 'Boltzmann_based':
      return DG.get_DeltaG_Boltzmann, DG.get_deltadeltaG_Boltzmann, 'mfe_unique_fifty'
   elif thermodynamic_quantity_input =='free_energy':
      return DG.get_DeltaG_energygap, DG.get_deltamfe, 'mfe_unique'

def GPmapdef_to_GPmapfilename(GPmapdef):
   assert GPmapdef in ['mfe', 'mfe_unique', 'mfe_unique_fifty']
   return './GPmapdata/GPmap_L'+str(L)+GPmapdef+'.npy'
###############################################################################################
# 'general' 
###############################################################################################
L, K = 13, 4
GPmap_filename = './GPmapdata/GPmap_L'+str(L)+'mfe.npy'
GPmap_unique_filename = './GPmapdata/GPmap_L'+str(L)+'mfe_unique.npy'
GPmap_uniquefifty_filename = './GPmapdata/GPmap_L'+str(L)+'mfe_unique_fifty.npy'
deltaGBoltzarray_filename = './thermodynamics_data/deltaGBoltzarray_L'+str(L)+'mfe.npy'
energy_gap_array_filename = './thermodynamics_data/energy_gap_array_L'+str(L)+'mfe.npy'
mfearray_filename = './thermodynamics_data/mfearray_L'+str(L)+'mfe.npy'
thermodynamic_quantity_vs_filename_array = {'energygap': energy_gap_array_filename,
                                            'Boltzmann_based': deltaGBoltzarray_filename, 
                                            'free_energy': mfearray_filename}