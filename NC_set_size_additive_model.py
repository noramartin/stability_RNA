import numpy as np
import matplotlib
matplotlib.use('ps')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from GPfunctions.general_functions import *
import GPfunctions.thermodynamic_functions as DG
import pandas as pd
import GPfunctions.rna_structural_functions as cg
import GPfunctions.neutral_component as NC
from GPfunctions.functions_for_sampling_with_bp_swap import find_x_sequences_per_structure
from GPfunctions.matplotlib_plot_functions import scatter_plt_set_size_plot, add_colorbar_next_to_plot, dotbracketstring_to_color, dotbracketstring_to_marker,add_colorbar_above_plot
from GPfunctions.additive_model_functions import *
import parameters as param
import seaborn as sns
from scipy.stats import pearsonr

def get_ddG_by_NCindex_and_neutralset(thermodynamic_quantity, L, K, structure_list, NCindex_array, NCindex_list, GPmap):
   """load precomupted deltadeltaG arrays and record the lower/upper quartile for each structure and each NC"""
   if thermodynamic_quantity =='Boltzmann_based':
      deltadeltaGdata = {(pos, new_residue_type): np.load('./thermodynamics_data/deltadeltaGarrayBoltzmann_L'+str(L)+'pos'+str(pos)+'new'+str(new_residue_type)+GPmapdef.split('_')[0]+'.npy') for new_residue_type in range(K) for pos in range(L)}
   elif thermodynamic_quantity =='energygap':
      deltadeltaGdata = {(pos, new_residue_type): np.load('./thermodynamics_data/deltadeltaGarray_energy_gap_L'+str(L)+'pos'+str(pos)+'new'+str(new_residue_type)+GPmapdef.split('_')[0]+'.npy') for new_residue_type in range(K) for pos in range(L)}      
   structure_vs_deltadeltaG_list = {p: [] for p in structure_list}
   NC_vs_deltadeltaG_list = {NCindex: [] for NCindex in NCindex_list}
   for sequence, structure_int in np.ndenumerate(GPmap):
      if not structure_invalid_test(structure_int):
         NCindex = NCindex_array[sequence]
         for (pos, new_residue_type) in deltadeltaGdata:
            deltadeltaG = deltadeltaGdata[(pos, new_residue_type)][tuple(sequence)]
            if not np.isnan(deltadeltaG):
               structure_vs_deltadeltaG_list[structure_int].append(np.abs(deltadeltaG)) 
               NC_vs_deltadeltaG_list[NCindex].append(np.abs(deltadeltaG))       
   return [np.percentile(structure_vs_deltadeltaG_list[p], 25) for p in structure_list], [np.percentile(NC_vs_deltadeltaG_list[NCindex], 25) for NCindex in NCindex_list]


###############################################################################################
###############################################################################################
#print  'set parameters'
###############################################################################################
###############################################################################################
L, K = 13, 4
thermodynamic_quantity = 'Boltzmann_based'#'energygap' 
stability_function, deltadeltaG_function, GPmapdef = param.set_parameters_functions_for_thermodynamic_quantity(thermodynamic_quantity)   
structure_invalid_test = isunfoldedstructure_or_isolated_bps
number_sequences_per_structure = 100
type_str_vs_marker = {'stack': 'o', 'bulge': 'o', 'internal': 'o'} 
bp_vs_greyscale_blue = {N+2: color for N, color in enumerate(sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)(np.linspace(0.3, 1.0, 4)))}
bp_vs_greyscale_red = {4: 'r'}
### greyscale-compatible version
color_dict = {'stack': bp_vs_greyscale_blue, 'bulge': {4: bp_vs_greyscale_red[4]}, 'internal': {4: bp_vs_greyscale_red[4]}} 
marker_dict = {'stack': 'x', 'bulge': '_', 'internal': '|'} 
maker_size = [25, 100] 
markeredgewidth = 0.2

###############################################################################################
###############################################################################################
print  'get table for NC and neutral sets with size, mean stability, max stability etc.'
###############################################################################################
###############################################################################################
GPmap = np.load(param.GPmapdef_to_GPmapfilename(GPmapdef))
seq_vs_NCindex_array, NCindex_vs_NCsize = NC.find_all_NCs_parallel(GPmap, GPmapdef)
deltaGdata = pd.read_csv('./thermodynamics_data/'+thermodynamic_quantity+'dataL'+str(L)+GPmapdef+'.csv')
NC_vs_list_of_deltaG = {NCindex: [] for NCindex in deltaGdata['NC index']}
ph_vs_list_of_deltaG = {ph: [] for ph in deltaGdata['structure']} 
for rowindex, row in deltaGdata.iterrows():
   NC_vs_list_of_deltaG[row['NC index']].append(row['stability'])
   ph_vs_list_of_deltaG[row['structure']].append(row['stability'])
structure_list = [DG. dotbracket_to_int(structure) for structure in ph_vs_list_of_deltaG.keys()]
NCindex_list = NC_vs_list_of_deltaG.keys()
quartile_ddG_sets, quartile_ddG_NC = get_ddG_by_NCindex_and_neutralset(thermodynamic_quantity, L, K, structure_list, seq_vs_NCindex_array, NCindex_list, GPmap)
###############################################################################################
### for NCs
###############################################################################################
NCindex_list = NC_vs_list_of_deltaG.keys()
NCsize_list = [len(NC_vs_list_of_deltaG[NCindex]) for NCindex in NCindex_list]
structure_list_NC = [get_dotbracket_from_int(NC.get_structure_forNCindex(seq_vs_NCindex_array, GPmap, NCindex)) for NCindex in NCindex_list]
maxdeltaG_list_NC = [max(NC_vs_list_of_deltaG[NCindex]) for NCindex in NCindex_list]
meandeltaG_list_NC = [np.mean(NC_vs_list_of_deltaG[NCindex]) for NCindex in NCindex_list]
dfNC = pd.DataFrame().from_dict({'NC index': NCindex_list, 'size': NCsize_list, 'structure': structure_list_NC, 
                             'maximum dG': maxdeltaG_list_NC, 'mean dG': meandeltaG_list_NC, 'lower quartile of abs(ddG)': quartile_ddG_NC})
dfNC.to_csv('./GPmapdata/NC_sizes_stabilities'+str(L)+GPmapdef+thermodynamic_quantity+'.csv')
del dfNC, NCindex_list, NCsize_list, structure_list_NC, maxdeltaG_list_NC, meandeltaG_list_NC, quartile_ddG_NC
###############################################################################################
## for neutral sets
###############################################################################################
structure_list_sets = [get_dotbracket_from_int(p) for p in structure_list]
neutralsetsize_list = [len(ph_vs_list_of_deltaG[get_dotbracket_from_int(p)]) for p in structure_list]
maxdeltaG_list_sets = [max(ph_vs_list_of_deltaG[get_dotbracket_from_int(p)]) for p in structure_list]
meandeltaG_list_sets = [np.mean(ph_vs_list_of_deltaG[get_dotbracket_from_int(p)]) for p in structure_list]
dfsets = pd.DataFrame().from_dict({'structure integer': structure_list, 'size': neutralsetsize_list, 'structure': structure_list_sets, 
                             'maximum dG': maxdeltaG_list_sets, 'mean dG': meandeltaG_list_sets, 'lower quartile of abs(ddG)': quartile_ddG_sets})
dfsets.to_csv('./GPmapdata/neutralset_sizes_stabilities'+str(L)+GPmapdef+thermodynamic_quantity+'.csv')
del dfsets, structure_list, structure_list_sets, neutralsetsize_list, maxdeltaG_list_sets, meandeltaG_list_sets, quartile_ddG_sets

###############################################################################################
###############################################################################################
print  'get p-sampled deltadeltaG distribution'
###############################################################################################
###############################################################################################
GPmap = np.load('./GPmapdata/GPmap_L'+str(L)+GPmapdef+'.npy')
role_to_less_detail = {'hairpin loop': 'L', 'exterior loop': 'L', 'multiloop': 'L', 'bulge': 'B', 'stack': 'S', 'internal loop': 'B'}

list_of_deltadeltaG, list_of_mut_type, list_of_categories = [], [], []
for sample_seq in find_x_sequences_per_structure(GPmap, number_sequences_per_structure):
   secondarystructure = get_dotbracket_from_int(GPmap[tuple(sample_seq)])
   for newsequence_indices in NC.neighbours_g(sample_seq, 4, L):
      deltadeltaG = deltadeltaG_function(sample_seq, newsequence_indices) 
      pos = [i for i, g1 in enumerate(sample_seq) if g1 != newsequence_indices[i]][0]
      if not np.isnan(deltadeltaG) and not structure_invalid_test(GPmap[tuple(sample_seq)]) and not np.isinf(deltadeltaG): 
         role = cg.find_role_in_structure(pos, secondarystructure)[0]
         list_of_deltadeltaG.append(np.abs(deltadeltaG))
         list_of_mut_type.append('point')
         if secondarystructure[pos] != '.':
            list_of_categories.append(role_to_less_detail[role]+str(cg.find_len_of_stack(pos, secondarystructure)))
         else:
            list_of_categories.append(role_to_less_detail[role])
   for newsequence_indices in NC.bp_mutation_not_point_mutation(sample_seq, secondarystructure):
      deltadeltaG = deltadeltaG_function(sample_seq, newsequence_indices) 
      if not np.isnan(deltadeltaG) and not structure_invalid_test(GPmap[tuple(sample_seq)]): 
         pos = [i for i, g1 in enumerate(sample_seq) if g1 != newsequence_indices[i]][0]
         list_of_categories.append('S' + str(cg.find_len_of_stack(pos, secondarystructure)))
         list_of_deltadeltaG.append(np.abs(deltadeltaG))   
         list_of_mut_type.append('bp')
df_deltadeltaG = pd.DataFrame().from_dict({'mutation type': list_of_mut_type, 'deltadeltaG': list_of_deltadeltaG, 'category': list_of_categories})
df_deltadeltaG.to_csv('./thermodynamics_data/deltadeltaG_for_additive_model_L'+str(L)+GPmapdef+'sample'+str(number_sequences_per_structure)+'_bp_mut.csv')
del list_of_deltadeltaG, list_of_mut_type, list_of_categories
###############################################################################################
###############################################################################################
print  'get mean deltadeltaG for model'
###############################################################################################
###############################################################################################
df_deltadeltaG = pd.read_csv('./thermodynamics_data/deltadeltaG_for_additive_model_L'+str(L)+GPmapdef+'sample'+str(number_sequences_per_structure)+'_bp_mut.csv')
list_of_deltadeltaG  = df_deltadeltaG['deltadeltaG'].tolist()
list_of_categories = df_deltadeltaG['category'].tolist()
list_of_mut_type =  df_deltadeltaG['mutation type'].tolist()
set_of_categories = list(set(list_of_categories))
site_category_vs_deltadelatG_list_point_only = {category: [deltadeltaG for i, deltadeltaG in enumerate(list_of_deltadeltaG) if list_of_categories[i]==category and list_of_mut_type[i]=='point'] for category in set_of_categories}
site_category_vs_deltadelatG_list = {category: [deltadeltaG for i, deltadeltaG in enumerate(list_of_deltadeltaG) if list_of_categories[i]==category] for category in set_of_categories}
df_parameters_additive_model = pd.DataFrame().from_dict({'category': set_of_categories, 'ddG lower quartile': [np.percentile(site_category_vs_deltadelatG_list_point_only[c], 25) for c in set_of_categories],
                                           'ddG median': [np.median(site_category_vs_deltadelatG_list_point_only[c]) for c in set_of_categories],
                                           'ddG upper quartile': [np.percentile(site_category_vs_deltadelatG_list_point_only[c], 75) for c in set_of_categories],
                                           'ddG lower quartile with bp swaps': [np.percentile(site_category_vs_deltadelatG_list[c], 25) for c in set_of_categories],
                                           'ddG median with bp swaps': [np.median(site_category_vs_deltadelatG_list[c]) for c in set_of_categories],
                                           'ddG upper quartile with bp swaps': [np.percentile(site_category_vs_deltadelatG_list[c], 75) for c in set_of_categories]})
df_parameters_additive_model.to_csv('./thermodynamics_data/parameters_additive_modelL'+str(L)+GPmapdef+'sample'+str(number_sequences_per_structure)+'_bp_mut.csv')
del list_of_deltadeltaG, list_of_mut_type, list_of_categories, df_deltadeltaG, set_of_categories
nbp_sites_NC = 3
###############################################################################################
###############################################################################################
print 'plot'
###############################################################################################
###############################################################################################
dfNC = pd.read_csv('./GPmapdata/NC_sizes_stabilities'+str(L)+GPmapdef+thermodynamic_quantity+'.csv')
dfsets = pd.read_csv('./GPmapdata/neutralset_sizes_stabilities'+str(L)+GPmapdef+thermodynamic_quantity+'.csv')
df_parameters_additive_model = pd.read_csv('./thermodynamics_data/parameters_additive_modelL'+str(L)+GPmapdef+'sample'+str(number_sequences_per_structure)+'_bp_mut.csv')
f1, ax1 = plt.subplots(ncols=2, figsize=(8,3.5))
f2, ax2 = plt.subplots(ncols=2, nrows=2, figsize=(7,6))
# create legend - follow matplotlib manual for custom legends
legend_content = [Line2D([0], [0], ms=10, marker=marker_dict['stack'], mew=markeredgewidth*10, lw=0.0, markerfacecolor=c, markeredgecolor=c, label=str(number_basepairs)) for number_basepairs, c in color_dict['stack'].items()]+\
    [Line2D([0], [0], marker=marker_dict['bulge'], mew=markeredgewidth*10, lw=0.0, ms=10, markerfacecolor=c, markeredgecolor=c, label=str(number_basepairs)+' & bulge') for number_basepairs, c in color_dict['bulge'].items()]+\
    [Line2D([0], [0], marker=marker_dict['internal'], mew=markeredgewidth*10, lw=0.0, ms=10, markerfacecolor=c, markeredgecolor=c, label=str(number_basepairs)+' & internal loop') for number_basepairs, c in color_dict['internal'].items()]
cmap = plt.get_cmap('inferno')
ddG_list_nc = dfNC['lower quartile of abs(ddG)'].tolist()
norm_ddG_nc = matplotlib.colors.BoundaryNorm(np.arange(min(ddG_list_nc) - 0.05, max(ddG_list_nc) + 0.051, 0.1), cmap.N)                         
ticks_colorbar = np.arange(min(ddG_list_nc), max(ddG_list_nc) + 0.1, 0.5)
sc1 = scatter_plt_set_size_plot(list(dfNC['size'].tolist()), list(dfNC['mean'+' dG'].tolist()), c=list(dfNC['lower quartile of abs(ddG)'].tolist()), marker_list = ['o',],
                            xlabel='neutral component size', ylabel='mean'+r' $\Delta G$'+'\n(kcal/mol)', ax=ax1[ 0], s=15, cmap=cmap, norm=norm_ddG_nc)
add_colorbar_above_plot(ax1[0], sc1, label='25% quartile'+r' of $|\Delta \Delta G|$'+ '\n(kcal/mol)', ticks=ticks_colorbar)
#####
ddG_list_sets = dfsets['lower quartile of abs(ddG)'].tolist()
norm_ddG_sets = matplotlib.colors.BoundaryNorm(np.arange(min(ddG_list_sets) - 0.05, max(ddG_list_sets) + 0.051, 0.1), cmap.N)                         
ticks_colorbar = np.arange(min(ddG_list_sets), max(ddG_list_sets) + 0.1, 0.2)
sc2 = scatter_plt_set_size_plot(list(dfsets['size'].tolist()), list(dfsets['mean'+' dG'].tolist()), c=list(dfsets['lower quartile of abs(ddG)'].tolist()), marker_list = ['o',],
                            xlabel='neutral set size', ylabel='mean'+r' $\Delta G$'+'\n(kcal/mol)', ax=ax1[1], s=20, cmap=cmap, norm=norm_ddG_sets)   
add_colorbar_above_plot(ax1[1], sc2, label='25% quartile'+r' of $|\Delta \Delta G|$'+ '\n(kcal/mol)', ticks=ticks_colorbar)
color_basepairs_list = [dotbracketstring_to_color(p, color_dict) for p in dfNC['structure'].tolist()]
marker_list = [dotbracketstring_to_marker(p, marker_dict) for p in dfNC['structure'].tolist()]
scatter_plt_set_size_plot(list(dfNC['size'].tolist()), list(dfNC['maximum'+' dG'].tolist()), c=color_basepairs_list, marker_list=marker_list,
                            xlabel='neutral component size', ylabel='maximum'+r' $\Delta G$'+'\n(kcal/mol)', ax=ax2[0, 0], 
                            s=maker_size[0], markeredgewidth=markeredgewidth)
color_basepairs_list = [dotbracketstring_to_color(p, color_dict) for p in dfsets['structure'].tolist()]
marker_list = [dotbracketstring_to_marker(p, marker_dict) for p in dfsets['structure'].tolist()]
scatter_plt_set_size_plot(list(dfsets['size'].tolist()), list(dfsets['maximum'+' dG'].tolist()), c=color_basepairs_list, marker_list=marker_list,
                            xlabel='neutral set size', ylabel='maximum'+r' $\Delta G$'+'\n(kcal/mol)', ax=ax2[0, 1], 
                            s=maker_size[1], markeredgewidth=1)         
  
###############################################################################################
## additive model
###############################################################################################
for plotindex, type_set in enumerate(['neutral component', 'neutral set']):
   for number_basepairs in range(2, 6):
      for bulges in range(3):
         number_loop_sites = L - 2*number_basepairs - bulges
         if bulges:    
            deltaG_list = list(np.linspace(0, 2.5, num=500))
            stack_category = 'S'+str(number_basepairs//2)
            if number_basepairs < 4 or number_loop_sites < 3:
               continue
         else:
            stack_category = 'S'+str(number_basepairs)
            deltaG_list = list(np.linspace(0, 4.2, num=1000))
         if stack_category not in df_parameters_additive_model['category'].tolist():
               continue
         if type_set == 'neutral component':
            K_bp, bp_swaps_included = nbp_sites_NC, False # only point mutations and therefore one mutation possible per base pair (slight underestimate)
         elif type_set == 'neutral set':
            K_bp, bp_swaps_included = 6, True # base pair swaps allowed and therefore 6 possibilities per bp
         category_vs_number_sites = {stack_category: (number_basepairs, K_bp), 'L': (number_loop_sites, K), 'B': (bulges, K)}
         mutational_impact_list = []
         for category, (number_sites, K_eff) in category_vs_number_sites.items():
            for quantity in ['lower quartile', 'median', 'upper quartile']:
               if quantity == 'median':  
                  if number_sites-2*(number_sites//2) > 0:
                     ddG_param = look_up_value_for_model(df_parameters_additive_model, category, bp_swaps_included, quantity)
                     mutational_impact_list.append(tuple([ddG_param, number_sites-2*(number_sites//2), K_eff]))
               elif number_sites > 0:
                  ddG_param = look_up_value_for_model(df_parameters_additive_model, category, bp_swaps_included, quantity)
                  mutational_impact_list.append(tuple([ddG_param, number_sites//2, K_eff]))                  
         NCsize_list = [set_size_in_additive_model(mutational_impact_list, deltaG) for deltaG in deltaG_list]
         if not bulges:
            ax2[1, plotindex].plot(list(NCsize_list), list(np.abs(deltaG_list)), label=str(number_basepairs), c=color_dict['stack'][number_basepairs], rasterized=False)
            ax2[1, plotindex].scatter([N for listindex, N in enumerate(NCsize_list) if listindex%50 == 0], [np.abs(N) for listindex, N in enumerate(deltaG_list) if listindex%50 == 0], 
                                        c=color_dict['stack'][number_basepairs], marker=marker_dict['stack'], s=maker_size[1])
         else:
            ax2[1, plotindex].plot(list(NCsize_list), list(np.abs(deltaG_list)), label=str(number_basepairs)+'& bulge', c=color_dict['bulge'][number_basepairs], rasterized=False)
            ax2[1, plotindex].scatter([N for listindex, N in enumerate(NCsize_list) if listindex%50 == 0], [np.abs(N) for listindex, N in enumerate(deltaG_list) if listindex%50 == 0], 
                                         c=color_dict['bulge'][number_basepairs], marker=marker_dict['bulge'], s=maker_size[1])   
   ax2[1, plotindex].set_xscale('log')
   ax2[1, plotindex].set_xlabel(type_set+' size', fontsize=17)
   ax2[1, plotindex].set_ylabel(r'maximum $\Delta G$'+'\n(kcal/mol)', fontsize=17)
   ax2[1, plotindex].tick_params(axis='both', labelsize=15, length=7, width=1.5)
   ax2[1, plotindex].tick_params(axis='both', which='minor', length=3, width=1.2)
for i, axi in enumerate(ax1.reshape(-1)):
   axi.annotate('ABCDEFGH'[i], xy=(0.05, 0.87), xycoords='axes fraction', fontsize=25, fontweight='bold')
for i, axi in enumerate(ax2.reshape(-1)):
   axi.annotate('ABCDEFGH'[i], xy=(0.05, 0.87), xycoords='axes fraction', fontsize=25, fontweight='bold')
leg = f2.legend(handles = legend_content, title='number of base pairs', fontsize=16, bbox_to_anchor=(0.5, 1.0001), loc='lower center', ncol=3)#, title_fontsize=16)
leg.set_title('number of base pairs', prop={'size': 18})
f1.tight_layout()
f1.savefig('./figs_for_report/'+thermodynamic_quantity+'andNCsizeL'+str(L)+GPmapdef+'_columnformat_1.png', dpi=500, bbox_inches='tight')
f2.tight_layout()
f2.savefig('./figs_for_report/'+thermodynamic_quantity+'andNCsizeL'+str(L)+GPmapdef + 'K_inNCpairedsites' + str(nbp_sites_NC) + '_columnformat_2.png', dpi=500, bbox_inches='tight')
