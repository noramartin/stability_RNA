import numpy as np
import os 
os.environ["PATH"]+=os.pathsep+'/usr/bin'
import matplotlib
matplotlib.use('ps')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from GPfunctions.general_functions import *
import GPfunctions.thermodynamic_functions as DG
import pandas as pd
import seaborn as sns
import GPfunctions.rna_structural_functions as cg
import GPfunctions.neutral_component as NC
from GPfunctions.functions_for_sampling_with_bp_swap import find_x_sequences_per_structure
import networkx as nx
import itertools
from GPfunctions.matplotlib_plot_functions import *
from matplotlib.lines import Line2D
import parameters as param
from copy import deepcopy

def ddGboxplot_stacks_by_element_length(df, filename, structural_categogies, Nmin = 20):
  """take dataframe with mutational impacts and their structural roles and plot
   - large boxplot with mutational impacts and structural roles
   - smaller plots underneath for each category: length of structural element vs deltadeltaG
     (plot median for each group and errorbars to lower/upper quartiles)
   - Nmin is the minimum naumber of ddG values required in the sample to plot one errorbar"""
  f, ax0 = plt.subplots(figsize=(6, 3.6)) #constrained_layout=True, 
  color_palette ={'middle': 'yellowgreen', 'end': 'tab:blue', 'fork': 'r'} 
  structural_categogies2 = deepcopy(structural_categogies)
  structural_categogies2.remove('stack')
  structural_categogies2 += ['stack\n'+r'($2$ bp)', 'stack\n'+r'($\leq 4$ bp)', 'stack\n'+r'($\geq 5$ bp)']
  df2_dict = {'structural role': [], 'deltadeltaG': [], 'position in structural segment': []}
  print 'compile second df'
  for structural_role, element_length, ddG, pos_in_element in zip(df['structural role'].tolist(), df['element length'].tolist(), df['deltadeltaG'].tolist(), df['position in structural segment'].tolist()):
    if structural_role == 'stack':
      if int(element_length) <= 2:
         df2_dict['structural role'].append('stack\n'+r'($2$ bp)')
      elif int(element_length) <= 4:
         df2_dict['structural role'].append('stack\n'+r'($\leq 4$ bp)')
      else:
         df2_dict['structural role'].append('stack\n'+r'($\geq 5$ bp)')         
    else:
      df2_dict['structural role'].append(structural_role)
    df2_dict['deltadeltaG'].append(ddG)
    df2_dict['position in structural segment'].append(pos_in_element)
  df2 = pd.DataFrame.from_dict(df2_dict)
  print sorted(list(set(df2_dict['structural role'])))
  sns.boxplot(data=df2, x='structural role', y='deltadeltaG', ax=ax0, hue='position in structural segment', 
               order=structural_categogies2, palette=color_palette, whis=(5,95), fliersize=2, boxprops = {'linewidth': '0.2'})
  for lineindex, lineboxplot in enumerate(ax0.lines):
      if (lineindex - 4)%6 == 0:
         lineboxplot.set_color('r')
  ax0.set_ylabel('stability change after substitution\n'+r'$|\Delta \Delta G|$ (kcal/mol)')
  ax0.set_xlabel('')
  legend = ax0.get_legend()
  legend.set_title('')
  legend.set_bbox_to_anchor(bbox=(0.8, 0.82))
  #labels = ax0.get_xticklabels()
  ax0.set_xticklabels([l if 'loop' not in str(l) else '\n'.join(str(l).split()) for l in structural_categogies2])
  f.tight_layout()
  f.savefig(filename, dpi=300, bbox_inches='tight')
  f.savefig(filename[:-4] + '.eps', dpi=300, bbox_inches='tight')
  plt.close('all')


L, K = 13, 4
structure_invalid_test = isunfoldedstructure_or_isolated_bps
L_long_seq, sample_size = 30, 10**7 # for sampling of sequences of longer length

for thermodynamic_quantity in ['energygap', 'free_energy', 'Boltzmann_based']:
   stability_function, deltadeltaG_function, GPmapdef = param.set_parameters_functions_for_thermodynamic_quantity(thermodynamic_quantity)   
   if thermodynamic_quantity =='energygap' or thermodynamic_quantity == 'free_energy':
      round_ddG = True
   elif thermodynamic_quantity == 'Boltzmann_based':
      round_ddG = False
   ###################################################################################################
   ###############################################################################################
   ###############################################################################################
   print  '\n--deltadeltaG by sampling for long sequences', thermodynamic_quantity
   ###############################################################################################
   ###############################################################################################
   categories_detailed = ['stack (middle)', 'stack (end)', 'bulge (middle)', 'bulge (end)', 'internal loop (middle)', 'internal loop (end)', 
               'hairpin loop (middle)', 'hairpin loop (end)', 'multiloop (middle)', 'multiloop (end)',  'exterior loop (middle)', 'exterior loop (end)']
   categories = ['stack', 'bulge', 'internal loop', 'hairpin loop', 'multiloop', 'exterior loop']
   csv_filename = './thermodynamics_data/deltadeltaGdata_and_role_in_structureL'+str(L_long_seq)+GPmapdef+thermodynamic_quantity+'sample'+str(sample_size)+'.csv'
   if not isfile(csv_filename):
      list_of_sequences, mutation_list = [], []
      deltadeltaG_list, list_of_roles_in_structure, list_of_element_length, role_only_list, rel_to_structural_element_list = [], [], [], [], []
      while len(deltadeltaG_list) < sample_size:
         sample_sequence = tuple(np.random.randint(0, K, size=L_long_seq))
         secondarystructure = DG.get_mfe_structure(sample_sequence)
         if '(' in secondarystructure and abs(stability_function(sample_sequence)) > 0.001 and not cg.has_length_one_stack(secondarystructure): #stable and not unfolded and no isolated bas pairs
            for pos in range(L_long_seq):
               role, rel_to_structural_element = cg.find_role_in_structure(pos, secondarystructure)
               length_structural_element = cg.find_element_length(pos, secondarystructure)
               for alternive_base in range(K):
                  if alternive_base != sample_sequence[pos]:
                     newsequence_indices = tuple([c if p != pos else alternive_base for p, c in enumerate(sample_sequence)])
                     deltadeltaG = deltadeltaG_function(sample_sequence, newsequence_indices)
                     if not np.isnan(deltadeltaG) and not np.isinf(deltadeltaG):              
                        deltadeltaG_list.append(abs(deltadeltaG))
                        list_of_roles_in_structure.append(role +' (' + rel_to_structural_element + ')')
                        rel_to_structural_element_list.append(rel_to_structural_element)
                        role_only_list.append(role)
                        list_of_element_length.append(length_structural_element)
                        list_of_sequences.append(DG.sequence_int_to_str(sample_sequence))
                        mutation_list.append(str(pos) + DG.sequence_int_to_str([alternive_base,])[0])
            del role, rel_to_structural_element, alternive_base, pos
         del sample_sequence, secondarystructure
      df_deltadeltaG_long_seq = pd.DataFrame().from_dict({'role in structure': list_of_roles_in_structure, 'deltadeltaG': deltadeltaG_list, 
        'element length': list_of_element_length, 'structural role': role_only_list, 'position in structural segment': rel_to_structural_element_list, 'sequence': list_of_sequences, 'mutation': mutation_list})
      df_deltadeltaG_long_seq.to_csv(csv_filename)
      del deltadeltaG_list, list_of_roles_in_structure, list_of_element_length, role_only_list, rel_to_structural_element_list, list_of_sequences, mutation_list 
   df_deltadeltaG_long_seq = pd.read_csv(csv_filename)
   if df_deltadeltaG_long_seq['role in structure'].tolist().count('multiloop (middle)') < 10 or df_deltadeltaG_long_seq['role in structure'].tolist().count('multiloop (end)') < 10:
      categories_detailed.remove('multiloop (middle)')
      categories_detailed.remove('multiloop (end)')
      categories.remove('multiloop')
   ddGboxplot_stacks_by_element_length(df_deltadeltaG_long_seq, structural_categogies=categories,
                           filename='./figs_for_report/deltadeltaG_structural_role_and_length'+thermodynamic_quantity+'L'+str(L_long_seq)+'sample'+str(int(np.log10(sample_size)))+GPmapdef+'_stacksonly.png')
   ddGboxplot_and_by_element_length(df_deltadeltaG_long_seq, structural_categogies=categories,
                           filename='./figs_for_report/deltadeltaG_structural_role_and_length'+thermodynamic_quantity+'L'+str(L_long_seq)+'sample'+str(int(np.log10(sample_size)))+GPmapdef+'.png')
   save_percentiles_for_category_to_csv(df_deltadeltaG_long_seq, percentiles=[5, 25, 50, 75, 95], structural_categogies=categories_detailed, 
       filename='./thermodynamics_data/summary_deltadeltaGdata_and_role_in_structureL'+str(L_long_seq)+GPmapdef+thermodynamic_quantity+'sample'+str(sample_size)+'.csv', 
       include_length=False, round_ddG=round_ddG)
   save_percentiles_for_category_to_csv(df_deltadeltaG_long_seq, percentiles=[5, 25, 50, 75, 95], structural_categogies=categories_detailed, 
       filename='./thermodynamics_data/summary_deltadeltaGdata_and_role_in_structureL'+str(L_long_seq)+GPmapdef+thermodynamic_quantity+'sample'+str(sample_size)+'_elementlength.csv', 
       include_length=True, round_ddG=round_ddG)
   del df_deltadeltaG_long_seq

   ###############################################################################################
   ###############################################################################################
   print  'deltadeltaG depending on role in structure for full enumeration of L='+str(L)
   ###############################################################################################
   ###############################################################################################
   categories_detailed = ['hairpin loop (end)', 'hairpin loop (middle)', 'stack (end)', 'stack (middle)', 'bulge (end)', 'internal loop (end)', 
                         'exterior loop (end)', 'exterior loop (middle)']
   categories = ['hairpin loop', 'stack', 'bulge', 'internal loop', 'exterior loop']
   csv_filename = './thermodynamics_data/deltadeltaGdata_and_role_in_structureL'+str(L)+GPmapdef+thermodynamic_quantity+'.csv'
   if not isfile(csv_filename):
      GPmap = np.load(param.GPmapdef_to_GPmapfilename(GPmapdef))
      if GPmapdef == 'mfe_unique':
         stabilityarray = np.load(param.thermodynamic_quantity_vs_filename_array['energygap'])
      elif GPmapdef == 'mfe_unique_fifty':
         stabilityarray = np.load(param.thermodynamic_quantity_vs_filename_array['Boltzmann_based'])
      if thermodynamic_quantity =='Boltzmann_based':
          deltadeltaGdata = {(pos, new_letter): np.load('./thermodynamics_data/deltadeltaGarrayBoltzmann_L'+str(L)+'pos'+str(pos)+'new'+str(new_letter)+GPmapdef.split('_')[0]+'.npy') for new_letter in range(K) for pos in range(L)}
      elif thermodynamic_quantity =='energygap':
          deltadeltaGdata = {(pos, new_letter): np.load('./thermodynamics_data/deltadeltaGarray_energy_gap_L'+str(L)+'pos'+str(pos)+'new'+str(new_letter)+GPmapdef.split('_')[0]+'.npy') for new_letter in range(K) for pos in range(L)}      
      elif thermodynamic_quantity == 'free_energy':
          deltadeltaGdata = {(pos, new_letter): np.load('./thermodynamics_data/deltamfe_L'+str(L)+'pos'+str(pos)+'new'+str(new_letter)+GPmapdef.split('_')[0]+'.npy') for new_letter in range(K) for pos in range(L)}      
      list_of_deltadeltaG, list_of_roles_in_structure, list_of_element_length, role_only_list, rel_to_structural_element_list, list_of_structures = [], [], [], [], [], []
      list_of_sequences, mutation_list = [], []
      structure_and_pos_to_structural_context, int_to_structure = {}, {}
      for sequence, structure_int in np.ndenumerate(GPmap):
         if not structure_invalid_test(structure_int): 
            for pos in range(L):
               try:
                  role, rel_to_structural_element, element_length, secondarystructure = structure_and_pos_to_structural_context[(structure_int, pos)]
               except KeyError:
                  secondarystructure = get_dotbracket_from_int(structure_int)
                  role, rel_to_structural_element = cg.find_role_in_structure(pos, secondarystructure)
                  element_length = cg.find_element_length(pos, secondarystructure)
                  structure_and_pos_to_structural_context[(structure_int, pos)] = (role[:], rel_to_structural_element[:], element_length, secondarystructure)
               for new_letter in range(K):
                  deltadeltaG = deltadeltaGdata[(pos, new_letter)][sequence]
                  if stabilityarray[sequence] > 0 and not np.isnan(deltadeltaG) and not np.isinf(deltadeltaG): 
                     list_of_roles_in_structure.append(role+' ('+rel_to_structural_element+')')
                     rel_to_structural_element_list.append(rel_to_structural_element)
                     role_only_list.append(role)
                     list_of_deltadeltaG.append(np.abs(deltadeltaG))
                     list_of_element_length.append(element_length)
                     list_of_structures.append(secondarystructure)
                     list_of_sequences.append(DG.sequence_int_to_str(sequence))
                     mutation_list.append(str(pos)+DG.sequence_int_to_str([new_letter,])[0])
      df_deltadeltaG = pd.DataFrame().from_dict({'role in structure': list_of_roles_in_structure, 'deltadeltaG': list_of_deltadeltaG, 
           'element length': list_of_element_length, 'structure': list_of_structures, 'structural role': role_only_list, 
           'position in structural segment': rel_to_structural_element_list, 'sequence': list_of_sequences, 'mutation': mutation_list})
      df_deltadeltaG.to_csv(csv_filename)
      del list_of_deltadeltaG, list_of_roles_in_structure, list_of_element_length, role_only_list, rel_to_structural_element_list, list_of_structures
   df_deltadeltaG = pd.read_csv(csv_filename)
   save_percentiles_for_category_to_csv(df_deltadeltaG, percentiles=[5, 25, 50, 75, 95], structural_categogies=categories_detailed, 
      filename='./thermodynamics_data/summary_deltadeltaGdata_and_role_in_structureL'+str(L)+GPmapdef+thermodynamic_quantity+'.csv', round_ddG=round_ddG)
   save_percentiles_for_category_to_csv(df_deltadeltaG, percentiles=[5, 25, 50, 75, 95], structural_categogies=categories_detailed, 
      filename='./thermodynamics_data/summary_deltadeltaGdata_and_role_in_structureL'+str(L)+GPmapdef+thermodynamic_quantity+'_elementlength.csv', round_ddG=round_ddG, include_length=True)
   ddGboxplot_and_by_element_length(df_deltadeltaG, structural_categogies=categories,
                           filename='./figs_for_report/deltadeltaG_structural_role_and_lengthL'+str(L)+GPmapdef+thermodynamic_quantity+'.png')
   f, ax = plt.subplots(figsize=(8, 6))
   sns.boxplot(data=df_deltadeltaG, x='deltadeltaG', y='role in structure', order=categories_detailed, ax=ax, color='lightgray',
              whis=(5,95), fliersize=0.4, boxprops = {'linewidth': '0.3'})
   for lineindex, line in enumerate(ax.lines):
      if (lineindex - 4)%6 == 0:
         line.set_color('r')
         line.set_linewidth(1.4)
   ax.set_xlabel('stability change after substitution\n'+r'$|\Delta \Delta G|$ (kcal/mol)')
   ax.set_ylabel('')
   f.tight_layout()
   f.savefig('./figs_for_report/deltadeltaG_structural_role_simpleL'+str(L)+GPmapdef+thermodynamic_quantity+'.png')
   del df_deltadeltaG

###############################################################################################
###############################################################################################
print  'draw boxplot of deltadeltaG distributions of L='+ str(L) + ' for both mfe and energygap'
###############################################################################################
###############################################################################################
categories_detailed_horizontalplot = ['hairpin loop (middle)', 'hairpin loop (end)', 'stack (end)', 'stack (middle)',
                                      'bulge (end)', 'internal loop (end)', 'exterior loop (end)', 'exterior loop (middle)']
dict_ddG_different_types_energy = {'deltadeltaG': [], 'landscape': [], 'role in structure': []}
stability_function, deltadeltaG_function, GPmapdef = param.set_parameters_functions_for_thermodynamic_quantity('energygap')   
combined_energygap_freeenergy_file = './thermodynamics_data/deltadeltaGdata_and_role_in_structureL'+str(L)+GPmapdef+'mfe_energygap.csv'
if not isfile(combined_energygap_freeenergy_file):
  for thermodynamic_quantity in  ['free_energy', 'energygap']:
      GPmapdef = param.set_parameters_functions_for_thermodynamic_quantity(thermodynamic_quantity)[2]
      df_deltadeltaG = pd.read_csv('./thermodynamics_data/deltadeltaGdata_and_role_in_structureL'+str(L)+GPmapdef+thermodynamic_quantity+'.csv')
      dict_ddG_different_types_energy['role in structure'] += df_deltadeltaG['role in structure'].tolist()
      dict_ddG_different_types_energy['deltadeltaG'] += df_deltadeltaG['deltadeltaG'].tolist()
      dict_ddG_different_types_energy['landscape'] += [thermodynamic_quantity,]*len(df_deltadeltaG['deltadeltaG'].tolist())
  df = pd.DataFrame.from_dict(dict_ddG_different_types_energy)
  df.to_csv(combined_energygap_freeenergy_file)
df = pd.read_csv(combined_energygap_freeenergy_file)
f, ax = plt.subplots(figsize=(7, 4.5))  
(c1, c2) = sns.color_palette("Paired")[7], sns.color_palette("Paired")[9]
#c1, c2 = 'gold', 'springgreen' #, 'slateblue'
c1, c2 = (0.25, 0.25, 0.25, 1.) , (0.95, 0.95, 0.95, 1.) #(0.7, 0.7, 0.7, 1.) , (0.95, 0.95, 0.95, 1.)
color_palette={'free_energy': c1, 'energygap': c2}
sns.boxplot(data=df, y='role in structure', x='deltadeltaG', color='lightgray', ax=ax, hue='landscape', orient='h',
             order=categories_detailed_horizontalplot, palette=color_palette, whis=(5,95), fliersize=0.4, boxprops = {'linewidth': '0.3'}, medianprops= {'linewidth': 1.4, 'color': 'r'}) #, boxprops = {'linestyle': ':'})#, boxprops = {'linecolor': 'k'}, whiskerprops = {'color': 'k'},capprops = {'color': 'k'}, flierprops={'color': 'k'}, medianprops={'color': 'k'})
ax.set_xlabel('change in thermodynamic quantity\nafter substitution\n'+r'(in kcal/mol)')
ax.set_ylabel('')
ax.get_legend().remove()
# create legend - follow matplotlib manual for custom legends
legend_elements = [Line2D([0], [0], ms=12, marker='s', mew=1.0, markeredgecolor='k', markerfacecolor=c1, label=r'$|\delta G|$' + '\n(free energy landscape)', ls=''),
                   Line2D([0], [0], marker='s', mew=1.0, ms=12, markeredgecolor='k', markerfacecolor=c2, label=r'$|\Delta \Delta G|$'+ '\n(stability landscape)', ls='')]

ax.legend(handles = legend_elements, loc='lower right')#, bbox_to_anchor=(0.96, 0.1))
f.savefig('./figs_for_report/deltadeltaG_role_in_structure_boxplotL'+str(L)+GPmapdef+'twolandscapes.eps', bbox_inches='tight', backend='ps')
f.tight_layout()
f.savefig('./figs_for_report/deltadeltaG_role_in_structure_boxplotL'+str(L)+GPmapdef+'twolandscapes.png', dpi=500, bbox_inches='tight', backend='agg')
plt.close('all')
###############################################################################################
###############################################################################################
print  'plot with larger font'
###############################################################################################
###############################################################################################
categories_detailed_horizontalplot = ['hairpin loop (middle)', 'hairpin loop (end)', 'stack (end)', 'stack (middle)',
                                      'bulge (end)', 'internal loop (end)', 'exterior loop (end)', 'exterior loop (middle)']

GPmapdef = param.set_parameters_functions_for_thermodynamic_quantity('energygap')[2]
combined_energygap_freeenergy_file = './thermodynamics_data/deltadeltaGdata_and_role_in_structureL'+str(L)+GPmapdef+'mfe_energygap.csv'
df = pd.read_csv(combined_energygap_freeenergy_file)
f, ax = plt.subplots(figsize=(5.3, 3.3))  
(c1, c2) = sns.color_palette("Paired")[7], sns.color_palette("Paired")[9]
c1, c2 = (0.25, 0.25, 0.25, 1.) , (0.95, 0.95, 0.95, 1.) 
color_palette={'free_energy': c1, 'energygap': c2}
sns.boxplot(data=df, y='role in structure', x='deltadeltaG', color='lightgray', ax=ax, hue='landscape', orient='h',
             order=categories_detailed_horizontalplot, palette=color_palette, whis=(5,95), fliersize=0.4, boxprops = {'linewidth': '0.3'}, medianprops= {'linewidth': 1.4, 'color': 'r'}) #, boxprops = {'linestyle': ':'})#, boxprops = {'linecolor': 'k'}, whiskerprops = {'color': 'k'},capprops = {'color': 'k'}, flierprops={'color': 'k'}, medianprops={'color': 'k'})
ax.set_xlabel('change in thermodynamic quantity\nafter substitution\n'+r'(in kcal/mol)')
ax.set_ylabel('')
ax.get_legend().remove()
# create legend - follow matplotlib manual for custom legends
legend_elements = [Line2D([0], [0], ms=12, marker='s', mew=1.0, markeredgecolor='k', markerfacecolor=c1, label=r'$|\delta G|$', ls=''),
                   Line2D([0], [0], marker='s', mew=1.0, ms=12, markeredgecolor='k', markerfacecolor=c2, label=r'$|\Delta \Delta G|$', ls='')]

ax.legend(handles = legend_elements, loc='lower right')#, bbox_to_anchor=(0.96, 0.1))
f.savefig('./figs_for_report/deltadeltaG_role_in_structure_boxplotL'+str(L)+GPmapdef+'twolandscapes_largefont.eps', bbox_inches='tight', backend='ps')
f.tight_layout()
f.savefig('./figs_for_report/deltadeltaG_role_in_structure_boxplotL'+str(L)+GPmapdef+'twolandscapes_largefont.png', dpi=500, bbox_inches='tight', backend='agg')
plt.close('all')

###############################################################################################
###############################################################################################
print  'distribution for different structures'
###############################################################################################
###############################################################################################
thermodynamic_quantity = 'energygap'
GPmapdef = param.set_parameters_functions_for_thermodynamic_quantity(thermodynamic_quantity)[2]
dfsets = pd.read_csv('./GPmapdata/neutralset_sizes_stabilities'+str(L)+GPmapdef+thermodynamic_quantity+'.csv')
dfsets2 = pd.DataFrame.from_dict({'structure': dfsets['structure'].tolist(), 'ddG': [round(ddG, 1) for ddG in dfsets['lower quartile of abs(ddG)'].tolist()], 
                                  'has bulge/internal loop': [cg.dotbracket_to_coarsegrained(s).count('[') -1 for s in dfsets['structure'].tolist()]})
print dfsets2.loc[dfsets2['has bulge/internal loop'] == 1]
print '\n_____\n'
print dfsets2.loc[dfsets2['has bulge/internal loop'] == 0]
f, ax = plt.subplots(figsize=(7, 4.5))  
sns.boxplot(data=dfsets2, y='has bulge/internal loop', x='ddG', color='lightgray', ax=ax,  orient='h',
             order=[0, 1], whis=(5,95), fliersize=0.4, boxprops = {'linewidth': '0.3'}, medianprops= {'linewidth': 1.4, 'color': 'r'}) #, boxprops = {'linestyle': ':'})#, boxprops = {'linecolor': 'k'}, whiskerprops = {'color': 'k'},capprops = {'color': 'k'}, flierprops={'color': 'k'}, medianprops={'color': 'k'})
ax.set_xlabel('25% quartile\n'+r' of $|\Delta \Delta G|$ per structure' + '\n(kcal/mol)')
#ax.set_ylabel('')
#ax.get_legend().remove()
f.savefig('./figs_for_report/deltadeltaG_per_structure_boxplotL'+str(L)+GPmapdef+'.eps', bbox_inches='tight', backend='ps')




      