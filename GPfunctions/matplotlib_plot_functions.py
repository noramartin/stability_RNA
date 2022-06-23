# import matplotlib
# matplotlib.use('ps')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
import numpy as np
from thermodynamic_functions import sequence_int_to_str, get_mfe_structure
from scipy.stats import spearmanr, pearsonr
import neutral_component as NC
import networkx as nx
from general_functions import get_dotbracket_from_int
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from os.path import isfile
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import rna_structural_functions as cg
import matplotlib.colors as mcolors

def save_percentiles_for_category_to_csv(df, percentiles, structural_categogies, filename, include_length=False, round_ddG=False):
   """take dataframe with mutational impacts and their structural roles 
   and record percentiles of deltadeltaG for each category"""
   dict_for_df = {str(percent) + 'percentile': [] for percent in percentiles}
   dict_for_df['role in structure'] = []
   if include_length:
      dict_for_df['element length'] = []
   for type_element in structural_categogies:
      if not include_length:
         deltadeltaG_list_given_cat = df.loc[df['role in structure']==type_element]['deltadeltaG'].tolist()
         dict_for_df['role in structure'].append(type_element)
         for percent in percentiles:
            if round_ddG:
               dict_for_df[str(percent) + 'percentile'].append(round(np.percentile(deltadeltaG_list_given_cat, percent), 1))
            else:   
               dict_for_df[str(percent) + 'percentile'].append(np.percentile(deltadeltaG_list_given_cat, percent))
         print type_element, 'fraction of 0 values in ddG:', len([ddG for ddG in deltadeltaG_list_given_cat if np.abs(ddG) < 0.0001])/float(len(deltadeltaG_list_given_cat))
      else:
         df_given_cat = df.loc[df['role in structure']==type_element]
         L_column_as_list = df_given_cat['element length'].tolist()
         for L_element in sorted(list(set(L_column_as_list))):
            deltadeltaG_list_given_cat_and_length = df_given_cat.loc[df['element length']==L_element]['deltadeltaG'].tolist()
            dict_for_df['role in structure'].append(type_element)
            dict_for_df['element length'].append(L_element)
            for percent in percentiles:
               if round_ddG:
                  dict_for_df[str(percent) + 'percentile'].append(round(np.percentile(deltadeltaG_list_given_cat_and_length, percent), 1))
               else:
                  dict_for_df[str(percent) + 'percentile'].append(np.percentile(deltadeltaG_list_given_cat_and_length, percent))
   df_to_save = pd.DataFrame.from_dict(dict_for_df)
   df_to_save.to_csv(filename)



def ddGboxplot_and_by_element_length(df, filename, structural_categogies, Nmin = 20):
   """take dataframe with mutational impacts and their structural roles and plot
   - large boxplot with mutational impacts and structural roles
   - smaller plots underneath for each category: length of structural element vs deltadeltaG
     (plot median for each group and errorbars to lower/upper quartiles)
   - Nmin is the minimum naumber of ddG values required in the sample to plot one errorbar"""
   f = plt.figure(figsize=(2.0*len(structural_categogies), 7)) #constrained_layout=True, 
   gs = gridspec.GridSpec(ncols=len(structural_categogies), nrows=4, figure=f)
   ax0 = f.add_subplot(gs[:2, :])
   color_palette ={'middle': 'yellowgreen', 'end': 'tab:blue', 'fork': 'r'} 
   #color_palette_bp ={'middle': (0.71, 0.86, 0.41, 0.6), 'end': (0.36, 0.61, 0.79, 0.6), 'fork': 'r'} 
   #color_palette_bp ={'middle': (0.9, 0.9, 0.9, 1.0), 'end': (0.75, 0.75, 0.75, 1.0), 'fork': 'r'} 
   sns.boxplot(data=df, x='structural role', y='deltadeltaG', ax=ax0, hue='position in structural segment', 
               order=structural_categogies, palette=color_palette, whis=(5,95), fliersize=2, boxprops = {'linewidth': '0.2'})
   for lineindex, lineboxplot in enumerate(ax0.lines):
      if (lineindex - 4)%6 == 0:
         lineboxplot.set_color('r')
   ax0.set_ylabel('stability change after substitution\n'+r'$|\Delta \Delta G|$ (kcal/mol)')
   ax0.set_xlabel('')
   legend = ax0.get_legend()
   legend.set_title('')
   ax_list, maxylist, minylist = [], [], []
   for columnindex, type_element in enumerate(structural_categogies):
      df_type_element = df.loc[df['structural role']==type_element]
      for rowindex, type_position in enumerate(['end', 'middle']):
         df_to_plot = df_type_element.loc[df_type_element['position in structural segment']==type_position]
         if len(df_to_plot['element length'].tolist()) > 0:
            ax0 = f.add_subplot(gs[2+rowindex, columnindex])
            #df_to_plot = df[df['structural role']==type_element][df['position in structural segment']==type_position]
            L_column_as_list = df_to_plot['element length'].tolist()
            deltadeltaG_column_as_list = df_to_plot['deltadeltaG'].tolist()
            #L_column_as_list, deltadeltaG_column_as_list = zip(*structural_context_vs_length_ddG_list[(type_position,  type_element)])
            if rowindex == 0 or type_position=='end': # share x axis with upper row, always need new x-axis for ends as they occur in shorter elements of L=1
               L_element_list = sorted(list(set(L_column_as_list)))
            L_element_vs_deltadeltaG = {L_element: [ddG for ddG_i, ddG in enumerate(deltadeltaG_column_as_list) if L_column_as_list[ddG_i] == L_element] for L_element in L_element_list}
            median_ddG_list = np.array([np.median(L_element_vs_deltadeltaG[L_element]) if len(L_element_vs_deltadeltaG[L_element]) >= Nmin else np.nan for L_element in L_element_list])
            lower_quartile_ddG_list = np.array([np.percentile(L_element_vs_deltadeltaG[L_element], 25) if len(L_element_vs_deltadeltaG[L_element]) >= Nmin else np.nan for L_element in L_element_list])
            upper_quartile_ddG_list = np.array([np.percentile(L_element_vs_deltadeltaG[L_element], 75) if len(L_element_vs_deltadeltaG[L_element]) >= Nmin else np.nan for L_element in L_element_list])
            ax0.errorbar(L_element_list, median_ddG_list, yerr=[median_ddG_list-lower_quartile_ddG_list, upper_quartile_ddG_list-median_ddG_list], 
                        color=[color_palette['end'], color_palette['middle']][rowindex], marker='.', linestyle='-') #
            ax0.set_ylabel(r'$|\Delta \Delta G|$ (kcal/mol)')
            ax0.set_xlabel('length of '+type_element)
            ax0.set_title(type_element+'\n('+type_position+')')
            ax0.set_xlim(min(L_element_list)-0.5, max(L_element_list)+0.5)
            ax_list.append(ax0)
            maxylist.append(max(upper_quartile_ddG_list))
            minylist.append(max(lower_quartile_ddG_list))
   miny, maxy = min(minylist), max(maxylist)
   for ax0 in ax_list:
      ax0.set_ylim(miny - 0.1 * (maxy - miny), maxy + 0.1 * (maxy - miny))
   f.tight_layout()
   f.savefig(filename, dpi=300, bbox_inches='tight')
   f.savefig(filename[:-4] + '.eps', dpi=300, bbox_inches='tight')
   plt.close('all')




def plot_NC_and_deltaG_discrete(NCindex, ax, f, seq_vs_NCindex_array, deltaGarray, string_saving, cax=None, cbar_label='', structure=''):
   """plot the NC with NCindex as a network, with sequence color given by deltaGarray
   - seq_vs_NCindex_array is used to builde the network and deltaGarray to find corresponding stability values
   - f, ax for drawing
   - string_saving: the network is saved for future use - string_saving should give sufficient information to identify the network uniquely
   - cax: axis for colorbar
   - cbar_label (colorbar label) and structure are used for labelling
   """
   graph_filename = './NC_graphs_data/graph_for_plotting'+string_saving+'NCindex'+str(NCindex)+'.gpickle'
   if not isfile(graph_filename):
      NC_graph = NC.get_NC_as_network(NCindex, seq_vs_NCindex_array, deltaGarray=deltaGarray)
      pos = nx.spring_layout(NC_graph)
      nx.set_node_attributes(NC_graph, name='position', values=pos)
      nx.write_gpickle(NC_graph, graph_filename)  
   else:
      NC_graph = nx.read_gpickle(graph_filename)
      pos = {n: data['position'] for n, data in NC_graph.nodes(data=True)}
   node_colors = [data['deltaG'] for n, data in NC_graph.nodes(data=True)]
   cmap = plt.get_cmap('viridis')
   positions_list = [data['position'] for n, data in NC_graph.nodes(data=True)]
   norm = mcolors.BoundaryNorm(np.arange(min(node_colors) - 0.05, max(node_colors) + 0.051, 0.1), cmap.N)
   scatterpoints = ax.scatter(zip(*positions_list)[0], zip(*positions_list)[1], 
                              c=node_colors, s=2, linewidths=0.0, cmap=cmap, norm=norm, zorder=2) # alpha=0.8
   nx.draw_networkx_edges(NC_graph, pos, width=0.05, alpha=0.3, c='k', ax=ax)
   ticks_colorbar = np.arange(min(node_colors), max(node_colors)+0.1, 0.5)
   if cax:
      cb = f.colorbar(scatterpoints, cax = cax, orientation='horizontal', fraction=0.04, ticks=ticks_colorbar)
   else:
      #cb = add_colorbar_above_plot(ax, scatterpoints, cbar_label, ticks=ticks_colorbar)
      cb = f.colorbar(scatterpoints, ax = ax, ticks=ticks_colorbar)
   cb.set_label(cbar_label)
   cb.ax.tick_params(labelsize=11)
   minx, maxx = min(zip(*positions_list)[0]), max(zip(*positions_list)[0])
   miny, maxy = min(zip(*positions_list)[1]), max(zip(*positions_list)[1])
   ax.set_xlim(minx - 0.05 * (maxx - minx), maxx + 0.05 * (maxx - minx))
   ax.set_ylim(miny - 0.05 * (maxy - miny), maxy + 0.05 * (maxy - miny))
   ax.axis('off')
   seq_list, dG_list = [], []
   for n, data in NC_graph.nodes(data=True):
      seq_list.append(sequence_int_to_str(n))
      dG_list.append(round(data['deltaG'], 1))
   df = pd.DataFrame.from_dict({'sequence': seq_list, 'deltaG': dG_list})
   df.to_csv('./NC_graphs_data/graph_for_plotting'+string_saving+'NCindex'+str(NCindex)+'.csv')

def add_colorbar_next_to_plot(f, sc, label, y_fraction, ticks=None):
   """add a colorbar left of a plot sc in figure f and label it;
   y_fraction specifies the lower corner and length of the colorbar as a fraction of the figure height"""
   cax = f.add_axes([-0.025, y_fraction[0], 0.02, y_fraction[1]])
   cb = f.colorbar(sc, cax=cax, ticks=ticks)
   cb.ax.set_ylabel(label, fontsize= 17)
   cb.ax.yaxis.set_ticks_position('left')
   cb.ax.yaxis.set_label_position('left')


def add_colorbar_above_plot(ax, sc, label, ticks=None, extra_space=False):
   """add a colorbar above a plot sc in subplot ax and label it"""
   locatable_ax = make_axes_locatable(ax) # following Tutorial on "Colorbar with AxesDivider" on Matplotlib documentation
   if extra_space:
      pad='35%'
   else:
      pad = '25%'
   cax = locatable_ax.append_axes("top", size='5%', pad=pad) #pad is needed to decrease overlap between figure and axis label
   cb = colorbar(sc, cax=cax, orientation='horizontal', ticks=ticks)
   cb.set_label_text(label)
   return cb

def hist2D(x, y, ax, xlabel='', ylabel='', title='', y_edges_input=[], x_edges_input=[], alpha=1, c='steelblue', scale=0.01):
   """2D histogram for x/y
   x_edges_input and y_edges_input can be integers or arrays"""
   if min(len(y_edges_input), len(x_edges_input)) == 0:
      x_edges_input, y_edges_input = 10, 10
   H, xedges, yedges = np.histogram2d(x, y, bins=[x_edges_input, y_edges_input])
   x_for_2D_hist_list, y_for_2D_hist_list, size_list = [], [], []
   size_list = []
   for i in range(len(xedges)-1):
      for j in range(len(yedges)-1):
         if H[i,j]>0:
            x_for_2D_hist_list.append((xedges[i]+xedges[i+1])*0.5)
            y_for_2D_hist_list.append((yedges[j]+yedges[j+1])*0.5)
            size_list.append(H[i,j] * scale)
   scatterplot=ax.scatter(x_for_2D_hist_list, y_for_2D_hist_list, edgecolors='none', s=size_list, alpha=alpha, c=c)
   ax.set_xlabel(xlabel)
   ax.set_ylabel(ylabel)
   ax.set_title(title)
   corr_coeff, pvalue = pearsonr(x, y)
   ax.annotate(r'Pearson $\rho=$'+str(round(corr_coeff,2)), xy=(0.06, 0.06), xycoords='axes fraction')


def plot_simple(xlist, ylist, ax, xlabel='', ylabel='', title=''):
   """simple plot"""
   ax.plot(xlist, ylist, c='steelblue', ms=3, marker='o')
   ax.set_xlabel(xlabel)
   ax.set_ylabel(ylabel)
   ax.set_title(title)



def scatter_plot_pale(xlist, ylist, ax, xlabel='', ylabel='', title='', s=1, alpha=0.02, c='steelblue'):
   """scatter plot with small and pale scatter points for large data sets; annotated with Pearson correlation"""
   sc = ax.scatter(xlist, ylist, alpha=alpha, s=s, c=c)
   ax.set_xlabel(xlabel)
   ax.set_ylabel(ylabel)
   ax.set_title(title)
   corr_coeff, pvalue = pearsonr(xlist, ylist)
   if corr_coeff < 0:
      ax.annotate(r'Pearson $\rho=$'+str(round(corr_coeff,2)), xy=(0.06, 0.06), xycoords='axes fraction')
   else:
      ax.annotate(r'Pearson $\rho=$'+str(round(corr_coeff,2)), xy=(0.98, 0.06), xycoords='axes fraction', horizontalalignment='right')
   return sc


def scatter_plt_set_size_plot(x, y, c, marker_list, xlabel, ylabel, ax, s=10, cmap=plt.cm.inferno, markeredgewidth=0.0, norm=None):
   """scatter plot with log scale on x axis"""
   if len(set(marker_list)) > 1: # markers differ
      for i in range(len(x)):
         sc = ax.scatter(x[i], y[i], s=s, c=c[i], cmap=cmap, marker=marker_list[i], linewidths=markeredgewidth, norm=norm)
   else:
      sc = ax.scatter(x, y, s=s, c=c, cmap=cmap, marker=marker_list[0], linewidths=markeredgewidth, norm=norm)
   ax.set_xlabel(xlabel, fontsize= 17)
   ax.set_ylabel(ylabel, fontsize= 17)
   ax.set_xscale('log')
   ax.set_xlim(0.5*min(x), 2*max(x))
   ax.set_ylim(min(y)-0.1*abs(max(y)), max(y)+0.1*abs(max(y)))
   ax.tick_params(axis='both', labelsize=15, length=7, width=1.5)
   ax.tick_params(axis='both', which='minor', length=3, width=1.2)
   return sc
  


def dotbracketstring_to_color(dotbracketstring, type_str_number_basepairs_vs_color):
   """return color to highlight structural features"""
   roles_in_structure = [cg.find_role_in_structure(pos, dotbracketstring)[0] for pos in range(len(dotbracketstring))]
   if 'bulge' in roles_in_structure:
      return type_str_number_basepairs_vs_color['bulge'][dotbracketstring.count('(')]
   elif 'internal loop' in roles_in_structure:
      return type_str_number_basepairs_vs_color['internal'][dotbracketstring.count('(')]
   else:
      return type_str_number_basepairs_vs_color['stack'][dotbracketstring.count('(')]

def dotbracketstring_to_marker(dotbracketstring, type_str_vs_marker):
   """return color to highlight structural features"""
   roles_in_structure = [cg.find_role_in_structure(pos, dotbracketstring)[0] for pos in range(len(dotbracketstring))]
   if 'bulge' in roles_in_structure:
      return type_str_vs_marker['bulge']
   elif 'internal loop' in roles_in_structure:
      return type_str_vs_marker['internal']
   else:
      return type_str_vs_marker['stack']

