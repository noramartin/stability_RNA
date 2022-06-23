import numpy as np
import networkx as nx
from general_functions import isunfoldedstructure, get_frequencies_in_array
from os.path import isfile
from functools import partial
from multiprocessing import Pool
import networkx as nx
from rna_structural_functions import get_basepair_indices_from_dotbracket

base_to_number={'A':0, 'C':1, 'U':2, 'G':3, 'T':2}
allowed_basepairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
allowed_basepairs_tuples = [(base_to_number[bp[0]], base_to_number[bp[1]]) for bp in allowed_basepairs]

############################################################################################################
## point-mutational neighbours
############################################################################################################
def neighbours_g(g, K, L): 
   """list all pont mutational neighbours of sequence g (integer notation)"""
   return [tuple([oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)]) for pos in range(L) for new_K in range(K) if g[pos]!=new_K]

def neighbours_g_given_site(g, K, L, pos): 
   """list all pont mutational neighbours of sequence g (integer notation) with a substitution at position pos"""
   return [tuple([oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)]) for new_K in range(K) if g[pos]!=new_K]

def bp_mutation_not_point_mutation(g_int, structure_dotbracket):
   """perform all swaps of paired positions with allowed base pairs except those that are single point mutants
   (GC -> AU swap performed as well as GC -> CG, but not GC-> GU)"""
   L, g_list = len(g_int), []
   base_pair_index_mapping = get_basepair_indices_from_dotbracket(structure_dotbracket)
   bp_list = list(set([(min(bp1, bp2), max(bp1, bp2)) for bp1, bp2 in base_pair_index_mapping.iteritems()]))
   for bp0, bp1 in bp_list:
      for new_bp in allowed_basepairs_tuples:
         if new_bp[0] != g_int[bp0] and new_bp[1] != g_int[bp1]: # not identical bp or point mutants of it
            g_new = [x for x in g_int]
            g_new[bp0] = new_bp[0]
            g_new[bp1] = new_bp[1]
            g_list.append(tuple(g_new))
   return g_list

def bp_swaps_g_given_site(g_int, pos, structure_dotbracket): 
   """perform all base pair swaps at the given position and its paired position"""
   g_list = []
   assert structure_dotbracket[pos] in ['(', ')']
   paired_pos = get_basepair_indices_from_dotbracket(structure_dotbracket)[pos]
   for new_bp in allowed_basepairs_tuples:
      if new_bp[0] != g_int[pos] or new_bp[1] != g_int[paired_pos]: # not identical bp 
         g_new = [x for x in g_int]
         g_new[pos] = new_bp[0]
         g_new[paired_pos] = new_bp[1]
         g_list.append(tuple(g_new))
   return g_list

############################################################################################################
## identify neutral components
############################################################################################################
def find_all_NCs_parallel(GPmap, GPmapdef=None):
   """find individual neutral components for all non-unfolded structures in GP map;
   saves/retrieves information if GPmapdef given and files present; otherwise calculation from scratch"""
   K, L = GPmap.shape[1], GPmap.ndim
   if not GPmapdef or not isfile('./GPmapdata/NCmap_L'+str(L)+GPmapdef+'.npy'):
      structure_list = list(set([ph for g, ph in np.ndenumerate(GPmap) if not isunfoldedstructure(ph)]))
      find_NC_correct_const_arguments = partial(find_NC, GPmap=GPmap)
      pool = Pool()
      NC_per_structure_list = pool.map(find_NC_correct_const_arguments, structure_list) 
      pool.close()
      pool.join()
      NCindex_global_count = 1 #such that 0 remains for undefined structure
      seq_vs_NCindex_array = np.zeros((K,)*L, dtype='uint32')
      for structureindex, structure in enumerate(structure_list):
         NC_dict_this_structure = NC_per_structure_list[structureindex]
         map_to_global_NCindex = {NCindex: NCindex_global_count + NCindex for NCindex in set(NC_dict_this_structure.values())}
         NCindex_global_count = max(map_to_global_NCindex.values()) + 1
         for seq, NCindex in NC_dict_this_structure.iteritems():
            seq_vs_NCindex_array[seq] = map_to_global_NCindex[NCindex]
      if GPmapdef:
         np.save('./GPmapdata/NCmap_L'+str(L)+GPmapdef+'.npy', seq_vs_NCindex_array)
   else:
      seq_vs_NCindex_array = np.load('./GPmapdata/NCmap_L'+str(L)+GPmapdef+'.npy')
   NCindex_vs_NCsize = get_frequencies_in_array(seq_vs_NCindex_array, ignore_undefined=False)
   del NCindex_vs_NCsize[0]
   return seq_vs_NCindex_array, NCindex_vs_NCsize

def find_NC(structure_int, GPmap):
   """find individual neutral components in the neutral set of the structure structure_int in GPmap;
   NCindex will start at 0"""
   K, L = GPmap.shape[1], GPmap.ndim
   neutral_set = nx.Graph()
   neutral_set.add_nodes_from([tuple(g) for g in np.argwhere(GPmap==structure_int)])
   for g in neutral_set.nodes():
      for g2 in neighbours_g(g, K, L):
         if GPmap[g2] == structure_int:
            neutral_set.add_edge(g, g2)
   geno_vs_NC = {g: NCindex for NCindex, list_of_geno in enumerate(nx.connected_components(neutral_set)) for g in list_of_geno}
   return geno_vs_NC

def get_structure_forNCindex(seq_vs_NCindex_array, GPmap, NCindex):
   """find the common structure (integer representation) of the NC denoted by 
   NCindex in the array seq_vs_NCindex_array"""
   example_seq = [tuple(g) for g in np.argwhere(seq_vs_NCindex_array==NCindex)][0]
   return GPmap[example_seq]

############################################################################################################
## NC as network
############################################################################################################
def get_NC_as_network(NCindex, seq_vs_NCindex_array, deltaGarray=[0,0,]):
   """convert the NC with index NCindex to a networkx network (for plotting);
   if deltaGarray is given, this will be recorded as attribute deltaG"""
   K, L =  seq_vs_NCindex_array.shape[1], seq_vs_NCindex_array.ndim
   seqlist_NC_array = np.argwhere(seq_vs_NCindex_array==NCindex)
   neutral_component_list_of_seq = [tuple(seqlist_NC_array[i]) for i in range(len(seqlist_NC_array))]
   NC_graph = nx.Graph()
   NC_graph.add_nodes_from(neutral_component_list_of_seq)
   for g1 in neutral_component_list_of_seq:
      for g2 in neighbours_g(g1, K, L):
         if seq_vs_NCindex_array[g2] == NCindex:
            NC_graph.add_edge(g1, g2)
   if deltaGarray.ndim == L:
      NCgeno_vs_deltaG = {g: deltaGarray[tuple(g)] for g in NC_graph.nodes()}
      nx.set_node_attributes(NC_graph, name='deltaG', values=NCgeno_vs_deltaG)    
   return NC_graph

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   print 'start tests: neutral_component.py'
   ####
   print 'test mutational neighbours'
   ####
   g, K, L = (1, 2, 3, 1), 4, 4
   point_mutational_neighbours = neighbours_g(g, K, L)
   point_mutational_neighbours_reference = [(0, 2, 3, 1), (2, 2, 3, 1), (3, 2, 3, 1), (1, 0, 3, 1), (1, 1, 3, 1), (1, 3, 3, 1),
                                            (1, 2, 0, 1), (1, 2, 1, 1), (1, 2, 2, 1), (1, 2, 3, 0), (1, 2, 3, 2), (1, 2, 3, 3)]
   substitutions_reference = [(1, 0, 3, 1), (1, 1, 3, 1), (1, 3, 3, 1)]
   substitutions = neighbours_g_given_site(g, K, L, 1)
   assert len(point_mutational_neighbours) == len(point_mutational_neighbours_reference)
   for mutational_neighbour in point_mutational_neighbours_reference:
      assert mutational_neighbour in point_mutational_neighbours
   assert len(substitutions) == len(substitutions_reference)
   for mutational_neighbour in substitutions_reference:
      assert mutational_neighbour in substitutions
   from epistasis import Hamming_dist
   for test_no in range(1000):
      seq = tuple(np.random.choice(np.arange(4), L, replace=True)) 
      point_mutational_neighbours = neighbours_g(seq, K, L)
      substitutions = neighbours_g_given_site(seq, K, L, np.random.choice(L))
      assert len(point_mutational_neighbours) == (K-1) * L == len(set(point_mutational_neighbours))
      assert len(substitutions) == (K-1) == len(set(substitutions))
      for seq2 in point_mutational_neighbours:
         assert Hamming_dist(seq2, seq) == 1
      for seq2 in substitutions:
         assert Hamming_dist(seq2, seq) == 1
   ####
   print 'test base pair swaps'
   ####
   import thermodynamic_functions as DG
   from rna_structural_functions import sequence_compatible_with_basepairs
   structure_dotbracket = '((...))'
   sequence = (3, 1, 0, 0, 0, 3, 1)
   bp_swaps_reference = [(0, 1, 0, 0, 0, 3, 2), (3, 0, 0, 0, 0, 2, 1), (3, 2, 0, 0, 0, 0, 1), (1, 1, 0, 0, 0, 3, 3),
                         (3, 3, 0, 0, 0, 1, 1), (2, 1, 0, 0, 0, 3, 0), (3, 3, 0, 0, 0, 2, 1), (2, 1, 0, 0, 0, 3, 3)]
   bp_swaps = bp_mutation_not_point_mutation(sequence, structure_dotbracket)
   bp_swaps0 = bp_swaps_g_given_site(sequence, 0, structure_dotbracket)
   bp_swaps6 = bp_swaps_g_given_site(sequence, 0, structure_dotbracket)
   assert len(bp_swaps) == len(bp_swaps_reference)
   assert len(bp_swaps0) == 5 == len(bp_swaps6)
   for seq2 in bp_swaps_reference:
      assert seq2 in bp_swaps and sequence_compatible_with_basepairs(DG.sequence_int_to_str(seq2), structure_dotbracket)
   for seq2 in bp_swaps0:
      assert 1 <= Hamming_dist(seq2, sequence) <= 2 and sequence_compatible_with_basepairs(DG.sequence_int_to_str(seq2), structure_dotbracket)
   for seq2 in bp_swaps6:
      assert 1 <= Hamming_dist(seq2, sequence) <= 2 and sequence_compatible_with_basepairs(DG.sequence_int_to_str(seq2), structure_dotbracket)
   ####
   print 'neutral component: toy examples'
   ####
   def create_epistasis_testcase(seq_vs_struct, K=4):
      """ build test case arrays to test NC detection"""
      L = len(seq_vs_struct.keys()[0])
      testGPmap = np.zeros((K,)*L, dtype='uint32')
      for seq, structure in seq_vs_struct.iteritems():
         testGPmap[seq] = structure
      return testGPmap
   def create_deltaG_testcase(seq_list, K=4):
      """ build test case arrays to test NC network"""
      L = len(seq_vs_struct.keys()[0])
      testdeltaG = np.zeros((K,)*L, dtype='uint32')
      for seq in seq_list:
         testdeltaG[seq] = np.sum(seq)
      return testdeltaG
   seq_vs_struct = {(0, 0, 0): 50, (0, 1, 0): 50, (2, 0, 0): 50, (1, 1, 1): 50, (1, 2, 1): 50, (2, 0, 1): 82, (0, 0, 1): 82, (0, 1, 1): 82}
   testGPmap = create_epistasis_testcase(seq_vs_struct, K=4)
   seq_vs_NCindex_array, NCindex_vs_NCsize = find_all_NCs_parallel(testGPmap, GPmapdef=None)
   for seq, NCindex in np.ndenumerate(seq_vs_NCindex_array):
      assert NCindex == 0 or seq in seq_vs_struct
   assert seq_vs_NCindex_array[(0, 0, 0)] == seq_vs_NCindex_array[(0, 1, 0)] == seq_vs_NCindex_array[(2, 0, 0)] != seq_vs_NCindex_array[(1, 1, 1)]
   assert seq_vs_NCindex_array[(1, 1, 1)] == seq_vs_NCindex_array[(1, 2, 1)] != seq_vs_NCindex_array[(0, 0, 1)]
   assert seq_vs_NCindex_array[(2, 0, 1)] == seq_vs_NCindex_array[(0, 0, 1)] == seq_vs_NCindex_array[(0, 1, 1)] != seq_vs_NCindex_array[(0, 0, 0)]
   assert NCindex_vs_NCsize[seq_vs_NCindex_array[(0, 0, 0)]] == 3
   assert NCindex_vs_NCsize[seq_vs_NCindex_array[(1, 1, 1)]] == 2
   assert NCindex_vs_NCsize[seq_vs_NCindex_array[(2, 0, 1)]] == 3
   assert get_structure_forNCindex(seq_vs_NCindex_array, testGPmap, seq_vs_NCindex_array[(0, 0, 0)]) == 50
   ####
   print 'create network'
   ####
   NC_network = get_NC_as_network(NCindex=seq_vs_NCindex_array[(0, 0, 0)], seq_vs_NCindex_array=seq_vs_NCindex_array, deltaGarray=create_deltaG_testcase(seq_vs_struct.keys(), K=4))
   for n, data in NC_network.nodes(data=True):
      assert sum(n) == data['deltaG']
   assert len( NC_network.nodes()) == 3
   for n1 in NC_network.nodes(data=True):
      for n2 in NC_network.nodes(data=True):
         if len([i for i in range(len(n1)) if n1[i] != n2[i]]) == 1:
            assert nx.has_edge(n1, n2)
   print '\n\n-------------\n\n'


