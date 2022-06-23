import numpy as np
import rna_structural_functions as cg



def isunfoldedstructure(structure_int):
   """ check from the integer structure representation if a structure 
   is the "undefined"/unfolded structure (no base pairs)"""
   if structure_int > 1:
      structure_bin = bin(structure_int)[3:] # cut away '0b' and the starting 1
      return '1' not in structure_bin
   else: # unstable structure in $deltaG<0$ map is '0'
      return True

def isunfoldedstructure_or_isolated_bps(structure_int, return_warning=True):
   """ check from the integer structure representation if a structure 
   is the "undefined"/unfolded structure (no base pairs) or if the 
   structure has isolated base pairs"""
   if isunfoldedstructure(structure_int):
      return True
   elif cg.has_length_one_stack(get_dotbracket_from_int(structure_int)):
      if return_warning:
         print 'warning: lonely base pair?'
      return True
   else:
      return False

def get_dotbracket_from_int(structure_int):
   """ retrieve the full dotbracket string from the integer representation"""
   dotbracketstring = ''
   bin_to_db = {'10': '(', '00': '.', '01': ')'}
   structure_bin = bin(structure_int)[3:] # cut away '0b' and the starting 1
   assert len(structure_bin) % 2 == 0
   for indexpair in range(0, len(structure_bin), 2):
      dotbracketstring = dotbracketstring + bin_to_db[structure_bin[indexpair]+structure_bin[indexpair+1]]
   return dotbracketstring



def get_frequencies_in_array(GPmap, ignore_undefined=True, structure_invalid_test=isunfoldedstructure):
   """ sum over an entire GP array: and sum the number of times each structure is found;
   if ignore_undefined=True, the function structure_invalid_test will determine, which structures are selected; 
   otherwise all structures will be selected"""
   ph_vs_f = {}
   for p in GPmap.copy().flat:
      try:
         ph_vs_f[p] += 1
      except KeyError:
         ph_vs_f[p] = 1
   if not ignore_undefined:
      return ph_vs_f
   else:
      return {p:f for p, f in ph_vs_f.iteritems() if not structure_invalid_test(p)}

def site_versatility(sequence_indices, GPmap):
   """ find versatility for each site for given sequebce from GP array"""
   K = GPmap.shape[1]
   site_vs_versatility = {}
   for position in range(len(sequence_indices)):
      neutral_neighbours_pos = 0
      mutated_seq = list(sequence_indices[:]) #construct location of the neighbouring genotypes
      alternative_bases = list(range(K))
      alternative_bases.remove(mutated_seq[position])
      for new_base in alternative_bases:
         mutated_seq[position]=new_base
         if GPmap[tuple(mutated_seq)] == GPmap[tuple(sequence_indices)]:
            neutral_neighbours_pos +=1
      site_vs_versatility[position] = float(neutral_neighbours_pos)/(K-1)
      del mutated_seq
   return site_vs_versatility



############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   print 'test: general_functions.py'
   import thermodynamic_functions as DG 
   teststructure1 = '...(((..((...)))))..((....)).'
   teststructure2 = '(((...))).'
   assert get_dotbracket_from_int(DG.dotbracket_to_int(teststructure1, check_size_limit=False)) == teststructure1
   assert get_dotbracket_from_int(DG.dotbracket_to_int(teststructure2)) == teststructure2
   assert not isunfoldedstructure(DG.dotbracket_to_int(teststructure1, check_size_limit=False))
   assert isunfoldedstructure(DG.dotbracket_to_int('.......'))
   assert not isunfoldedstructure(DG.dotbracket_to_int(teststructure1, check_size_limit=False))
   assert not isunfoldedstructure(DG.dotbracket_to_int(teststructure2, check_size_limit=False))
   assert isunfoldedstructure_or_isolated_bps(DG.dotbracket_to_int('.((.(...)))..'), return_warning=False)
   assert isunfoldedstructure_or_isolated_bps(DG.dotbracket_to_int('.((.(...).))..'), return_warning=False)
   assert isunfoldedstructure_or_isolated_bps(DG.dotbracket_to_int('.(((...).))..'), return_warning=False)
   print '\n\n-------------\n\n'
