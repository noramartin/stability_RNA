Code for “Thermodynamics and neutral sets in the RNA sequence-structure map“

This code performs the data analysis and plots presented in the paper. All data is calculated using ViennaRNA (2.4.14). The code for analysis and plotting is written in Python 2.7 and the following packages were used: Matplotlib (2.2.3), NetworkX (2.2), script (1.2.1), numpy (1.16.5), seaborn (0.9.0), pandas (0.24.2) and subprocess (3.5.4).

Usually sequences are represented as tuple of integers - this allows us to store the GPmap in an array indexed by sequence tuples rather than as a dictionary. The conversion from nucleotides to integers is: {'A':0, 'C':1, 'U':2, 'G':3, 'T':2}.
Similarly, structures are converted into integers to allow easy storage in arrays. This is performed by converting them into binary strings first and then into decimals. The calculation is onto-one and can be calculated in either direction.

There are several sequence-structure map definitions: 'mfe' for simple RNAfold calculations, 'mfe_unique' where an additional check is included whether the mfe structure is unique and 'mfe_unique_fifty' where in addition it is checked if the mfe structure has a Boltzmann frequency of > 50%. Discarded structures are denoted by "2"/as unfoldded structures.

The files in this folder are for saving data and generating the figures; the methods are implemented as functions in the GPfunctions folder. The script save_thermodynamic_data_parallel.py needs to be run first to generate the data for full enumeration of L=13 sequences.

To estimate neutral set sizes for L=30, an external command line tool is needed: the neutral network size estimator (Jörg et al. 2008) needs to be downloaded separately from the Wagner lab website. Since the original tool allows isolated base pairs, two lines have to be added in the main script in the C code:
int    lppar = 1;
noLonelyPairs = lppar;
Both following the ViennaRNA manual.  Then this code needs to be compiled following the instructions provided by Jörg et al. (or the instructions given in the data by Weiß and Ahnert) and applied to all structures in the final L=30 structural sample. 

References:
- Site-scanning is adapted from the algorithm described in Weiß, Marcel; Ahnert, Sebastian E. (2020): Supplementary Information from Using small samples to estimate neutral component size and robustness in the genotype-phenotype map of RNA secondary structure. The Royal Society. Journal contribution. https://doi.org/10.6084/m9.figshare.12200357.v2; here base pair swaps are used as well as point mutations
- ViennaRNA manual: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf
- Neutral network size estimator reference: Jörg, T., Martin, O. C. & Wagner, A. Neutral network sizes of biological RNA molecules can be computed and are not atypically small. BMC Bioinformatics 9, 464 (2008).


Notation used throughout the code:
L - sequence length
K - alphabet size (4 for RNA), following notation by Schaper and Louis (doi:10.1371/journal.pone.0086635)