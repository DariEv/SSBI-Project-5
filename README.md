# SSBI-Project-5

## Secondary structure annotation of tertiary structures

### Authors: Tobias Nietsch, Florian Koenig, Daria Evseeva
### Supervisor: Eugen Netz  

Required packages: sklearn, Bio.PDB, pickle

##### Usage Guide:

Main scripts 
`k_nearest_neighbor.py`, `RandomForest.py` and `SVM.py`
can be directly called from command line:

`python <predictor_name>.py`

Results of predictions will be written in .pkl files 
`<predictor_name>/<predictor_name>_q6.pkl` and
`<predictor_name>/<predictor_name>_q3.pkl`.

The scripts require features file `Extracted_Features.pkl`
and scoring function script `SOV.py`
to be in the same directory.

+ Link zu precomputed Features: https://drive.google.com/open?id=1U2U7UJ6AXCnndyAJYBrgqnmnrKgFXt1l

Otherwise the features can be extracted from a directory 
containing .pdb files
and written to 
`Extracted_Features.pkl` file by
`feature_extractor.py`, which requires 
`h_bonds_calculator.py` and
`ssbi_project_h_atoms.py`
for calculations. The path to the directory
can be edited in the **main()** function of 
`feature_extractor.py`.
