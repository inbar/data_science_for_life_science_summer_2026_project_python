# Readme and External Links

## Submission
The rendered python scripts (pdf) are uploaded to the Whiteboard. 

Everything else can be found in the GitHub repo below. 

Analysis was conducted accross two axes of granularity: 
1. The annotation level (how many different cell types were annontated)
    * L1 - 8 cell types
    * L2 - 30 cell types
2. The sample size
    * Full dataset (161k cells)
    * Subsamples: 10k, 25k

The analysis for the different combinations is available in the corresponding notebooks:

|| L1 | L2 | 
|--|--|--|
|10k|✅|❎|
|25k|✅|✅|
|Full dataset (161K)|✅|✅*|

\* Note: Functional annotation (GO enrichment) was only conducted on the L2/161K dataset. 

## Code 
* Processing and application of statistical/ML methods is done via standalone python scripts. 
* Analysis and plot generation done within python notebooks

The entire codebase is available via GitHub:

https://github.com/inbar/data_science_for_life_science_summer_2026_project_python


## Data

Data is available under

https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/

