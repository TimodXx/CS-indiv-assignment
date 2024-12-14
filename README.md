To run the results, in total 4 files need to be runned and the input docuement TVs-all-merged.json is used as input
1. data_cleaning.ipynb:
    1.1 Run the code under Run1
    1.2 Run the code under Run2
    1.3 Run the code under Run3
    1.4 Run the code under Run4
    1.5 Run the code under Run5

2. baseline_cleaning.ipynb
    2.1 Run the code under Run1
    2.2 Run the code under Run2

3. run main.py

4. LSH RESULTS: unfortunately, the file didnt get saved. I cant find it anymore, but I think it was plots_results.ipynb,
    This file combined the 2 dataframes 
    I plotted the results, of the mean over the 5 bootstrap samples, here I saved the excel file and also the plots
    The plots are called:
        1. f1 star vs fractions of comparison Plot.png
        2. Pair Completeness vs fractions of comparison Plot.png
        3. Pair Quality vs fractions of comparison Plot.png
    
    The results are in these 2 excel files:
        1. Baseline: results_msmp+_paper.xlsx
        2. MSMPB: results_msmpb_paper.xlsx
    
    As you can see these values are inlign with the plot and will be the same for 5 bootstraps
    
