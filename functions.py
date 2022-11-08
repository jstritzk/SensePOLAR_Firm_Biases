import bertFuncs as func
import functions
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import math

from sklearn.metrics.pairwise import cosine_similarity

from transformers import logging
logging.set_verbosity_error()

from tqdm.notebook import tqdm
import numpy as np


from scipy.stats import ttest_ind
from collections import defaultdict

from nltk.corpus import wordnet as wn

import json
import pickle





def get_firm_embeddings(firm_name, context, numPolar, dimensions): 
    
    """Gets male and female polar embeddings for "firm_name" in "context"
    numPolar specifies the number of considered polar dimensions (currently maximum 1762)
    "dimensions" specifies which of those 1762 dimensions should be included.
    If "all" is selected for "dimensions", all dimensions are included.
    """
    
    # Specify context. 
    context1 = f"He {context} at {firm_name}"
    context2 = f"She {context} at {firm_name}"
    
    # Get embeddings.
    context1 = pd.DataFrame(func.analyzeWord(firm_name, context1 ,numberPolar = numPolar)).transpose()
    context1 = context1.rename(columns={2: 'Value 1'})
    context1.drop([0, 1], axis = 1, inplace = True)
    
    context2 = pd.DataFrame(func.analyzeWord(firm_name, context2 ,numberPolar = numPolar)).transpose()
    context2 = context2.rename(columns={2: 'Value 2'})
    
    # Merge embeddings. 
    df_merged = pd.merge(context1, context2, left_index=True, right_index=True)
    df_merged["female-male"] = df_merged["Value 2"] - df_merged["Value 1"]
    df_merged["tuple"] = list(zip(df_merged[0], df_merged[1]))
    #df = df_merged.sort_values(by=['female-male'], ascending=False)
    
    if dimensions == "all":
        return df_merged
    
    else: 
        df = df_merged[(df_merged["tuple"].isin(dimensions))]
        return df 




def plot_firm_comparison(embeddings, firm_name, context, top_n): 
    
    """Plots the male and female vector for the top_n POLAR dimensions ranked by difference between 
    male and female vector.
    """
    
    embeddings = embeddings.sort_values(by=['delta'], ascending=False).head(top_n)
    
    my_range=range(1,len(embeddings.index)+1)
    ax = plt.axes()
    plt.rcParams["figure.figsize"] = (4,6) 
    plt.scatter(embeddings["Value 1"], my_range, color='#3AA8F3', label="He", zorder = 5, s = 50)
    plt.scatter(embeddings["Value 2"], my_range, color='fuchsia', label="She", zorder = 5, s = 50)
    plt.legend()
    ax.hlines(y=my_range, color='#424242', alpha=1, linewidth = 1.5, xmin=embeddings["Value 1"], xmax=embeddings["Value 2"])
    ax.axvline(x = 0, color = 'grey', linewidth = 1.2, linestyle = "dashed", alpha = 0.7)
    maximum = max(embeddings["Value 1"].max(), embeddings["Value 2"].max(), abs(embeddings["Value 1"].min()), abs(embeddings["Value 2"].min())) * 1.1
    plt.xlim(-maximum,maximum)
    plt.yticks(my_range, embeddings[0])
    plt.title(f"He vs. She {context} at {firm_name.capitalize()}", weight = "bold")
    plt.xlabel('Value')
    plt.ylabel('Dimension')
    ax2 = ax.twinx()
    plt.yticks(my_range, embeddings[1])
    ax2.set_ylim(ax.get_ylim())
    ax.grid(False)
    ax2.grid(False)
    #plt.savefig(f'{company1}_vs_{company2}.png', bbox_inches = "tight")
    plt.show()


def check_significance(p_value): 
    if p_value < 0.05: 
        stat_res = "Significant Difference"
    else: 
        stat_res = "No Significant Difference"
        
    return stat_res



def get_dimension_distribution(considered_companies, antonym_pair, nouns_sample, path): 
    
    result = read_embedding_values(considered_companies, antonym_pair, path)
    rand_result = read_embedding_values(nouns_sample, antonym_pair, path)
    

    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    
    gs = GridSpec(2, 1, figure=fig)
    fig.suptitle(f"Distributions for {str(antonym_pair)}", weight="bold")

    # Plot male and female distribution 
    stat, p_value = ttest_ind(result["value_male"], result["value_female"])
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title(f"Male vs. Female Firm Distribution ({check_significance(p_value)}, p-value = {round(p_value, 5)})", fontsize=9)
    sns.distplot(result["value_male"], ax = ax1, label = "Male Context", color = '#3AA8F3')
    sns.distplot(result["value_female"], ax = ax1, label = "Female Context", color = 'fuchsia')
    plt.axvline(0, color = "black", ls = "--")
    plt.axvline(result["value_male"].mean(), color = "#3AA8F3")
    plt.axvline(result["value_female"].mean(), color = "fuchsia")
    plt.legend() 
    
    # Plot difference
    stat, p_value = ttest_ind(result["difference"], rand_result["difference"])
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title(f"Corporate vs. Random Difference ({check_significance(p_value)}, p-value = {round(p_value, 5)})", fontsize=9)
    sns.distplot(result["difference"], ax = ax4, label = "Firm difference", color = "cyan")
    sns.distplot(rand_result["difference"], ax = ax4, label = "random difference", color = "grey")
    plt.axvline(0, color = "black", ls = "--")
    plt.axvline(rand_result["difference"].mean(), color = "grey")
    plt.axvline(result["difference"].mean(), color = "cyan")
    plt.legend()
    
    plt.savefig(f'./distribution_plots/{antonym_pair}.png', bbox_inches = "tight")



def get_dimension_pvalues(considered_companies, antonym_pair, nouns_sample, path): 
    
    result = read_embedding_values(considered_companies, antonym_pair, path)
    rand_result = read_embedding_values(nouns_sample, antonym_pair, path)
    
    stat1, p_value1 = ttest_ind(result["value_male"], result["value_female"])
    stat2, p_value2 = ttest_ind(result["difference"], rand_result["difference"])
    
    res1 = round(p_value1, 5)
    res2 = round(p_value2, 5)
    
    return [res1, res2]




def read_embedding_values(embedding_names, antonym_pair, path): 
    
    result = pd.DataFrame()
    result["companies"] = embedding_names
    result["value_male"] = 0
    result["value_female"] = 0
    result["difference"] = 0

    for embedding in embedding_names: 
        comparison = pd.read_csv(f"{path}{embedding}.csv")

        comparison = comparison.groupby("tuple", as_index = False).mean() # Account for duplicates.         
        value_male = float(comparison[comparison["tuple"] == antonym_pair]["average_male"])
        value_female = float(comparison[comparison["tuple"] == antonym_pair]["average_female"])
        difference = value_female - value_male
        result["value_male"].loc[result["companies"] == embedding] = value_male
        result["value_female"].loc[result["companies"] == embedding] = value_female
        result["difference"].loc[result["companies"] == embedding] = difference
    
    return result    



def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor







def get_average_embedding(target_list, context): 
    
    for entry in target_list: 
        
        names = entry.split(" ")
        
        # Get contextual embedding for each part of the name.  
        embeddings = pd.DataFrame()
        for name in names: 
            embeddings[[f"{name}_male", f"{name}_female", "tuple"]] = get_firm_embeddings(name, context, 111, "all")[["Value 1", "Value 2", "tuple"]]

        # Take average of different name parts. 
        embeddings["average_male"] = embeddings.loc[:, embeddings.columns.str.endswith('_male')].mean(axis=1).round(7)
        embeddings["average_female"] = embeddings.loc[:, embeddings.columns.str.endswith('_female')].mean(axis=1).round(7)

        # Calculate delta / bias. 
        embeddings["female-male"] = (embeddings["average_female"] - embeddings["average_male"]).round(7)

        # Store for faster import later. 
        embeddings.to_csv(f"./averaged_embeddings/{entry}.csv")