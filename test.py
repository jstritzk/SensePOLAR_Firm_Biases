import bertFuncs as func
from functions import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging
logging.set_verbosity_error()
from tqdm.notebook import tqdm
import numpy as np
from scipy.stats import ttest_ind
from collections import defaultdict
from nltk.corpus import wordnet as wn

from scipy.stats import ttest_ind
import math

import pathlib

import multiprocess as mp


def get_embedding_method_1(context, embedded_word, numPolar, method): 
    
    
    # Create dataframe for final embedding. 
    embedding = pd.DataFrame()
    
    
    # Split multi-word names to get average embedding. 
    all_words = embedded_word.split(" ")
    
    # Define full context. 
    male_context = f"He {context} at {embedded_word}"
    female_context = f"She {context} at {embedded_word}"
    
    for word in all_words: 

        # Get embeddings for male and female context.
        male_embedding = pd.DataFrame(func.analyzeWord(word, male_context ,numberPolar = numPolar)).transpose()
        male_embedding = male_embedding.rename(columns={2: 'Value_Male'})
        male_embedding.drop([0, 1], axis = 1, inplace = True)

        female_embedding = pd.DataFrame(func.analyzeWord(word, female_context ,numberPolar = numPolar)).transpose()
        female_embedding = female_embedding.rename(columns={2: 'Value_Female'})

        # Merge embeddings. 
        df_merged = pd.merge(male_embedding, female_embedding, left_index=True, right_index=True)
        df_merged["male-female"] = df_merged["Value_Male"] - df_merged["Value_Female"]
        df_merged["tuple"] = list(zip(df_merged[0], df_merged[1]))
    
        embedding[[f"{word}_male", f"{word}_female", "tuple"]] = df_merged[["Value_Male", "Value_Female", "tuple"]]
        
    # Take average of different name parts. 
    embedding["average_male"] = embedding.loc[:, embedding.columns.str.endswith('_male')].mean(axis=1).round(7)
    embedding["average_female"] = embedding.loc[:, embedding.columns.str.endswith('_female')].mean(axis=1).round(7)

    # Calculate delta / bias. 
    embedding["male-female"] = (embedding["average_male"] - embedding["average_female"]).round(7)

    # Store for faster import later. 
    embedding.to_csv(f"./{method}/{embedded_word}_{context}.csv")




if __name__ == '__main__':
    
    for company in ["Apple", "Philip Morris"]: 
        mp.Pool().starmap(get_embedding_method_1, [("works", company, 111, "method1"), ("manages", company, 111, "method1")])