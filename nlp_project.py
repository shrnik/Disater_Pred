#import keras
#import nltk
import pandas as pd 
import numpy as np 
import re
import codecs

#input_file = codecs.open('socialmedia-disaster-tweets-DFE.csv',"r",encoding='utf-8',errors='replace')
#output_file = open("socialmedia_relevant_cols_clean.csv", "w")


questions = pd.read_csv("socialmedia-disaster-tweets-DFE.csv",encoding='latin-1')

print(questions.columns)
columns=['text', 'choose_one']

data = questions[columns]
print(data.head())

print(data.choose_one.value_counts())

data['num_label'] = data.choose_one.map({'Not Relevant':0,'Relevant':1,'Can\'t Decide':2})

print(data.groupby('num_label').count())

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

new_data = standardize_text(data,'text')

print(new_data.tail())

new_data.to_csv("clean_data.csv")


