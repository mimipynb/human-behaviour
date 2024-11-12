"""Data Preparation utils for datasets. This was used to train small models to handle chat sessions."""

import pandas as pd 
import numpy as np 

big5labels = ['response', 'personality']
big5target = 'personality'
big5tags = {
    'extraversion': 0,
    'agreeableness': 1,
    'neuroticism': 2,
    'openness': 3,
    'conscientiousness': 4
 }

mbtitags = {'INFJ': 0,
 'ENTP': 1,
 'INTP': 2,
 'INTJ': 3,
 'ENTJ': 4,
 'ENFJ': 5,
 'INFP': 6,
 'ENFP': 7,
 'ISFP': 8,
 'ISTP': 9,
 'ISFJ': 10,
 'ISTJ': 11,
 'ESTP': 12,
 'ESFP': 13,
 'ESTJ': 14,
 'ESFJ': 15}

def _load_big5_data():
    """Loads the dataset from hugging face for big 5"""
    from datasets import load_dataset
    columns = {
        'Target Personality': 'personality',
        'Edit Topic': 'topic',
        'Question': 'question',
        'Answer': 'response'
    }
    data = load_dataset("Navya1602/Personality_dataset")['train']
    df = data.to_pandas()
    df.columns = df.columns.map(columns)
    df['persona'] = df['personality'].map(big5tags)
    df['topic_x'] = df['topic'].map(dict(zip(df['topic'].unique(), list(range(len(df['topic'].unique()))))))
    return df 

def _load_personality_data():
    """Loads and preprocess the personality dataset."""
    import pandas as pd

    df = pd.read_csv("hf://datasets/kl08/myers-briggs-type-indicator/mbti_1.csv")
    df['mbti'] = df['type'].map(mbtitags)
    x0_label = {i: list(i)[0] for i in df['type'].unique()}
    x1_label = {i: list(i)[1] for i in df['type'].unique()}
    x2_label = {i: list(i)[2] for i in df['type'].unique()}
    x3_label = {i: list(i)[3] for i in df['type'].unique()}

    df['x0'] = df['type'].map(x0_label)
    df['x1'] = df['type'].map(x1_label)
    df['x2'] = df['type'].map(x2_label)
    df['x3'] = df['type'].map(x3_label)
    return df 

# milkis = MilkBarn(base=base, data=df, labels=['response', 'personality'], group_target='personality')
