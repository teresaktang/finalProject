# Recommendation system
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import nltk
from nltk.stem.porter import PorterStemmer
from scipy import spatial

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer()
pd.options.mode.chained_assignment = None

# dataset
data = pd.read_csv("Coursera.csv")

# Select Columns from dataset
data = data[['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Course Rating']]
# preprocess dataset to create keywords
# replace white space and other punctuation with commas
data['Course Name'] = data['Course Name'].str.replace(' ', ',').str.replace(':', '').str.replace(',,', ',')
data['Course Description'] = data['Course Description'].str.replace(' ', ',').str.replace(',,', ',').str.replace('_',
                                                                                                                 '') \
    .str.replace(':', '').str.replace('(', '').str.replace(')', '')
data['Skills'] = data['Skills'].str.strip('[]')

data['keywords'] = data['Course Name'] + data['Difficulty Level'] + data['Course Description'] + data['Skills']

difficultyList = []
for index, row in data.iterrows():
    levels = row["Difficulty Level"]
    if levels not in difficultyList:
        difficultyList.append(levels)


# print(keywordList[ :10])
# check the unique keywords
def binary(difficulty_list):
    binaryList = []

    for level in difficultyList:
        if level in difficulty_list:
            binaryList.append(1)
        else:
            binaryList.append(0)

    return binaryList


data['level_bin'] = data['Difficulty Level'].apply(lambda x: binary(x))

for i, j in zip(data['keywords'], data.index):
    wordList = []
    wordList = i
    data.loc[j, 'keywords'] = str(wordList)
data['keywords'] = data['keywords'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
data['keywords'] = data['keywords'].str.split(',')
for i, j in zip(data['keywords'], data.index):
    wordList = []
    wordList = i
    wordList.sort()
    data.loc[j, 'keywords'] = str(wordList)
words_list = []
for index, row in data.iterrows():
    levels = row["keywords"]

    for level in levels:
        if level not in words_list:
            words_list.append(level)
            data['words_bin'] = data['keywords'].apply(lambda x: binary(x))

df = data[['Course Name', 'keywords', 'Course Rating', 'level_bin', 'words_bin']]
df['keywords'] = data['keywords'].str.replace(',',' ')
df['keywords'] = data['keywords'].apply(lambda x: x.lower())
df['Course Name'] = data['Course Name'].str.replace(',', ' ')

# replaces commas with white space in course name column
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['keywords']).toarray()


def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


df['keywords'] = df['keywords'].apply(stem)
similarity = cosine_similarity(vectors)


def recommend(course):
    course_index = df[df['Course Name'] == course].index[0]
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]

    for i in course_list:
        print(df['Course Name'].iloc[i[0]])


recommend('Python Programming Essentials')
