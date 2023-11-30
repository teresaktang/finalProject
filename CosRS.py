#Cosine similarity recommendation system

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import openpyxl
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

# replace "Not Calibrated" rating with NaN
data['Course Rating'] = data['Course Rating'].replace("Not Calibrated", np.nan)

df = data[['Course Name', 'keywords', 'Course Rating']]
df['keywords'] = data['keywords'].str.replace(',', ' ')
df['keywords'] = data['keywords'].apply(lambda x: x.lower())
df['Course Name'] = data['Course Name'].str.replace(',', ' ')
# replaces commas with white space in course name column

cv = CountVectorizer(max_features=5000, stop_words='english')
#keywords to array
vectors = cv.fit_transform(df['keywords']).toarray()


# steming preprocessing keywords
def stem(text):
    y = []

    for x in text.split():
        y.append(ps.stem(x))

    return " ".join(y)


df['keywords'] = df['keywords'].apply(stem)
similarity = cosine_similarity(vectors)


# limited to course name
def recommend(course):
    course_index = df[df['Course Name'] == course].index[0]
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:100]
    result = []
    for c in course_list:
        course_name = (df['Course Name'].iloc[c[0]])
        course_rating = (df['Course Rating'].iloc[c[0]])
        similarity_score = c[1]
        result.append((course_name, similarity_score, course_rating))
        # print(course_name, "Similarity score: " + str(similarity_score))

    return result


# test
recommendation = (recommend('Python Programming Essentials'))

df_recommendations = pd.DataFrame(recommendation, columns=['Recommended Course', 'Similarity Score', 'Course Rating'])

# export dataframe to Excel file
df_recommendations.to_excel('cosineRecommendations.xlsx', index=False)