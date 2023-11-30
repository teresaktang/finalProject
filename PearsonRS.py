# recommendation system pearson coefficient
import numpy as np

import pandas as pd
import random

from scipy.spatial import distance

# dataset 1 of courses
data1 = pd.read_csv("Coursera.csv")
# dataset 2 of 100k course reviews
data2 = pd.read_csv("reviews_by_course.csv")
# Select Columns from datasets
# Course rating is a float
# user review rating is a number in range 1-5 for how much the user liked the course
data1 = data1[['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Course Rating']]
data2 = data2[['CourseId', 'Rating']]
# Create users that uses several courses
np.random.seed(123)
data2['UserId'] = np.random.randint(1, 1001, size=len(data2))
# preprocess dataset

# assign unique index to duplicate values in datset 2
data2['newcourse_id'] = data2.groupby(['CourseId']).ngroup()

# print(data2[:30])

# merge reviews and courses
data = pd.merge(data1, data2, left_index=True, right_on='newcourse_id')
# print(data[:30])
# print(data.columns)

# print((data.head()))

# average user rating

avg_review = data.groupby(['Course Name'])['Rating'].mean().reset_index().round(1)

# print(avg_review[:3000])
# print(avg_review.columns)
# print(avg_review[avg_review['Course Name'] == 'Python Programming Essentials']['Rating'])

# change dataset to matrix
coursereview = data.pivot_table(index=['UserId'], columns=['Course Name'], values='Rating')


# print(coursereview.columns)
# print(coursereview.index)
# print(coursereview.head())

# pearson coefficient correlation
# takes two courses and returns correlation

def correlation(course, avg_review):
    corr = coursereview.corr()[course].sort_values(ascending=False).iloc[:100]
    avg_rating = avg_review.loc[avg_review['Course Name'] == course, 'Rating'].item()
    recommendation = corr.to_frame().join(avg_review.set_index('Course Name'), how='left')
    recommendation['Rating'] = recommendation['Rating'].fillna(avg_rating)
    recommendation = recommendation.reset_index().rename(columns={'index': 'Course Name'})
    recommendation.columns = ['Course Name', 'Correlation', 'Average Rating']
    return recommendation
    # print(coursereview.corr()[course].sort_values(ascending=False).iloc[:20]  )


result = (correlation('Python Programming Essentials', avg_review))
result.to_excel('pearson.xlsx', index=False)
print(result)
# Write A Feature Length Screenplay For Film Or Television
