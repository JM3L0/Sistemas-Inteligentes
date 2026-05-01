import pandas as pd

salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')

salary.columns

y = salary['Salary']
X = salary[['Years_Experience']]

from sklearn.linear_model import train_test_split