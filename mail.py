import pandas as pd
import numpy as np

data = pd.read_csv(r'emails.csv')

print(data.head())

ones = []
ones_data = []
zeros = []
zeros_data = []

print(data['spam'][0])

for i in range(3000):

    if (data['spam'][i]) == "1":
        ones.append(1)
        ones_data.append(data['text'][i])
    elif (data['spam'][i]) == "0":
        zeros.append(0)
        zeros_data.append(data['text'][i])

#print(len(ones), len(ones_data), len(zeros), len(zeros_data))

email_set = []
email_label = []

for i in range(1368):
    email_set.append(ones_data[i])
    email_label.append(ones[i])
    email_set.append(zeros_data[i])
    email_label.append(zeros[i])

# print(len(email_set), len(email_label))

# print(email_set[0:3], email_label[0:3])

df = pd.DataFrame({'text': email_set, 'spam': email_label})

df.to_csv('emaildata.csv')
