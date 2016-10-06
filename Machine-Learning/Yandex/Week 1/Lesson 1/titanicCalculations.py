import sys
import pandas
import re

data = pandas.read_csv('D:\\Dropbox\\Study\\Programming\\Coursera\\Week 1\\Lesson 1\\data\\titanic.csv', index_col='PassengerId')


#1 count of males and females
sex_count = data['Sex'].value_counts()
(males_count, female_count) = sex_count['male'], sex_count['female']
print('#1')
print(str(males_count) + ' ' + str(female_count))


#2 percent of survived passenger
total_count = data['Survived'].count()
survived = data['Survived'].where(data['Survived'] > 0).value_counts()[1]
percent_survived = round((survived/total_count)*100, 2)
print('#2')
print(percent_survived)


#3 percent of passengers of the first class

first_class = data['Pclass'].where(data['Pclass'] == 1).value_counts()[1]
percent_first_class = round((first_class/total_count)*100, 2)
print('#3')
print(percent_first_class)

#4 median and mean of passengers age
age_median = data['Age'].median()
age_mean =  data['Age'].mean()
print('#4')
print(str(age_mean) + ' ' + str(age_median))


#5 Pearson correlation between SibSp and Parch

corr = data['Parch'].corr(data['SibSp'], method='pearson')
print('#5')
print(corr)


#6 Popular Name 
names = data['Name'].where(data['Sex']  == 'female')
names_filtered = filter((lambda x : x == x), names)
name_mapped =  map((lambda x: x[x.find(',') + 1: ].strip()), names_filtered)

regex=re.compile("(?<=[(])\w*|(?<=(Miss.\s))\w*")
result_list = list([m.group(0) for l in name_mapped for m in [regex.search(l)] if m])
result = max(set(result_list), key=result_list.count)

print('#6')
print(result)