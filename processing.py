# Import required libraries
import pandas as pd
import sys
sys.stdout._encoding = 'utf-8'

# Filepath things
paths = ['finnish.txt','french.txt','english.txt','latin.txt','spanish.txt']
folder = 'langdata/'

# We have the cleaned text files, so we are appending everything into sets by language
word_sets = []
for path in paths:
    with open(folder+path,'r',encoding='utf-8') as file:
        words = []
        for line in file:
            stripped = line.strip()
            if len(stripped) > 2:
                words.append(stripped)
        word_sets.append(set(words))    

# List of all words, need a set unifier because of some overlap
all_words = list(set().union(*word_sets))

# Rows represent words, and columns represent features. There are some transpose shenanigans .
rows = [all_words]

# Collects all unigrams and bigrams, then filters out duplicates for data integrity
grams = []
for word in all_words:
    for i in range(len(word)):
        grams.append(word[i])
        grams.append(word[i:i+2])
grams = list(set(grams))
unigrams = [x for x in grams if len(x) == 1]
bigrams = [x for x in grams if len(x) == 2]

# Returns the count for each unigram in each word
for ug in unigrams:
    row = []
    for word in all_words:
        row.append(word.count(ug))
    rows.append(row)

# Returns a simple membership test for each bigram, for each word
for bg in bigrams:
    row = []
    for word in all_words:
        row.append(1 if bg in word else 0)
    rows.append(row)

# Langauge membership classification, this becomes our label vector
for word_set in word_sets:
    row = []
    for word in all_words:
        row.append(1 if word in word_set else 0)
    rows.append(row)

# Building the correct names for our column vector
columns = ['Word'] + unigrams + bigrams + ['Finnish','French','English','Latin','Spanish']

# Creates a dataframe from our processed features and writes it to a file
df = pd.DataFrame(rows).T
df.columns = columns
df.drop(columns=['Word']).to_csv('features_full.csv',index=False)

# Sample output
print(df)

