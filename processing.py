import pandas as pd
import sys
sys.stdout._encoding = 'utf-8'

# Cleaned text data filepaths
paths = ['finnish.txt','french.txt','english.txt','latin.txt','spanish.txt']
folder = 'langdata/'
word_sets = []
for path in paths:
    with open(folder+path,'r',encoding='utf-8') as file:
        words = set()
        for line in file:
            stripped = line.strip()
            if len(stripped) > 2:
                words.add(stripped)
        word_sets.append(words)    

all_words = list(set().union(*word_sets))

rows = [all_words]

unigrams = []
bigrams = []
for word in all_words:
    for i in range(len(word)):
        unigrams.append(word[i])
        bigrams.append(word[i:i+2])
        
unigrams = list(set(unigrams))
bigrams = list(set(bigrams))

for ug in unigrams:
    row = []
    for word in all_words:
        row.append(word.count(ug))
    rows.append(row)

for bg in bigrams:
    row = []
    for word in all_words:
        row.append(1 if bg in word else 0)
    rows.append(row)
              
for word_set in word_sets:
    row = []
    for word in all_words:
        row.append(1 if word in word_set else 0)
    rows.append(row)

columns = ['Word'] + unigrams + bigrams + ['Finnish','French','English','Latin','Spanish']

df = pd.DataFrame(rows).T
df.columns = columns

df.to_csv('features.csv',index=False)

print(df)


