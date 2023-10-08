# Necessary imports, reading data
import pandas as pd
import numpy as np
df = pd.read_csv('features_full.csv')

# This code aims to decide which features to keep, since 1000 features is far too noisy
to_keep = []

# Find the most frequent features present in each language
columns = ['Finnish','French','English','Latin','Spanish']
for lang in columns:
    freq = df[df[lang]==1].drop(columns=[lang]).mean().sort_values()
    ug = freq[freq.index.map(len) == 1][-20:]
    bg = freq[freq.index.map(len) == 2][-20:]
    [to_keep.append(x) for x in ug.index.tolist()]
    [to_keep.append(x) for x in bg.index.tolist()]
    print("Most frequent unigrams in " + lang + ":")
    print(ug)
    print("Most frequent bigrams in " + lang + ":")
    print(bg)

# Get the correlations between languages and all features
corrs = {}
for col in df.columns:
    corr = df[columns].corrwith(df[col])
    corrs[col] = corr
dfc = pd.DataFrame(corrs).T
lang_corr = dfc.iloc[-5:, -5:]
ngram_corr = dfc.iloc[:-5,:]

# Now, we can separate features that distinguish a language best from the others
for lang in ngram_corr.columns:
    diff = ngram_corr[lang] - ngram_corr.drop(columns=[lang]).sum(axis=1)
    sort = diff.sort_values()
    [to_keep.append(x) for x in sort[-20:].index.tolist()]
    [to_keep.append(x) for x in sort[:20].index.tolist()]
    print(lang + " best classifiers:")
    print(sort[-10:])
    print(lang + " worst classifiers:")
    print(sort[:10])

# Filter non-unique values
to_keep = list(set(to_keep))
print(to_keep)

# Make a new dataframe and write to csv
features_filtered = df[to_keep + columns]
features_filtered.to_csv('features.csv', index=False)
print(features_filtered)

# Now we have a list of about 100 features to use for our learning task
    




