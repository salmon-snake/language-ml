import re

folder='langdata/'
# in_file -> cleaning capturing group -> out_file
def clean(in_path, out_path, line_pattern):
    # open the file, read the data
    with open(folder+in_path, 'r', encoding='utf-8') as file:
        data = file.read().lower()
    # find all word matches    
    matches = re.findall(line_pattern,data)
    # write matches to the output file
    with open(folder+out_path, 'w', encoding='utf-8') as file:
        for match in matches:
            file.write(match + '\n') 
read_paths = ['finnish_dirty.txt', 'spanish_dirty.txt', 'latin_dirty.txt', 'french_dirty.txt', 'english_dirty.txt']
write_paths = ['finnish.txt', 'spanish.txt', 'latin.txt', 'french.txt', 'english.txt']

# Grabs all words in these files. Words not part of the dataset are also grabbed, but they still belong to the target language
for rp,wp in zip(read_paths, write_paths):
    clean(rp, wp, r'([a-zéèêëàáâîíóôûúüçæœùïÿñôäöå\'\-]+)')






