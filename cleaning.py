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

in_path = 'finnish_dirty.txt'
out_path = 'finnish.txt'
pattern = r'\d ([a-zéèêëàáâîíóôûúüçæœùïÿñôäöå\'\-\_]+)'

clean(in_path, out_path, pattern)

in_path = 'spanish_dirty.txt'
out_path = 'spanish.txt'
pattern = r'\.\t([a-zéèêëàáâîíóôûúüçæœùïÿñôäöå\'\-\_]+)'

clean(in_path, out_path, pattern)

in_path = 'latin_dirty.txt'
out_path = 'latin.txt'
pattern = r'\n([a-zéèêëàáâîíóôûúüçæœùïÿñôäöå\'\-\_]+)'

clean(in_path, out_path, pattern)

in_path = 'french_dirty.txt'
out_path = 'french.txt'
pattern = r'\. ([a-zéèêëàáâîíóôûúüçæœùïÿñôäöå\'\-\_]+)'

clean(in_path, out_path, pattern)






