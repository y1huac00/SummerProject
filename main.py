import pandas as pd
import os
import glob

directory = '/Users/chenyihua/desktop/All_forums/*'
files = [filename for filename in glob.glob(directory)]
destination = '/Users/chenyihua/desktop/Overall/'
files.sort()
df = pd.DataFrame(columns=['filename', 'genus_species'])
for i in files:
    genus_species = os.path.basename(i)
    files2 = [os.path.basename(filename) for filename in glob.glob(i+'/images/*')]
    df2 = pd.DataFrame(files2, columns=['filename'])
    df2['genus_species'] = genus_species
    df = df.append(df2, ignore_index=True)
    for j in files2:
        os.rename(i + '/images/' + j, destination+j)

df.to_csv(destination+'input.csv', index=False)

