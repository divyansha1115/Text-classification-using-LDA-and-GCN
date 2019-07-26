import pandas as pd

df = pd.read_csv('pre_processed_data.csv')
data = list(df['0'])

all_data = ' '.join(data)

f = open("all_words.txt", "w+")
f.write(all_data)
f.close()