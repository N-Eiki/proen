import janome

import pandas as pd
bad = pd.read_csv('../data/bad/bad.csv')

wordlist = bad.word
wordlist=pd.DataFrame(wordlist)
wordlist = wordlist[~wordlist.duplicated()]

test = "プリンはとても美味しいです"
from janome.tokenizer import Tokenizer

tokenizer = Tokenizer()
t = tokenizer.tokenize(test)[0]
print(t)