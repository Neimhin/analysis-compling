import pandas as pd
from functools import reduce



def ordering(df):
  ordering=[df.columns[i] for i in [4, 1, 3, 0, 2, 5, 6]]
def reorder(df):
  from scipy.cluster import hierarchy
  ordering=[df.columns[i] for i in [4, 1, 3, 0, 2, 5, 6]]
  print(ordering)
  return df[ordering]

def cmap():
  import seaborn as sns
  cmap = sns.diverging_palette(h_neg=220,h_pos=45,s=74,l=73,sep=10,n=14,center="light",as_cmap=True)
  return cmap

def calculate_ttr(text):
  from nltk.tokenize import word_tokenize
  from collections import Counter
  tokens = word_tokenize(text)
  types = Counter(tokens)
  ttr = len(types) / len(tokens) if len(tokens) > 0 else 0
  return ttr

def isTerminal(pt):
  return len(pt.children) == 0

def node2string(node):
  return label(node) + "_" + "_".join([label(n) for n in node.children])

def node_degree(node):
  return len(node.children)

def load_parses():
  return pd.read_pickle("data/parses.pickle")

def load(s):
  return pd.read_csv(s,delimiter="\t")

def saveid2x(df,s,f):
  df[["id",s]].to_csv(f,sep="\t",index=False)

def sentence_node_degree_list(s):
  return [node_degree(s)] + reduce(lambda a,x:a+x, list([sentence_node_degree_list(sub) for sub in s.children]),[])

def sentence_node_type_list(s):
  return [node2string(s)] + reduce(lambda a,x:a+x, list(filter(None,[sentence_node_type_list(sub) for sub in s.children])),[])

def label(node):
  if isTerminal(node):
    return "TERMINAL"
  return node.label
