import pandas as pd
from functools import reduce
import re
import nltk
import string
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib as mpl
sns.set_theme()
mpl.rcParams["text.usetex"]=True
mpl.rcParams["lines.linewidth"]=0.9
mpl.rcParams["figure.dpi"]=400


def lowercase(df, text='text'):
    df[text] = df[text].apply(lambda x: x.lower())
    return df

def remove_at_mentions(df, text='text'):
    df[text] = df[text].apply(lambda x: re.sub(r'@\w+', '__ATMENTION__', x))
    return df

def remove_hashtags(df, text='text'):
    df[text] = df[text].apply(lambda x: re.sub(r'#\w+', '__HASHTAG__', x))
    return df


def remove_urls(df, text='text'):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    df[text] = df[text].apply(lambda x: url_pattern.sub('__URL__', x))
    return df

def whitespace(df, text='text'):
    pattern1 = re.compile(r'\s+')
    pattern2 = re.compile(r'(^\s+)|(\s+$)')
    df[text] = df[text].apply(lambda x: pattern1.sub(' ',x).strip())
    return df

def punctuation(df, text="text"):
    import unicodedata
    punctuation_cats = set(['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'])
    def simple_punctuation(t):
      return ''.join((x if unicodedata.category(x) not in punctuation_cats or x == "_" else "!") for x in t)
    punct = string.punctuation.replace("_","")
    tra = "".maketrans(punct, '\t'*len(punct))
    df[text] = df[text].apply(lambda x: simple_punctuation(x).translate(tra).replace(r"\t+", " __PUNCT__ "))
    return df


def preprocess(df, text='text'):
    df = lowercase(df, text)
    df = remove_hashtags(df, text)
    df = remove_at_mentions(df, text)
    df = remove_urls(df, text)
    df = punctuation(df, text)
    df = whitespace(df, text)
    return df



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
