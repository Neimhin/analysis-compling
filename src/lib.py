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

def is_question_number_ending(row):
  if row["forum"] != "question-written" and row["forum"] != "question-oral":
    return False
  lemmas = row["lemmas"]
  def match(t):
    return is_question_number_ending.num_re.match(t)
  if lemmas[0] == "[" and lemmas[2] == "]" and match(lemmas[1]):
    return True
  return False
is_question_number_ending.num_re = re.compile(r"\d+/\d+")

def load_df():
  return pd.read_pickle("d.pickle")


def drop_non_serial(df):
  return df.copy().drop(["parse","parse-sentences"],axis=1)

def copy_proxies(df):
  cols = [ 'id', 
      'sentence_no',
      'name',
      'party',
      'age in days',
      'forum',
      'language',
      'date',
      'topic',
      'cardinal number in debate',
      'TTR words',
      'TTR lemmas',
      'TTR lemmas clean',
      'TTR lemmas clean and bare',
      'mean-node-degree',
      'max tree depth',
      'num_node_types',
      'node-type-diversity',
      'num-nodes',
      'num-node-types',
      'sentence-length-bare',
      'sentence-length',
      'mean-word-length',
      'lexical-density-words',
      'lexical-density-lemmas',
      'lexical-density-lemmas-clean',
      'lexical-density-lemmas-clean-bare'
    ]
  return df[cols].copy()

def lexical_density(series):
  t2l = token2lexscore(series)
  return series.apply(lambda sent: np.sum([t2l[word] for word in sent]) / float(len(sent)) )
  

def token2lexscore(series):
  from gensim import corpora
  dct = corpora.Dictionary(doc for doc in series)
  
  df = pd.DataFrame([(k,dct.cfs[v]) for k,v in dct.token2id.items()],columns=["token","frequency"])
  
  df = df.sort_values("frequency")
  
  df["rank"] = df["frequency"].rank(method="dense",ascending=False)
  df["lexscore"] = np.maximum(np.log2(df["rank"]),0)
  return {
    tok: score for tok,score in zip(df["token"],df["lexscore"])
  } 

def save_df(df):
  df.to_pickle("d.pickle")

def term_frequencies(document):
  from gensim import corpora
  dct = corpora.Dictionary([word] for word in document)  
  return [dct.cfs[dct.token2id[word]] for word in document]

def lexicon_frequency(series):
  from gensim import corpora
  dictionary = corpora.Dictionary(doc for doc in series)
  return dictionary

def dotfidfs(documents):
  from sklearn.feature_extraction.text import TfidfVectorizer
  
  # Custom tokenizer that skips tokenization
  def identity_tokenizer(text):
      return text
  
  # Create an instance of TfidfVectorizer with the custom tokenizer
  vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, token_pattern=None)
  
  # Fit and transform the documents into a term-document matrix
  tdm = vectorizer.fit_transform(documents)
  
  # Get the feature names
  feature_names = vectorizer.get_feature_names_out()
  
  # Function to extract the TF-IDF scores for a document
  def extract_tfidf_scores(document_idx):
      return {
          feature_names[word_idx]: tdm[document_idx, word_idx] for word_idx in range(tdm[document_idx].shape[1])
      }
  
  # Create lists of TF-IDF scores for each document
  tfidf_scores_lists = [
      [extract_tfidf_scores(doc_idx).get(word, 0) for word in doc] for doc_idx, doc in enumerate(documents)
  ]
  return pd.Series(tfidf_scores_list)  


def tdm_df(documents):
  import pandas as pd
  from sklearn.feature_extraction.text import TfidfVectorizer
  def identity_tokenizer(text):
      return text
  vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, token_pattern=None)
  tdm = vectorizer.fit_transform(documents)
  feature_names = vectorizer.get_feature_names()
  tdm_df = pd.DataFrame(tdm.toarray(), columns=feature_names)
  
  return [ tdms[word] for (word, (i,tdms)) in list(zip(documents,tdm_df.iterrows())) ]

at_mention_re = re.compile(r'@\w+')
hashtag_re = re.compile(r'#\w+')
url_re = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
punct_re = re.compile(r'^[^\w\s]+$')
special_re = re.compile(r'__[A-Z]+__')

sub = {
        "at":         lambda x: at_mention_re.sub("__ATMENTION__",x),
        "hashtag":    lambda x: hashtag_re.sub("__HASHTAG__",x),
        "url":        lambda x: url_re.sub("__URL__",x),
        "punct":      lambda x: punct_re.sub("__PUNCT__",x),
}

def count_specials(series):
  reg = {
          "at":         re.compile(r"__ATMENTION__"),
          "hashtag":    re.compile(r"__HASHTAG__"),
          "url":        re.compile(r"__URL__"),
          "punct":      re.compile(r"__PUNCT__"),
  }
  d = {}
  for t in reg:
    count = series.apply(lambda x: int(not not reg[t].match(x))).sum() 
    d[t] = (count, count/len(series)) 
  import numpy as np
  d["total"] = np.sum([d[t][0] for t in d])
  d["total"] = (d["total"],d["total"]/len(series))
  return d

def fora(df):
    import re
    unique_categories = df['forum'].unique()
    for forum in ["twitter","speech","question","answer"]:
        reg = re.compile(forum)
        yield (forum,df[df["forum"].apply(reg.match).apply(bool)])


def remove_special(series):
  def remove_special_token(token):
    if special_re.match(token):
      return None
    return token
  def remove_specials(tokens):
    return list(filter(lambda i: i is not None, [remove_special_token(token) for token in tokens]))
  return series.apply(remove_specials)

def clean_word(word):
    word = word.lower()
    for t in sub:
        a = sub[t](word)
        if word is not a:
            return a
    return word

def clean_sentence(s):
    return [clean_word(w) for w in s]

def clean(series):
    return series.apply(clean_sentence)


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

def calculate_ttr(tokens):
  from collections import Counter
  types = Counter(tokens)
  ttr = len(types) / len(tokens) if len(tokens) > 0 else 0
  return ttr

def isTerminal(pt):
  return len(pt.children) == 0

def isPenTerminal(pt):
  return len(pt.children) == 1 and len(pt.children[0].children) == 0

def max_tree_depth(pt):
  if len(pt.children) == 0:
    return 0
  return max(list(map(lambda x: x+1,map(max_tree_depth,pt.children))))

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
  return [node_degree(s)] + reduce(lambda a,x:a+x, list([sentence_node_degree_list(sub) for sub in s.children if not isPenTerminal(sub)]),[])

def sentence_node_type_list(s):
  return [node2string(s)] + reduce(lambda a,x:a+x, list(filter(None,[sentence_node_type_list(sub) for sub in s.children if not isPenTerminal(sub)])),[])

def label(node):
  if isTerminal(node):
    return "TERMINAL"
  return node.label
