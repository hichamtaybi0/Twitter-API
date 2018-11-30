import logging

import warnings
from multiprocessing import freeze_support

warnings.filterwarnings("ignore")
# Plotting tools
import gensim  # don't skip this

from gensim.models.coherencemodel import CoherenceModel

from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary


#Set up logging
logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)
logging.debug("test")


import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

# English stop words list
from nltk.corpus import stopwords
stopwords_en = stopwords.words('arabic')


import pandas as pd

documents=[""]
df = pd.read_csv('data.csv')
for te in df['text']:
    documents.append(str(te))


# remove common words and tokenize
texts = [[word for word in document.lower().split() if word not in stopwords_en]
         for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1


texts = [[token for token in text if frequency[token] > 1] for text in texts]


dictionary = Dictionary(texts)
print("\n --- dictionary \n", dictionary)
bow_vectors = [dictionary.doc2bow(text) for text in texts]


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        print(coherencemodel.get_coherence())
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

freeze_support()
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=bow_vectors, texts=texts, start=2, limit=40, step=1)

# Show graph
import matplotlib.pyplot as plt
limit = 40; start=2; step = 1
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
