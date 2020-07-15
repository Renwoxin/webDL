import gensim
from gensim.models import Word2Vec

sentences = [['first', 'sentence'], ['second', 'sentence']]
model = gensim.models.Word2Vec(sentences, min_count=10)
print(model['first'])
# model = Word2Vec(sentences, min_count=10)
# print(model)