import bm25s
import jieba
import re

# import Stemmer  # optional: for stemming

def chinese_tokenizer(text):
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5\w]', ' ', text)
    
    tokens = jieba.cut(cleaned_text)
    tokens = [tok.strip() for tok in list(tokens) if tok.strip()!=""]
    return tokens

# Create your corpus here
corpus = [
"猫是一种猫科动物，喜欢咕噜咕噜叫",
"狗是人类最好的朋友，喜欢玩耍",
"鸟是一种美丽的动物，会飞",
"鱼是一种生活在水中的生物，会游泳",
]
# optional: create a stemmer
# stemmer = Stemmer.Stemmer("english")

# Tokenize the corpus and only keep the ids (faster and saves memory)
# corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", split_fn=chinese_tokenizer)


# Create the BM25 model and index the corpus
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# Query the corpus
query =  "鱼会像猫一样咕噜咕噜叫吗？"

# query_tokens = bm25s.tokenize(query, stemmer=stemmer)
query_tokens = bm25s.tokenize(query, split_fn=chinese_tokenizer)


# Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

# You can save the arrays to a directory...
retriever.save("animal_index_bm25")

# You can save the corpus along with the model
retriever.save("animal_index_bm25", corpus=corpus)

# ...and load them when you need them
import bm25s
reloaded_retriever = bm25s.BM25.load("animal_index_bm25", load_corpus=True)
# set load_corpus=False if you don't need the corpus