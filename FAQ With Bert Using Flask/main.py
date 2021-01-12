
from flask import Flask,render_template,request
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

   
def clean_sentence(sentence, stopwords=False):
    
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    
    if stopwords:
        sentence = remove_stopwords(sentence)
        
    return sentence   

def get_cleaned_sentences(df, stopwords=False):
    cleaned_sentences=[]
    
    for index, row in df.iterrows():
        cleaned = clean_sentence(row["questions"], stopwords)
        cleaned_sentences.append(cleaned)
        
    return cleaned_sentences

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/Result")
def result():
    return render_template("result.html")

@app.route("/Result1",methods = ['POST'])
def result1():
    
    df = pd.read_csv("FAQ.csv")
    bert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    cleaned_sentences = get_cleaned_sentences(df, stopwords=False)
    sent_bertphrase_embeddings=[]
    for sent in cleaned_sentences:
        sent_bertphrase_embeddings.append(bert_model.encode([sent]))
    min_similarity = 0.1
    
    question = request.form['qa']
    
    question = clean_sentence(question, stopwords=False)

    question_embedding = bert_model.encode([question])

    max_sim = -1;
    index_sim = -1
    
    # cosine similarity
    for index, faq_embedding in enumerate(sent_bertphrase_embeddings):
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
        #print(index, sim, sentences[index])
        
        if sim > max_sim:
            max_sim = sim
            index_sim = index
    
    similarity = (f"\nSimilarity: {max_sim}")
    if max_sim > min_similarity:
        retrived = df.iloc[index_sim, 0]
        ans = df.iloc[index_sim, 1]
    else:
        print("\nCouldn't find a relevant answer to your question.\nPlease write us a mail of your query.\nOur experts will reach out to you ASAP.\n")


    return render_template("result1.html",similarity=similarity,retrived=retrived,ans=ans)

if __name__ == '__main__':
   app.run()