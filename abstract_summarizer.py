# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:25:10 2020

@author: Biswabhusan Pradhan
"""
from GoogleNews import GoogleNews
import nltk
from newspaper import Article
from newspaper import fulltext
from datetime import datetime
from flask import Flask

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

app = Flask(__name__)
nltk.download('punkt')

googlenews = GoogleNews()
news_sources = ['https://timesofindia.indiatimes.com/']
                
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

def extractSummary(text):
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    print ("original text preprocessed: \n", preprocess_text)
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=100,
                                    max_length=150,
                                    early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print('******************** Summary *********************')
    print(output)
    print('*********************************************')
    return output


def search_news(search_text, news_source, start_date, end_date):
    googlenews = GoogleNews(start=start_date,end=end_date)
    googlenews.search(search_text + ' ' + news_source)
    googlenews.getpage(1)
    
    search_result = googlenews.result()
    
    for news in search_result:
        article = Article(url=news['link'], language='en', memoize_articles=True)
        article.download()
        article.parse()
        article.nlp()
        fetched_news = {
                'headline' : article.title, 
                'content' : article.text, 
                'summary' : extractSummary(article.text),
                'tags' : article.keywords, 
                'url' : news['link'], 
                'language' : 'en', 
                'source' : news_source,
                'publisher' : news_source,
                'publishDate' : article.publish_date if article.publish_date else datetime.today(),
                'isDeleted' : False,
                'category' : 'Others'
                }

def start_analysis():
    for source in news_sources:
        search_news('missiles in china', source, '07/14/2020', '07/17/2020')
        
start_analysis()
