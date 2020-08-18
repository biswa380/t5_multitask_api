# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:25:28 2020

@author: SDRC_DEV
"""
import nltk
from flask import Flask

from transformers import *
from pipelines import pipeline
import torch

app = Flask(__name__)
nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")

model = AutoModelWithLMHead.from_pretrained("valhalla/t5-small-qa-qg-hl")

nlp = pipeline("multitask-qa-qg")

@app.route("/getAnswer/<context>/<question>")
def generateAnswer(context, question):
    return nlp({
    "question": question,
    "context": context
    })

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('tromedlov/t5-small-cnn')
device = torch.device('cpu')

@app.route("/summarize/<text>") 
def extractSummary(text):
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
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
    
    
