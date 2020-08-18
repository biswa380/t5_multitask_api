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

    
    
