# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:34:22 2020

@author: SDRC_DEV
https://huggingface.co/valhalla/t5-small-qa-qg-hl
"""
from transformers import *
from pipelines import pipeline

tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")

model = AutoModelWithLMHead.from_pretrained("valhalla/t5-small-qa-qg-hl")

nlp = pipeline("multitask-qa-qg")

nlp({
    "question": "Where we are using machine learning?",
    "context": "Machine learning (ML) is the study of computer algorithms that improve automatically through experience. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks."
})
