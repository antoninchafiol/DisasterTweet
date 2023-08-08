# Repository summary

This repository is used to make and understand concepts of NLP through multiple examples/models and data processing method.

All methods used will be written in a document when I'll consider to have got enough information to make a useful note.
(This will likely be after the implementation of tranformers networks)

## Specifications
To clarify, my ressources to run the code come from the shadow computer most used for cloud gaming, here's the specs though:

- Intel Xeon CPU E5-2678
- 12 GB RAM
- Nvidia Quadro P5000

## Branches

- main branch is used as a meddle of multiples models/methods 
- tranformer branch is used exclusively for transformer network and to improve these 


## Current Performance

### Main Branch
The main metric is F1 Score

Using a simple LSTM with only a coarse usage of "CountVectorizer": 
    Train: ~ 90%+ | Eval: ~70%
Using LSTM with Embedded layer and glove pretrained weights:
    Train: ~ 95%+ | Eval: ~40%

Might come back on these later (After transformer built)

### Tranformer Branch

WIP
