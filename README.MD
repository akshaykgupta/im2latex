# im2latex
Deep Learning Course, Fall 2016

The project aims to generate Latex source code given the image of a Latex formula. The model we use is a convolutional encoder followed by a GRU decoder with attention applied to the image representation, and beam search to predict final output. The model is inspired from Show, Attend and Tell (Xu, Kelvin et. al., 2016).

We used the 1m2latex-100k dataset from the Harvard NLP Group. This is one of the projects listed at OpenAI's requests for research page. 