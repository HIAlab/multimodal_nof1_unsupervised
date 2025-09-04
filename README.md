
# Multimodal N-of-1

## Introduction to Multimodal N-of-1

This project presents an analytic pipeline for analysing multimodal N-of-1 trials. N-of-1 trials can be seen as individualized multi-crossover trails, investigating the effect of a treatment within a patient. By aggregating N-of-1 trials, you can also analyze them on a population level.

In a multimodal N-of-1 trial, multimodal outcomes such as images for a trial are analyzed instead of more common numeric outcome types (e.g. rating scales). 

In a first demo, we focus on images collected during a study of acne, where we investigate whether skin lotion A or B improve the skin condition within 5 individuals. The data set was collected by students. [2]

In a separate simulation study, we simulated 1000 trials each for two scenarios: one with an effect difference between two skin treatments and one without.

![Vizualisation of a multimodal n-of-1 trial.](assets/MultimodalNof1.png "Multimodal Nof1")


## Methods

In an initial analysis, a CNN model was trained on labeled data to showcase that deep learning could be used to analyse multimodal N-of-1 trials. [2] However, the model needs to be trained on labeled data, which is expensive. So, we came up with an unsupervised approach for analysing the data. [1]

![Usage of an Autoencoder in the analysis pipeline.](assets/Autoencoder.png "Analysis Pipeline")


For that, we trained an Autoencoder on the dataset to learn image representations in a lower-dimensional feature space. This embedding space should capture relevant features of the images, which we then use in the analysis step. For that, we further reduce the feature space by using the first principal components of the embedding vectors. These one-dimensional principal component values in turn serve as the base for statistcal hypothesis tests. With that, we are able to use typical statistical test routines, although we are aware that multidemensional tests on the embedding as maximum mean discrepancy could be useful as well.


### Structure

- `src`: source code for functions
- `data`:  Should contain the data. 
- `code`: code for creating the embeddings.
- `results`: some data created in the experiments (limited).


### Tools and Software

We are using mainly:

- Python
- Pytorch
- pandas


Data from our simulation can be found [here](https://www.doi.org/10.5281/zenodo.17054020).


## Literature

1) Schneider J, Gärtner T, Konigorski S (2023). _Multimodal Outcomes in N-of-1 Trials: Combining Unsupervised Learning and Statistical Inference_. [arXiv:2309.06455](https://doi.org/10.48550/arXiv.2309.06455)
2) Fu J, Liu S, Du S, Ruan S, Guo X, Pan W, Sharma A, Konigorski S (2023). _Multimodal N-of-1 trials: A Novel Personalized Healthcare Design_. [arXiv:2302.07547](https://doi.org/10.48550/arXiv.2302.07547)