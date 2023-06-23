## Objectif
Introducing *ImageFinder*, a fictional application that uses machine learning to provide users with a powerful image retrieval system. With ImageFinder, users can easily find similar images from a large dataset with just a click.

## Technical Choices
We use a Vision Transformer (transformer encoder model) named *google/vit-base-patch16-224-in21k* from Hugging Face to proceed this content-based image retrieval task.

This model is  pretrained on a large collection of Images (ImagetNet). We choose this model because it learns an inner representation of images that can then be used to extract features.

This task is decomposed as follow:

### Part I - Implement an image retrieval system
1. Creation of embedding vectors from the image dataset
2. Use of FAISS vector database for semantic search

We choose the library FAISS because it assures faster similar search when the number of embedded vectors is huge which would be the case in a real data world with thousands of images.

FAISS uses index operations and euclidean distance to measure vector similarity. We could have used the cosine similarity too, but in this task the two measures won’t differ the results, so we choose to keep the original L2 FAISS index.

### Part II - Create Post Route API
1. FastAPI to interact with the system and retrieved the three closest images from the input image

FastAPI enables to develop a quick and serverless API in python. The documentation to interact with the API is really useful and understandable by non-developers that’s why we choose the library.


## Getting Started

- Install [Docker Desktop](https://www.docker.com/*)
- Run `docker-compose build`
- Run `docker-compose up`
- Fast API app, open on a web browser: `http://localhost:8000/docs`
- Try the app with the url image of your choice

## Limitations and future updates

**Limitations**
1. Optimisation: Compute the embedding vectors is not optimised. It can be really long if we have a lot of images.
2. Memory Usage: We stock in memory all the objects (model, dataset etc.) and it's not optimal.

**Future Updates**
1. Print the retrieval images in the API to compare directly with the input
2. Rework on point 1 & 2 from Limitations.
