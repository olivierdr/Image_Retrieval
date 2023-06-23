# Importing the FastApi class
from fastapi import FastAPI, HTTPException
from src.model import prepare_model
from src.predict import predict
from requests.exceptions import ConnectionError, HTTPError, MissingSchema


app = FastAPI()

model, processor, dataset_with_embeddings = prepare_model()


@app.post("/Images Retrieval")
async def get_similar_images(url: str):
    global model, processor, dataset_with_embeddings
    try:
        # Try to load image and find similarities
        results_, _, _ = predict(model, processor, dataset_with_embeddings, url)

    except ConnectionError as err:
        raise HTTPException(
            status_code=404,
            detail=f"The url is not valid or the image could not be loaded. Error {err}"
        )
    except HTTPError as err:
        raise HTTPException(
            status_code=404,
            detail=f"The url is not valid or the image could not be loaded. Error {err}"
        )
    except MissingSchema as err:
        raise HTTPException(
            status_code=404,
            detail=f"The url is not valid. Error {err}"
        )

    return results_
