from src.utils import score_and_retrieved_examples, retrieved_id, download_image
from src.const import QUERY_IMAGE_DIR_NAME
from PIL import Image


def url_to_image(url, dir_name_query_image):
    """Download image from URL

    Args:
        url (str)
        dir_name_query_image (str)

    Returns:
        filename
    """
    filename = download_image(url, dir_name_query_image)
    return filename


def predict(model, processor, dataset_with_embeddings, url):
    """Retrieved the closest images from url

    Args:
        model (transformers.ViTModel)
        processor (transformers.ViTImageProcessor)
        dataset (Dataset)
        dataset_with_embeddings (Dataset)
        url (str)

    Returns:
        id_results, retrieved_examples, query_image
    """
    filename = url_to_image(url, QUERY_IMAGE_DIR_NAME)

    query_image = Image.open(filename)

    # Compute similarities
    scores, retrieved_examples = score_and_retrieved_examples(
        model,
        processor,
        dataset_with_embeddings,
        query_image
    )

    id_results = retrieved_id(retrieved_examples)

    return id_results, retrieved_examples, query_image
