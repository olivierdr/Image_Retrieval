import os
import src.utils as utils
from src.const import IMAGE_DIR, HUGGING_FACE_MODEL_NAME, CSV_NAME

# Cause Error dataset.map "Initializing libiomp5.dylib, but found libomp.dylib already initialized"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def prepare_model():
    """DOC """
    utils.execute_download_image(IMAGE_DIR, CSV_NAME)

    df = utils.load_csv(CSV_NAME)
    utils.create_metadata(df, IMAGE_DIR)

    dataset = utils.load_data_from_dir(IMAGE_DIR)
    model, processor = utils.load_model(HUGGING_FACE_MODEL_NAME)

    dataset_with_embeddings = dataset.map(
      lambda row: {'embeddings': utils.extract_embedding(row["image"], model, processor)}
    )

    dataset_with_embeddings = utils.set_index(dataset_with_embeddings)

    return model, processor, dataset_with_embeddings


if __name__ == '__main__':
    model, processor, dataset_with_embeddings = prepare_model()
