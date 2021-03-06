import src
import pathlib




# set directories
PACKAGE_DIRECTORY = pathlib.Path(src.__file__).resolve().parents[1]
DATASET_DIRECTORY = PACKAGE_DIRECTORY / 'data'
MODEL_DIRECTORY = PACKAGE_DIRECTORY / 'models'


# image and text data folders/files
IMAGES_DIRECTORY = DATASET_DIRECTORY / 'raw' / 'Flicker8k_Dataset'
CAPTIONS_DIRECTORY = DATASET_DIRECTORY / 'raw' / 'Flickr8k_text'
FEATURES_DIRECTORY = DATASET_DIRECTORY / 'processed'

CAPTIONS_TOKENS_FILE = CAPTIONS_DIRECTORY / 'Flickr8k.token.txt'
CAPTIONS_TRAIN_FILE = CAPTIONS_DIRECTORY / 'Flickr_8k.trainImages.txt'
CAPTIONS_VALIDATION_FILE = CAPTIONS_DIRECTORY / 'Flickr_8k.devImages.txt'
CAPTIONS_TEST_FILE = CAPTIONS_DIRECTORY / 'Flickr_8k.testImages.txt'