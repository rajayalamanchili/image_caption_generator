import src
import pathlib



# set directories
PACKAGE_DIRECTORY = pathlib.Path(src.__file__).resolve().parents[1]
DATASET_DIRECTORY = PACKAGE_DIRECTORY / 'data'
MODEL_DIRECTORY = PACKAGE_DIRECTORY / 'models'

# image and text data folders/files
IMAGES_DIRECTORY = DATASET_DIRECTORY / 'raw' / 'Flickr8k_Dataset'
CAPTIONS_DIRECTORY = DATASET_DIRECTORY / 'raw' / 'Flickr8k_text'

CAPTIONS_FILE = CAPTIONS_DIRECTORY / 'Flickr8k.token.txt'