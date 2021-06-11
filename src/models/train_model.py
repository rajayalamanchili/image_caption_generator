from src.data.datasets import FlickrDataset
from src.config import config

class CaptionGeneratorModel():
    
    def __init__(self):
        

if __name__ == "__main__":
    
    training_dataset = FlickrDataset(file_name=config.CAPTIONS_TRAIN_FILE, dtype="train")
    training_loader = 