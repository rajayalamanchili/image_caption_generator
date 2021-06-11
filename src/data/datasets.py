import pickle
from os import path
from src.features import build_features
from src.config import config
import torch
from torch.utils.data import Dataset


class FlickrDataset(Dataset):

    img_caption_dict = {}
    image_ids = []

    word_to_int_map = {}
    int_to_word_map = {}


    def __init__(self, file_name, dtype="train"):

        # read image ids
        self.images_directory = config.IMAGES_DIRECTORY
        
        with open(file_name, "r") as f:
            self.image_ids = f.read().strip().splitlines()
            
        img_caption_dict = build_features.load_img_caption_data(
                config.CAPTIONS_TOKENS_FILE)
        
        self.img_caption_dict = img_caption_dict
        
        # create word mappings
        word_mapping_fname = config.FEATURES_DIRECTORY / "word_mappings.pkl"
        
        if path.exists(word_mapping_fname):
            
            print("loading word maps")
            (word_to_int_map, int_to_word_map) = pickle.load(open(
                word_mapping_fname, "rb"))
            
            self.word_to_int_map = word_to_int_map
            self.int_to_word_map = int_to_word_map
                        
        else:
            
            # create vocabulary
            print("Creating word maps")
            
            # merge caption text data
            text_data = " ".join([" ".join(txt) for txt in 
                                  img_caption_dict.values()])

            # create word to int mappings
            word_to_int_map, int_to_word_map = build_features.create_word_mappings(
                text_data)

            self.word_to_int_map = word_to_int_map
            self.int_to_word_map = int_to_word_map
            
            pickle.dump((word_to_int_map, int_to_word_map),
                        open(word_mapping_fname, "wb"))
        
        # extract image features
        fname = dtype + "_image_features.pt"
        img_features_fname = config.FEATURES_DIRECTORY / fname
        
        if path.exists(img_features_fname):

            print("loading image features")
            
            # load features
            self.image_features = torch.load(img_features_fname)
            
        else:
            
            
            image_features = build_features.extract_image_features(
                self.images_directory,
                self.image_ids)
            
            self.image_features = image_features
            
            # save features            
            torch.save(image_features, img_features_fname)
            
            
    def __getitem__(self, index):
        
        img_fname = self.image_ids[index]
        
        # load image features
        if(img_fname in self.image_features.keys()):
            
            img_tensor = self.image_features[img_fname]
        
        else:
            
            img_features = build_features.extract_image_features(
                self.images_directory,
                self.img_fname)
            
            img_tensor = img_features[img_fname]
        
        # load caption features
        caption_txt = self.img_caption_dict[img_fname]
        
        txt_tensor = torch.tensor(build_features.convert_text_to_int(
            caption_txt[1], self.word_to_int_map))
        
        
        return img_tensor, txt_tensor

    def __len__(self):
        return len(self.img_caption_dict)

    
if __name__ == "__main__":

    training_dataset = FlickrDataset(file_name=config.CAPTIONS_TRAIN_FILE, dtype="train")
    validation_dataset = FlickrDataset(file_name=config.CAPTIONS_VALIDATION_FILE, dtype="valid")
    test_dataset = FlickrDataset(file_name=config.CAPTIONS_TEST_FILE, dtype="test")
    print(test_dataset[0])