import os
import torch
from src.config import config
from tqdm import tqdm

from PIL import Image
from torchvision import models, transforms

def load_img_caption_data(filepath):
    """ This function loads text from  input text file

    :param filepath: full path to text file
    :return: returns list containing
    """

    # read text data from file
    with open(os.path.join(filepath), "r") as f:
        text_data = f.read().strip().splitlines()

    # create dictionary with image ids and associated captions
    img_caption_dict = {}

    for text in text_data:

        img_id, caption = text.split("\t")
        img_id = img_id[:-2]

        if img_id in img_caption_dict:
            img_caption_dict[img_id].append(caption)
        else:
            img_caption_dict[img_id] = [caption]

    return img_caption_dict


def preprocess_text(text_str):
    """ function to apply preprocessing on input text

    :param text_str: input string
    :return: string after applying preprocessing
    """

    # replace punctuations with word tokens
    punctuation_tokens = {".": "||Period||",
                          ",": "||Comma||",
                          "\"": "||Quotation_Mark||",
                          ";": "||Semicolon||",
                          "!": "||Exclamation_Mark||",
                          "?": "||Question_Mark||",
                          "(": "||Left_Parantheses||",
                          ")": "||Right_Parantheses||",
                          "-": "||Dash||",
                          "\n": "||Return||"}

    for key, value in punctuation_tokens.items():
        text_str = text_str.replace(key, value)

    # convert text to lower
    text_str = text_str.lower()

    return text_str


def create_word_mappings(text_str):
    """ function to create vocabulary and create word to int mappings

    :param text_str: input string
    :return: tuple of dictionaries with word to int and int to word mappings
    """

    # create word to int, int to word mappings
    text_str = data_utils.preprocess_text(text_str).split()
    unique_words = set(text_str)

    word_to_int_map = {word: idx for idx, word in enumerate(unique_words)}

    int_to_word_map = {val: key for key, val in word_to_int_map.items()}

    return (word_to_int_map, int_to_word_map)

def create_img_caption_int_data(filepath):
    """ function to load captions from text file and convert them to integer
    format

    :return: dictionary with image ids and associated captions in int format
    """

    print("\nLoading caption data : started")
    
    # load caption data
    img_caption_dict = data_utils.load_img_caption_data(filepath)

    # merge caption text data
    text_data = " ".join([" ".join(txt) for txt in img_caption_dict.values()])

    # create word to int mappings
    (word_to_int_map, int_to_word_map) = create_word_mappings(text_data)

    # convert caption data to int
    img_caption_int_dict = {}

    for key, value in img_caption_dict.items():
        img_caption_int_dict[key] = [convert_text_to_int(txt, word_to_int_map)
                                     for txt in value]
    print("\nLoading caption data : completed")
    
    return img_caption_int_dict

def convert_text_to_int(text_data, word_to_int_map):

    text_data = preprocess_text(text_data)

    return [word_to_int_map[word] for word in text_data.split()]


def convert_int_to_text(int_data, int_to_word_map):

    return " ".join([int_to_word_map[idx] for idx in int_data])

def extract_image_features(file_list):
    
    print("\nExtracting image features : started")
    
    features = {}
    
    # check device for cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("\nDevice: ", device)
    
    # set feature extractor model
    model = models.vgg16(pretrained=True)
    model.classifier = model.classifier[:-2]
    
    for param in model.features.parameters():
        param.requires_grad = False
        
    model.to(device)
    
    img_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])
    # extract features
    for img_id in tqdm(file_list, desc="Processing images"):
        
        img = Image.open(config.IMAGES_DIRECTORY / img_id)
        
        img = img_transform(img)
        img.to(device)
        
        features[img_id] = model(img.unsqueeze(0))
        
    return features
    
    print("\nExtracting image features : completed")