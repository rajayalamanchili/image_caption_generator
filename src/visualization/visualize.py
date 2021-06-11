from src.data.datasets import FlickrDataset
from src.config import config

import matplotlib.pyplot as plt
import torch
from PIL import Image

def display_img_FlickrDataset(dataset, index=0, predicted_caption=None):
    
    image = Image.open(dataset.images_directory / dataset.image_ids[index])
    caption_txt = "\n".join(dataset.img_caption_dict[dataset.image_ids[index]])
    
    fig = plt.figure(figsize=(30, 12))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image)
    ax.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    ax.text(0,0.1,"Actual:", fontsize=15, verticalalignment="top", weight="bold")
    ax.text(0,0.15,caption_txt, fontsize=15, verticalalignment="top", weight="bold")
    ax.text(0,0.4,"Predicted:", fontsize=15, verticalalignment="top", weight="bold")
    ax.text(0,0.45,caption_txt, fontsize=15, verticalalignment="top", weight="bold")
    ax.axis("off")
    ax.invert_yaxis()
    
if __name__ == "__main__":

    training_dataset = FlickrDataset(file_name=config.CAPTIONS_TRAIN_FILE, dtype="train")
    
    display_img_FlickrDataset(training_dataset, 100)
    