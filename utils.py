import torch, torchvision
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_segmentation_model():
    # Load the DeepLab v3 model to system
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.to(device).eval()
    return model

def show_images(images, color=False):
    if color:
        sqrtimg = int(np.ceil(np.sqrt(images.shape[2]*images.shape[3])))
    else:
        images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
        sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if color:
            plt.imshow(np.swapaxes(np.swapaxes(img, 0, 1), 1, 2))
        else:
            plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return 

def in_image_bounds(image, center_x, center_y, crop_size):
    start = int(np.floor(crop_size / 2))
    end = int(np.ceil(crop_size / 2))
    
    x_in_bounds = center_x - start >= 0 and center_x + end <= image.shape[2]
    y_in_bounds = center_y - start >= 0 and center_y + end <= image.shape[1]
    
    return x_in_bounds and y_in_bounds 

def get_positive_image_pair(image, object_mask, crop_size = 50, object_area_threshold = .1, gray_threshold = .5):
    '''
    Given an image and object mask, retrieve a pair of augmented image crops from the given image to acquire a positive pair of images
    that contain the same object
    
    Designate one of these images as the "main" image, and designate the other as the positive pair corresponding to the "main" image
    
    params:
        image - The image we are trying to acquire the positive image pair for
        object_mask - The object mask retrieved from saliency estimation for the image
        object_area_threshold - The minimum area of the object that must be included to be considered a positive augmented view of the image
        gray_threshold - The grayscale threshold minimum that we will consider part of the object mask
    '''
    
    main_image = None
    positive_image = None
    
    start = int(np.floor(crop_size / 2))
    end = int(np.ceil(crop_size / 2))
    object_indices = torch.where(object_mask > gray_threshold)

    while main_image == None or positive_image == None:
        if main_image == None:
            idx = np.random.randint(0, len(object_indices[2]))
            center_y = int(object_indices[2][idx].item())
            center_x = int(object_indices[3][idx].item())
             
            if in_image_bounds(image, center_x, center_y, crop_size):
                crop_mask = object_mask[:, :, center_y - start : center_y + end, center_x - start : center_x + end]
                
                if crop_mask[crop_mask > gray_threshold].size()[0] >= object_area_threshold * object_mask[object_mask > gray_threshold].size()[0]:
                    main_image = image[:, center_y - start : center_y + end, center_x - start : center_x + end]
                
        if positive_image == None:
            idx = np.random.randint(0, len(object_indices[2]))
            center_y = int(object_indices[2][idx].item())
            center_x = int(object_indices[3][idx].item())
    
            if in_image_bounds(image, center_x, center_y, crop_size):
                crop_mask = object_mask[:, :, center_y - start : center_y + end, center_x - start : center_x + end]
                
                if crop_mask[crop_mask > gray_threshold].size()[0] >= object_area_threshold * object_mask[object_mask > gray_threshold].size()[0]:
                    positive_image = image[:, center_y - start : center_y + end, center_x - start : center_x + end] 
                    
    transform = torchvision.transforms.Resize((image.shape[1], image.shape[2]))
    
    return transform(main_image), transform(positive_image)

def get_negative_images(image, object_mask, num_negatives = 128, crop_size = 50, object_area_threshold = .1, gray_threshold = .5):
    ''' 
    Given an image and object mask, retrieve num_negatives augmented image crops from the given image to acquire a list of negative images that are guaranteed to not contain the object.

    image - The image we are trying to acquire the positive image pair for
    object_mask - The object mask retrieved from saliency estimation for the image
    gray_threshold - The grayscale threshold minimum that we will consider part of the object mask
    '''
    transform = torchvision.transforms.Resize((image.shape[1], image.shape[2]))
    neg_images = []
    
    start = int(np.floor(crop_size / 2))
    end = int(np.ceil(crop_size / 2))
    object_indices = torch.where(object_mask < gray_threshold)
    
    while len(neg_images) < num_negatives:
        idx = np.random.randint(0, len(object_indices[2]))
        center_y = int(object_indices[2][idx].item())
        center_x = int(object_indices[3][idx].item())

        if in_image_bounds(image, center_x, center_y, crop_size):
            crop_mask = object_mask[:, :, center_y - start : center_y + end, center_x - start : center_x + end]

            if crop_mask[crop_mask > gray_threshold].size()[0] < object_area_threshold * object_mask[object_mask > gray_threshold].size()[0]:
                new_image = image[:, center_y - start : center_y + end, center_x - start : center_x + end]  
                neg_images.append(transform(new_image))
    
    return neg_images  