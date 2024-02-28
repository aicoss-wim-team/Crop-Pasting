# Crop and Pasting
**Problem & Solution**: Data shifting problem causing imbalanced class distribution  
**Task**: Recycling waste data augmentation  
**Object information**: Recycling waste objects on a conveyer belt  
### Crop and Paste condition

1. **Background**
    1. objects must be pasted inside the conveyor belt area in an empty conveyor belt image
    2. should not go outside the belt area based on the creation coordinates
2. **Objects**
    1. objects should be saved at their cropped size
    2. when saving, remove the background of the object through the alpha channel
    3. .png format must be used to apply the alpha channel
    4. up to N objects can be glued together
    5. allow up to N% overlap between objects

Translated with DeepL.com (free version)
### 1. Generate masks with SAM(Segment Anything Model)([Link](https://github.com/facebookresearch/segment-anything))
- Get segmentation mask via object detection bounding box (already have labeled data)
![mask1](./asset/image.png)
![mask2](./asset/image-1.png)
### 2. Crop objects with generated masks

### 3. Paste objects on empty conveyer belt
- **Configuration Parameters in ``crop_paste.py``**
    ```
    CFG = {
    'base_image_path': '../wim_data/SAM_2_objects/conveyer_resized.png', # Back ground image (to be pasted)
    'object_images_folder': '../wim_data/objects/images_object2/', # Object images path
    'max_objects' : 12,  # max objects to paste
    'rectangle' : (300, 0, 680, 740), # Setting ROI (x start, y start, width, height)
    'max_overlap': 0.3, # Overlapping ratio with objects
    'num_iter': 3, # number of images to be generated
    'max_dim': 500, # Number of sampled object images limit
    'output_folder': '../wim_data/crop_paste/', # generated images save path
    }
    ```
- **Randomly Pasting Algorithm**  
    Depends on number of class distribution, sampled more from fewer classes to solve imbalancing problem