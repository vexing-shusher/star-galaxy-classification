import glob

import PIL.Image as pil
import numpy as np

def thresh_function(img: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img)
    thr = np.mean(img)+np.std(img)
    out[img > thr] = img[img > thr]
    return out

def get_data(data_path: str) -> tuple:

    data = []
    labels =[]
    
    # read filenames from the corresponding directories
    galaxy_files = glob.glob(f'{data_path}/galaxy/*')
    star_files = glob.glob(f'{data_path}/star/*')

    print(f"Number of galaxies: {len(galaxy_files)}")
    print(f"Number of stars: {len(star_files)}")

    for x in galaxy_files:
        image = pil.open(x)
        image= np.array(image)
        data.append(image)
        labels.append(0)
        
    for x in star_files:
        image = pil.open(x)
        image= np.array(image)
        data.append(image)
        labels.append(1)
        
    data, labels = np.array(data).astype(np.uint8), np.array(labels)

    return data, labels

def random_mean_oversampling(x_train: np.ndarray, 
                             y_train: np.ndarray,
                             mean_ovs: int = 10,
                            ) -> tuple:
    
    #OVERSAMPLE THE GALAXY CLASS BY ADDING AVERAGES OF {mean_ovs} GALAXY SAMPLES CHOOSEN RANDOMLY WITH REPLACEMENT
    
    new_samples = []
    new_labels = []
    galaxy_size = len(x_train[y_train==0])
    star_size = len(x_train[y_train==1])
    galaxy_idx = np.arange(0, len(x_train), 1).astype(np.int32)[y_train==0]
    while galaxy_size < star_size:
        random_samples = [x_train[np.random.choice(galaxy_idx)] for k in range(mean_ovs)]
        mean_sample = np.mean(random_samples, axis=0).astype(np.uint8)
        new_samples.append(mean_sample)
        new_labels.append(0)
        
        galaxy_size += 1
    
    x_train = np.concatenate((x_train, np.array(new_samples)))
    y_train = np.concatenate((y_train, np.array(new_labels)))
    
    return x_train, y_train
