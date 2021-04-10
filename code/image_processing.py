import os
import io
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, ImageOps, ImageEnhance
sys.modules['Image'] = Image 

def populate_directory(input_dir, output_dir, img_class = 'images', input_format='.bmp', subset=None):
    ''' example usage ...
    path_train = 'dataset_for_m1_model//input//train//'
    path_out = 'tsne_dashboard//slider_image_dataset//'
    populate_directory(input_dir=path_test, output_dir=path_out, img_class='images', input_format='.bmp')
    subset: populate directory with a randomly shuffled subset 
    '''
    if not os.path.exists(output_dir + img_class):
        os.makedirs(output_dir + img_class)
    
    ids_temp = next(os.walk(input_dir + img_class))[2]
    
    if subset is None:
        ids = []
        for i in ids_temp:
            if i.endswith(input_format):
                ids.append(i)
    else:
        ids_1 = []
        for i in ids_temp:
            if i.endswith(input_format):
                ids_1.append(i)
                
        random.seed(2019)
        id_order = np.arange(len(ids_1))
        np.random.shuffle(id_order)
        
        ids = []
        for n, i in enumerate(range(len(id_order))):
            if (subset is not None) and (n == subset):
                break
            else:
                ids.append(ids_1[np.int(id_order[i])])
        
            
    for n, id_ in enumerate(ids):
        # Load image
        img = load_img(input_dir + "//" + img_class + "//" + id_)
        # Save image
        print("\r saving {} / {}".format(n+1, len(ids)), end='')
        img.save(output_dir + "//" + img_class + "//" + id_)
        
    print('\n done!')
    
    
def populate_directory_temp(input_dir, output_dir, input_format='.bmp', subset=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ids_temp = next(os.walk(input_dir))[2]
    
    if subset is None:
        ids = []
        for i in ids_temp:
            if i.endswith(input_format):
                ids.append(i)
    else:
        ids_1 = []
        for i in ids_temp:
            if i.endswith(input_format):
                ids_1.append(i)
                
        random.seed(2019)
        id_order = np.arange(len(ids_1))
        np.random.shuffle(id_order)
        
        ids = []
        for i in range(len(id_order)):
            ids.append(ids_1[np.int(id_order[i])])
        
            
    for n, id_ in enumerate(ids):
        # Load image
        img = load_img(input_dir + "//" + id_)
        # Save image
        print("\r saving {} / {}".format(n+1, len(ids)), end='')
        img.save(output_dir + '/' + id_)
        if (subset is not None) and (n == subset):
            break
        
    print('\n done!')

# path_in = 'data//dataset_for_m1_model_combined//'
# path_out = 'data//experimental_dataset//'
# populate_directory(input_dir=path_in, output_dir=path_out, img_class='images', input_format='.bmp')
# populate_directory(input_dir=path_in, output_dir=path_out, img_class='masks', input_format='.png', subset=100)

# path_in = 'data//dataset_for_m1_model_combined//masks'
# path_out = 'data//dataset_for_m1_model_combined//masks_partial'

# populate_directory_temp(input_dir=path_in, output_dir=path_out, input_format='.png', subset=999)

def normalize_rgb_channels(X_train, X_valid):
    m1 = np.mean(X_train[:, :, :, 0])
    m2 = np.mean(X_train[:, :, :, 1])
    m3 = np.mean(X_train[:, :, :, 2])

    print('Mean value of the first channel: {}'.format(m1))
    print('Mean value of the second channel: {}'.format(m2))
    print('Mean value of the third channel: {}'.format(m3))

    s1 = np.std(X_train[:, :, :, 0])
    s2 = np.std(X_train[:, :, :, 1])
    s3 = np.std(X_train[:, :, :, 2])

    print('Std value of the first channel: {}'.format(s1))
    print('Std value of the second channel: {}'.format(s2))
    print('Std value of the third channel: {}'.format(s3))

    X_train[:,:,:,0] -= m1
    X_train[:,:,:,1] -= m2
    X_train[:,:,:,2] -= m3

    X_train[:,:,:,0] = X_train[:,:,:,0]/s1
    X_train[:,:,:,1] = X_train[:,:,:,1]/s2
    X_train[:,:,:,2] = X_train[:,:,:,2]/s3

    X_valid[:,:,:,0] -= m1
    X_valid[:,:,:,1] -= m2
    X_valid[:,:,:,2] -= m3

    X_valid[:,:,:,0] = X_valid[:,:,:,0]/s1
    X_valid[:,:,:,1] = X_valid[:,:,:,1]/s2
    X_valid[:,:,:,2] = X_valid[:,:,:,2]/s3
    
    return X_train, X_valid


def normalize_single_array_rgb_channels(X):
    m1 = np.mean(X[:, :, :, 0])
    m2 = np.mean(X[:, :, :, 1])
    m3 = np.mean(X[:, :, :, 2])

    print('Mean value of the first channel: {}'.format(m1))
    print('Mean value of the second channel: {}'.format(m2))
    print('Mean value of the third channel: {}'.format(m3))

    s1 = np.std(X[:, :, :, 0])
    s2 = np.std(X[:, :, :, 1])
    s3 = np.std(X[:, :, :, 2])

    print('Std value of the first channel: {}'.format(s1))
    print('Std value of the second channel: {}'.format(s2))
    print('Std value of the third channel: {}'.format(s3))
    
    X[:,:,:,0] -= m1
    X[:,:,:,1] -= m2
    X[:,:,:,2] -= m3

    X[:,:,:,0] = X[:,:,:,0]/s1
    X[:,:,:,1] = X[:,:,:,1]/s2
    X[:,:,:,2] = X[:,:,:,2]/s3
    
    return X
    

def numpy_embeddings_to_df(X_encoded):
    d = {'x': X_encoded[:,0], 'y': X_encoded[:,1]}
    df = pd.DataFrame(data=d, index=None)
    return df


def get_normalized_image(x_img, rgb_df):
  
    m1 = rgb_df.loc[0, ["mean"]].values.tolist()[0]
    m2 = rgb_df.loc[1, ["mean"]].values.tolist()[0]
    m3 = rgb_df.loc[2, ["mean"]].values.tolist()[0]

    s1 = rgb_df.loc[0, ["std"]].values.tolist()[0]
    s2 = rgb_df.loc[1, ["std"]].values.tolist()[0]
    s3 = rgb_df.loc[2, ["std"]].values.tolist()[0]
    
    x_img[:,:,0] -= m1
    x_img[:,:,1] -= m2
    x_img[:,:,2] -= m3
    x_img[:,:,0] = x_img[:,:,0]/s1
    x_img[:,:,1] = x_img[:,:,1]/s2
    x_img[:,:,2] = x_img[:,:,2]/s3   
    
    return x_img
    

def embeddings_df_to_dict(df):
    column = 'label'
    labels = df['label'].unique().tolist()
    embeddings_dict = {var: df.query(f'{column} == "%s"' % var) for var in labels}
    return embeddings_dict


def ids_list_to_df(ids):
    ids_list = []
    for n, id_ in enumerate(ids):
        i = id_.split('.')
        ids_list.append(i[0])

    ids_df = pd.DataFrame(ids_list) # , dtype="string")
    return ids_df

def img_array_from_img_directory(path, img_height, img_width, img_format='.bmp', randomseed=2019, masks=False, n_classes=5):
    ''' Iterate over directory of images, shuffle, scale, and return numpy array along with corresponding IDs'''
    
    ids_temp = next(os.walk(path + "images"))[2]
    ids_1 = []
    for i in ids_temp:
        if i.endswith('.bmp'):
            ids_1.append(i)
            
    random.seed(randomseed)
    id_order = np.arange(len(ids_1))
    np.random.shuffle(id_order)

    ids = []
    for i in range(len(id_order)):
        ids.append(ids_1[np.int(id_order[i])])

    X = np.zeros((len(ids), img_height, img_width, 3), dtype=np.float32) 
    if masks:
        y = np.zeros((len(ids), im_height, im_width, n_classes), dtype=np.float32) 

    print('Number of images:' + str(len(ids)))
    if X.shape[0] == 0:
        print("no image found")
        sys.exit()
        
    for n, id_ in enumerate(ids):
        # Load images
        print('\r Loading %s / %s ' % (n+1, len(ids)), end='')
        img = load_img(path + '/images/' + id_)
        x_img = img_to_array(img)
        x_img = resize(x_img, (img_height, img_width, 3), mode='constant', preserve_range=True)
        x_img = x_img / 255
        
        # Load masks
        if masks:
            id_mask = id_[:-4] + ".png"
            mask = img_to_array(load_img(path + '/masks/' + id_mask, color_mode = 'grayscale'))
            mask = cv2.resize(mask, (im_width, im_height), interpolation=cv2.INTER_NEAREST) 
            mask.astype(np.int)
        
        X[n, ...] = x_img.squeeze()
        if masks:
            y[n] = to_categorical(mask.astype(int), n_classes)
            
        
    print('Done!')
    if masks:
        return np.array(X), np.array(y), ids
    else:
        return np.array(X)

def defect_labels(Y):
    '''Specifies an integer label [0,1,2,3,4] for each image. 
    Label corresponds to which defect is most present in the image, where no defect is zero'''
    n_classes = 5
    channel_sums = np.zeros((len(Y), 5))

    for j in range(0, len(Y)):
        for i in range(1, n_classes):
            channel = Y[j,:,:,i]
            channel_sums[j, i] = np.sum(channel)

    print("Channel scores ... shape: ", channel_sums.shape)
    defect_types = np.argmax(channel_sums, axis=1)
    print("Done ... Calculated argmax of channel scores")

    return defect_types


def encode_mask_to_rgb(mask):
    n_classes = 5
    # cv2 follows bgr, but skimage follows rgb
    mask_encoded = np.zeros( (mask.shape[0], mask.shape[1], 3) )
    
    colors = [( random.randint(0,255), random.randint(0,255), random.randint(0,255) ) for _ in range(5)]
    colors[0]=(0,0,0)
    colors[1]=(0,0,255) # blue
    colors[2]=(255,0,0) # red
    colors[3]=(0,255,255) # cyan
    colors[4]=(255,255,0) # yellow

    for c in range(1,n_classes):
        mask_encoded[mask[:,:] == c] = colors[c]

    return mask_encoded.astype(np.uint8)

def output_predictions(array):
    
    output_array = np.zeros((array.shape[0],array.shape[1],array.shape[2],3))    
    for i in range(len(array)):
        print('\r Predicting %s / %s ' % (i+1, len(array)), end='')
        # pred_mask = model.predict(array[i].reshape(1,im_height, im_width, 3))
        pred_mask = np.squeeze(array[i])
        pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask = maskcolorencoders_output(pred_mask)
        pred_mask = np.clip(pred_mask, 0, 255).astype('uint8')
        
        output_array[i] = pred_mask
        
    print("\nDone")
    return output_array


def encode(model, X):
    print("Encoding ...")
    feature_maps = model.predict(X)
    print("Encoder output shape: ", feature_maps.shape)
    X_codings = np.amax(feature_maps, axis=-1)
    print("Calculated amax of feature maps ... new shape: ", X_codings.shape)
    X_codings = np.reshape(X_codings, (X_codings.shape[0], X_codings.shape[1] * X_codings.shape[2]))
    print("Flattened to shape: ", X_codings.shape)
    return X_codings

def get_tsne_embeddings(X):
    np.random.seed(42)
    tsne = TSNE(
        n_components=2,
        method="barnes_hut",
        random_state=22,
        learning_rate=200,
        perplexity=20,
        n_iter=3000, 
        verbose=0
    )
    print("Calculating t-SNE embeddings ...")
    X_encoded_2D = tsne.fit_transform(X)
    X_encoded_2D = (X_encoded_2D - X_encoded_2D.min()) / (X_encoded_2D.max() - X_encoded_2D.min())
    print("t-SNE embeddings calculated, final output shape: ", X_encoded_2D.shape)

    return X_encoded_2D

def plot_labeled_encodings(X_encoded_2D, labels):
    colors = [( random.randint(0, 1), random.randint(0, 1), random.randint(0, 1) ) for _ in range(5)]
    colors[0] = (0,0,0) # black
    colors[1] = (1,0,0) # blue
    colors[2] = (0,0,1) # red
    colors[3] = (1,1,0) # cyan
    colors[4] = (0,1,1) # yellow
    newcmp = ListedColormap(colors)
    plt.figure(figsize=(15, 10))
    plt.title("t-SNE Encodings", fontsize=18)
    plt.scatter(X_encoded_2D[:,0], X_encoded_2D[:,1], c=labels, cmap = newcmp)
    plt.gca().axis("off")
    # plt.legend()
    plt.show()


def plot_tsne_encodings(X_encoded_2D, X_clusters):
    plt.figure(figsize=(15, 10))
    plt.title("t-SNE Encodings", fontsize=18)
    plt.scatter(X_encoded_2D[:,0], X_encoded_2D[:,1], c=X_clusters)
    plt.gca().axis("off")
    plt.show()
