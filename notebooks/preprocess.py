from keras.applications import InceptionV3
from cv2 import imread
from tqdm import tqdm
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
import pickle


def preprocess_image(model, image_path):
    img = None
    res = None
    try:
        img = imread(image_path)
        img = (np.expand_dims(img, axis=0) / 127.5) - 1.0
        res = np.average(model.predict(img)[0, :], axis=(0, 1))
        #print(np.min(res))
        #print(np.max(res))
    except:
        print('Unexecpected Error: {} \n{}'.format(image_path, sys.exc_info()))
        print('Img {}'.format(img))
        print('Res {}'.format(res))
    return res


def get_paths(base, ext, c_dir=''):
    curr = base + os.sep + c_dir
    res = []
    for f in os.listdir(curr):
        c_file = curr + os.sep + f
        if os.path.isfile(c_file) and f.endswith(ext):
            res.append(c_dir + f)
        elif os.path.isdir(c_file):
            res.extend(get_paths(base, ext, c_dir + f + os.sep))
    return res


def preprocess_images(directory):
    model = InceptionV3(include_top=False)
    imgs = get_paths(directory, '.jpg')
    features = [[f_p, preprocess_image(model, directory + os.sep + f_p)] for f_p in tqdm(imgs)]
    features = filter(lambda x: x[1] is not None, features)
    features = list(features)
    files = [x[0] for x in features]
    features = [x[1] for x in features]
    return files, np.asarray(features)


valid_chars = set('qwertyuiopasdfghjklzxcvbnm\' ')


def preprocess_captions(directory):
    descs = get_paths(directory, '.eng')
    res = {}
    for d in tqdm(descs):
        #print(directory + os.sep + d)
        try:
            tree = ET.parse(directory + os.sep + d)
        except:
            #print('Trying windows encoding')
            xmlp = ET.XMLParser(encoding='iso-8859-1')
            tree = ET.parse(directory + os.sep + d, parser=xmlp)
        root = tree.getroot()
        text = root.find('DESCRIPTION').text.lower().replace('-', ' ')
        text = ''.join(filter(lambda c: c in valid_chars, text)).split(' ')
        res[d] = text
    return res


if __name__ == '__main__':
    #files, img_features = preprocess_images('dataset/iaprtc12/images')
    files, img_features = preprocess_images('../neural_image_captioning/datasets/IAPR_2012/iaprtc12/images')
    np.savez('images_features.npz', img_features)
    pickle.dump(files, open('images_features.pck', 'wb'))
    #img_caption = preprocess_captions('dataset/iaprtc12/annotations_complete_eng')
    img_caption = preprocess_captions('../neural_image_captioning/datasets/IAPR_2012/iaprtc12/annotations_complete_eng')
    pickle.dump(img_caption, open('images_captions.pck', 'wb'))
