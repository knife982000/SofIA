from keras.layers import Dense, GRU, Dropout, Input, Embedding, TimeDistributed, RepeatVector, add
from keras.models import Model
from collections import Counter, deque
from sklearn.utils import shuffle, compute_class_weight
import numpy as np
import pickle
import random


def create_model(voc_size, img_features, caption_len, hidden=128):
    i_cap = Input(shape=(caption_len, voc_size), name='caption')
    time_dense = TimeDistributed(Dense(hidden), name='cap_dense')(i_cap)
    cap_drop = Dropout(0.5, )(time_dense)

    i_img = Input(shape=(img_features,), name='image')
    den = Dense(hidden, name='dense_img')(i_img)
    rec_img = RepeatVector(caption_len, name='repeat_img')(den)
    img_drop = Dropout(0.5, name='img_dense')(rec_img)

    con = add([cap_drop, img_drop])
    lstm = GRU(hidden, return_sequences=True)(con)
    den = TimeDistributed(Dense(voc_size, activation='softmax', name='next_word'))(lstm)

    model = Model(inputs=[i_cap, i_img], outputs=den)
    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])

    return model


start_token = '<S>'
end_token = '<E>'


class ImageGen:

    def __init__(self, feat_file, caption_file, caption_max_len, min_reps=2, batch_size=256, train=0.8):
        img_feat = np.load(feat_file + '.npz')['arr_0']
        img_files = pickle.load(open(feat_file + '.pck', 'rb'))
        self.len_img_feat = img_feat.shape[1]
        self.img_feat = {img_files[j].split('.')[0]: img_feat[j, :] for j in range(len(img_files))}

        capts = pickle.load(open(caption_file + '.pck', 'rb'))
        capts = {k.split('.')[0]: [start_token] + v + [end_token] for k, v in capts.items()
                 if k.split('.')[0] in self.img_feat and len(v) > 2}

        self.img_feat = {k: v for k, v in self.img_feat.items() if k in capts}#Filtro imagenes sin caption

        files = list(self.img_feat.keys())
        files.sort()
        sep = int(len(files) * train)
        self.train = files[:sep]
        self.test = files[sep:]

        self.word_cont = Counter()
        word_cont = self.word_cont

        for f in self.train:
            v = capts[f]
            for i, w in enumerate(v):
                if i == caption_max_len + 1:
                    break
                word_cont[w] += 1

        self.word_id = {}
        self.img_cap = {}
        idx = 0
        for k in self.train:
            v = capts[k]
            curr_sent = []
            i = 0
            for w in v:
                if word_cont[w] < min_reps:
                    continue
                if i == caption_max_len:
                    break
                if w not in self.word_id:
                    self.word_id[w] = idx
                    idx += 1
                curr_sent.append(self.word_id[w])
                i += 1
            self.img_cap[k] = curr_sent

        for k in self.test:
            v = capts[k]
            curr_sent = [self.word_id[w] for w in v if w in self.word_id]
            if len(curr_sent) > caption_max_len:
                curr_sent = curr_sent[:caption_max_len]
            self.img_cap[k] = curr_sent

        self.caption_max_len = caption_max_len
        self.id_words = {v: k for k, v in self.word_id.items()}
        self.batch_size = batch_size
        self.img_names = list(self.img_feat.keys())
        pass

    def get_dataset(self, files):
        #x_cap = np.zeros((len(files), self.caption_max_len), dtype=np.int32)
        x_cap = np.zeros((len(files), self.caption_max_len, len(self.word_id)))
        x_img = np.zeros((len(files), self.len_img_feat))
        y = np.zeros((len(files), self.caption_max_len, len(self.word_id)), dtype=np.int8)
        for i, f in enumerate(files):
            cap = self.img_cap[f]
            x_img[i, :] = self.img_feat[f]
            for j, v in enumerate(cap):
                x_cap[i, j, v] = 1
            #for j, v in enumerate(cap):
            #    x_cap[i, j] = v
            for j, v in enumerate(cap[1:]):
                y[i, j, v] = 1
        return [x_cap, x_img], y

    def train_set(self):
        return self.get_dataset(self.train)

    def test_set(self):
        return self.get_dataset(self.test)
