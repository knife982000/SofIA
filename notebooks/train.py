import tensorflow as tf
#Porcentaje de memoria de la GPU a usar en este caso 33.3% Admite hasta 3 procesos.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
x = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).__enter__()
#Esto es para ver que la session se definio correctamente
print(tf.get_default_session())

from model import create_model, ImageGen
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

if __name__ == '__main__':
    data_gen = ImageGen('images_features', 'images_captions', caption_max_len=30, min_reps=2)
    print(len(data_gen.img_feat))

    model = create_model(len(data_gen.word_id), data_gen.len_img_feat, 30)
    model.summary()

    x, y = data_gen.train_set()
    #import numpy as np
    #print(np.argmax(x[0][0, :, :], axis=-1))
    #print(np.argmax(y[0, :, :], axis=-1))
    #quit()
    #for i in range(5):
    #    import time
    #    s = time.clock()
    #    model.fit(x, y, epochs=1, batch_size=512,
    #              callbacks=[ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
    #                         ModelCheckpoint('weights.{epoch:02d}-{loss:.2f}.hdf5')],
    #              validation_data=data_gen.train_set())
    #    print("Time: {}".format(time.clock() - s))
    #quit()
    model.fit(x, y, epochs=5000, batch_size=512,
              callbacks=[ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
                         ModelCheckpoint('weights.{epoch:02d}-{loss:.2f}.hdf5', verbose=1, period=100)],
              validation_data=data_gen.test_set())

    model.save('caption_network.h5')