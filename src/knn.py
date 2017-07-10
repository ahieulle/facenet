from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf
import facenet
from pandas import DataFrame, concat
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_images(images_dir, image_size=160):
    images_path = []
    labels = []
    photo_names = []
    for employee_dir in os.listdir(images_dir):
        if employee_dir.endswith(".txt"):
            continue
        for employee_img in os.listdir(os.path.join(images_dir,employee_dir)):
            images_path.append(os.path.join(images_dir,employee_dir,employee_img))
            photo_names.append(employee_img)
            labels.append(get_name(employee_img))
    # return images_path
    images = facenet.load_data(images_path, False, False, image_size)
    return images, labels, photo_names

def get_name(employee_img):
    sp = employee_img.split("_")
    if len(sp) == 2:
        return sp[0]
    elif len(sp) == 3:
        return "_".join(sp[0:2])

def calculate_embeddings(images, model):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)

    return emb

def predict(employee_name, knn_model, labels_df, emb):
    index = labels_df[labels_df["labels"] == employee_name].sample(1).index[0]

    this_emb = emb[index].reshape(1,-1)

    distances, indices = knn_model.kneighbors(this_emb, return_distance=True, n_neighbors=20)

    res_df = DataFrame({"distances" : distances[0], "indices": indices[0]})
    res_df.index = res_df["indices"]

    res_df = concat([res_df, labels_df.loc[res_df["indices"]]], axis=1)
    res_df.sort_values("distances", ascending=True, inplace=True)

    return res_df

def evaluate_model_knn(knn_model, labels_df, emb, top=5):
    counts = DataFrame(labels_df.groupby("labels").size().rename('counts'))
    names = np.unique(counts[counts["counts"] >= 5].index)

    good = 0.0
    for name in names:
        pred = predict(name, knn_model, labels_df, emb)
        pred = pred.iloc[0:top]

        if pred[pred["labels"] == name].shape[0] == top:
            good += 1
        else:
            print ("wrong for {0}".format(name))

    print ("Accuracy : {0:0.2f}".format(good / len(names)))


if __name__ == '__main__':
    # images_dir = "data/lm_employees/mtcnnpy_160_clean"
    images_dir = "data/celebrities/mtcnnpy_160"
    model = "models/20170512-110547"

    images, labels, photo_names = load_images(images_dir)

    emb = calculate_embeddings(images, model)

    labels_df = DataFrame({"labels" : labels, "photo_names": photo_names})

    knn = NearestNeighbors()
    knn.fit(emb)
