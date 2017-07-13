from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf
import facenet
from pandas import DataFrame, concat
import numpy as np
from sklearn.neighbors import NearestNeighbors

from get_celebrities import get_celebrities_name

def load_images(images_dir, image_size=160, celeb=False):
    images_path = []
    labels = []
    photo_names = []
    for employee_dir in os.listdir(images_dir):
        if employee_dir.endswith(".txt"):
            continue
        for employee_img in os.listdir(os.path.join(images_dir,employee_dir)):
            images_path.append(os.path.join(images_dir,employee_dir,employee_img))
            photo_names.append(employee_img)
            if celeb:
                labels.append(get_celebrities_name(employee_img))
            else:
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


def load_como_pictures(images_dir, image_size=160):
    images_path = []
    photo_labels = []
    photo_names = []
    for folder in os.listdir(images_dir):
        if folder.endswith(".txt"):
            continue
        for face_img in os.listdir(os.path.join(images_dir, folder)):
            images_path.append(os.path.join(images_dir,folder,face_img))
            photo_labels.append(folder)
            photo_names.append(face_img)
    images = facenet.load_data(images_path, False, False, image_size)
    return images, photo_labels, photo_names


def load_celeb_others_pictures(images_dir, image_size=160):
    images_path = []
    photo_labels = []
    photo_names = []
    for folder in os.listdir(images_dir):
        if folder.endswith(".txt"):
            continue
        for celeb_img in os.listdir(os.path.join(images_dir, folder)):
            images_path.append(os.path.join(images_dir,folder,celeb_img))
            photo_names.append(celeb_img)
            ext = celeb_img.split(folder)[1]
            if len(ext.split("_")) == 2:
                ext2 = "_" + ext.split("_")[1].split(".png")[0]
            elif len(ext.split("_")) == 3:
                ext2 = "_" + ext.split("_")[1]
            photo_labels.append(folder + ext2)
    images = facenet.load_data(images_path, False, False, image_size)
    return images, photo_labels, photo_names



def como_automatic_tagging(cut=0.9):
    model = "models/20170512-110547"
    como_dir = "data/como_pictures/faces"
    employees_dir = "data/lm_employees/mtcnnpy_160_clean"
    # out = "data/como_pictures/tagging_cut_{0}.txt".format(str(cut).replace(".","_"))
    out = "data/como_pictures/tagging_f2.txt"
    emp_images, emp_labels, emp_photo_names = load_images(employees_dir)
    como_images, como_labels, como_photo_names = load_como_pictures(como_dir)

    nb_emp_photos = len(emp_photo_names)
    nb_como_images = len(como_photo_names)
    batch = np.vstack((emp_images, como_images))
    emb = calculate_embeddings(batch, model)
    emp_embeddings = emb[0:nb_emp_photos,:]
    como_embeddings = emb[nb_emp_photos:,:]

    knn = NearestNeighbors()
    knn.fit(emp_embeddings)

    dist, ind = knn.kneighbors(como_embeddings, n_neighbors=5)

    emp_labels_df = DataFrame({"emp_labels" : emp_labels, "emp_photo_names": emp_photo_names})
    como_labels_df = DataFrame({"como_labels" : como_labels, "como_photo_names" : como_photo_names})

    como_unique_photos = como_labels_df["como_labels"].unique()

    with open(out, "w") as f:
        for photo in como_unique_photos:
            tagged = []
            idx = como_labels_df.loc[como_labels_df["como_labels"] == photo,:].index

            for dist_i, ind_i in zip(dist[idx], ind[idx]):
                # if dist_i[0] >= cut:
                #     continue
                # guy = emp_labels_df.loc[ind_i[0], "emp_labels"]
                # tagged.append(guy)
                guys = emp_labels_df.loc[ind_i[0:2], "emp_labels"]
                if guys.unique().shape[0] > 1:
                    continue
                else:
                    guy = guys.unique()[0]
                    tagged.append(guy)

            txt = "\t".join([photo] + list(set(tagged)))
            print(photo, tagged)
            f.write(txt)
            f.write("\n")


def celebrities_automatic_tagging():
    model = "models/20170512-110547"
    celeb_dir = "data/celebrities/mtcnnpy_160"
    celeb_others_dir = "data/celebrities_with_others/faces"
    # out = "data/como_pictures/tagging_cut_{0}.txt".format(str(cut).replace(".","_"))
    out = "data/celebrities_with_others/tagging_f2.txt"
    celeb_images, celeb_labels, celeb_photo_names = load_images(celeb_dir, celeb=True)
    celeb_others_images, celeb_others_labels, celeb_others_photo_names = load_celeb_others_pictures(celeb_others_dir)

    nb_celeb_photos = len(celeb_photo_names)
    nb_celeb_others_images = len(celeb_others_photo_names)
    batch = np.vstack((celeb_images, celeb_others_images))
    emb = calculate_embeddings(batch, model)
    celeb_embeddings = emb[0:nb_celeb_photos,:]
    celeb_others_embeddings = emb[nb_celeb_photos:,:]

    knn = NearestNeighbors()
    knn.fit(celeb_embeddings)

    dist, ind = knn.kneighbors(celeb_others_embeddings, n_neighbors=5)

    celeb_labels_df = DataFrame({"celeb_labels" : celeb_labels, "celeb_photo_names": celeb_photo_names})
    celeb_others_labels_df = DataFrame({"celeb_others_labels" : celeb_others_labels, "celeb_others_photo_names" : celeb_others_photo_names})

    celeb_others_unique_photos = celeb_others_labels_df["celeb_others_labels"].unique()

    with open(out, "w") as f:
        for photo in celeb_others_unique_photos:
            tagged = []
            idx = celeb_others_labels_df.loc[celeb_others_labels_df["celeb_others_labels"] == photo,:].index

            for dist_i, ind_i in zip(dist[idx], ind[idx]):
                # if dist_i[0] >= cut:
                #     continue
                # guy = emp_labels_df.loc[ind_i[0], "emp_labels"]
                # tagged.append(guy)
                guys = celeb_labels_df.loc[ind_i[0:2], "celeb_labels"]
                if guys.unique().shape[0] > 1:
                    continue
                else:
                    guy = guys.unique()[0]
                    tagged.append(guy)

            txt = "\t".join([photo] + list(set(tagged)))
            print(photo, tagged)
            f.write(txt)
            f.write("\n")




if __name__ == '__main__':
    # images_dir = "data/lm_employees/mtcnnpy_160_clean"
    images_dir = "data/celebrities/mtcnnpy_160"
    model = "models/20170512-110547"

    images, labels, photo_names = load_images(images_dir)

    emb = calculate_embeddings(images, model)

    labels_df = DataFrame({"labels" : labels, "photo_names": photo_names})

    knn = NearestNeighbors()
    knn.fit(emb)
