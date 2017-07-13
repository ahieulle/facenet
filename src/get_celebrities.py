import os
import unicodedata
import urllib
import random
import itertools
try:
    from urllib.request import HTTPError
    from http.client import RemoteDisconnected
except ImportError:
    pass

import numpy as np
from pandas import read_csv, DataFrame

def clean_names(name):
    #rm unicode and accents
    name = unicodedata.normalize('NFD', name).encode('ascii','ignore').decode('utf-8')
    name = name.rstrip()
    name = name.replace(" ","_")
    name = name.replace(".","")
    name = name.lower()
    return name

def get_celebrities():
    path = os.path.join("data", "celebrities", "celebrities.csv")
    df = read_csv(path, sep="\t", header=None, encoding="utf-8")
    photos_columns = ["photo_{0}".format(i+1) for i in range(10)]
    df.columns = ["name"] + photos_columns
    df.fillna("", inplace=True)

    df["name"] = df["name"].apply(clean_names)

    def download_images(row):
        row = row[1]
        name = row["name"]

        wrong = []
        folder = os.path.join("data", "celebrities", name)
        if not os.path.exists(folder):
            os.mkdir(folder)

        for i, k in enumerate(photos_columns):
            url = row[k]
            if url == "":
                continue
            print("Downloading {0} image : {1}".format(name, url))
            namei = "{0}_{1}.png".format(name, i)
            path = os.path.join(folder, namei)
            try:
                urllib.request.urlretrieve(url, path)
            except HTTPError:
                wrong.append(url)
            except RemoteDisconnected:
                wrong.append(url)
            except Exception:
                wrong.append(url)

        return wrong
    wrongs = []
    for row in df.iterrows():
        wrong = download_images(row)
        wrongs.append(wrong)

    return wrongs


def get_celebrities_with_others():
    path = os.path.join("data", "celebrities_with_others", "celebrities_with_others.csv")
    def get_real_url(url):
        google =  "https://www.google.fr/imgres?imgurl="
        pattern = "&imgrefurl"
        if google in url:
            url = url.split(google)[1].split(pattern)[0]
            url = urllib.parse.unquote(url)
        return url

    df = read_csv(path, sep="\t", encoding="utf-8")
    df.columns = ["celeb1", "celeb2", "celeb3", "nfaces", "where", "link"]
    df.fillna("", inplace=True)

    for c in ["celeb1", "celeb2", "celeb3"]:
        df[c] = df[c].apply(clean_names)

    df["link"] = df["link"].apply(get_real_url)

    wrong = []

    for gr in df.groupby("celeb1"):
        name = gr[0]
        links = gr[1]["link"]

        folder = os.path.join("data", "celebrities_with_others", name)
        if not os.path.exists(folder):
            os.mkdir(folder)

        for i,url in enumerate(links):
            namei = "{0}_{1}.png".format(name, i)
            out = os.path.join(folder, namei)

            print("Downloading {0} image : {1}".format(namei, url))
            try:
                urllib.request.urlretrieve(url, out)
            except HTTPError:
                wrong.append(url)
            except RemoteDisconnected:
                wrong.append(url)
            except Exception:
                wrong.append(url)

    return wrong




def get_celebrities_name(celeb_img):
    sp = celeb_img.split("_")
    if len(sp) == 2:
        return sp[0]
    elif len(sp) > 2:
        nb_ = len(sp)
        last = nb_ - 1
        return "_".join(sp[0:last])

def generate_celebrities_pairs(images_dir, nb_pairs=10000):
    images_path = []
    celeb = []
    photo_names = []
    for celeb_dir in os.listdir(images_dir):
        if celeb_dir.endswith(".txt"):
            continue
        for celeb_img in os.listdir(os.path.join(images_dir,celeb_dir)):
            images_path.append(os.path.join(images_dir,celeb_dir,celeb_img))
            photo_names.append(celeb_img)
            celeb.append(get_celebrities_name(celeb_img))

    pairs = []
    issame = []
    df = DataFrame({"celeb" : celeb, "path": images_path})


    for i in range(int(nb_pairs/2)):
        random_idx = random.randint(0, df.shape[0] - 1)
        name = df.iloc[random_idx]["celeb"]
        same = df.loc[df["celeb"] == name].sample(2)

        pairs.append((same.iloc[0]["path"], same.iloc[1]["path"]))
        issame.append(True)

    for i in range(int(nb_pairs/2)):
        random_idx = random.randint(0, df.shape[0] - 1)
        name = df.iloc[random_idx]["celeb"]
        path1 = df.iloc[random_idx]["path"]
        not_same = df.loc[df["celeb"] != name].sample(1)
        path2 = not_same.iloc[0]["path"]

        pairs.append((path1, path2))
        issame.append(False)

    p = np.random.permutation(len(issame))

    pairs = list(itertools.chain.from_iterable(np.array(pairs)[p]))
    issame = np.array(issame)[p].tolist()

    return pairs, issame
