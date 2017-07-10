import os
from PIL import Image

def resize_lm_dataset():
    root = os.path.join("data", "lm_employees")
    raw_path = os.path.join(root, "raw")
    resize_path = os.path.join(root, "resize250")
    if not os.path.exists(resize_path):
        os.makedirs(resize_path)
    for employee in os.listdir(raw_path):
        employee_dir = os.path.join(raw_path, employee)
        if not os.path.exists(os.path.join(resize_path, employee)):
            os.makedirs(os.path.join(resize_path, employee))
        for (i, img_name) in enumerate(os.listdir(employee_dir)):
            if img_name.startswith("."):
                continue
            img_path = os.path.join(employee_dir, img_name)
            im = Image.open(img_path)
            resize_name = "{0}_{1}.jpg".format(employee,i)
            img_out = os.path.join(resize_path, employee, resize_name)
            im.resize((250,250)).save(img_out)


def clean_lm_dataset():
    root = os.path.join("data", "lm_employees")
    raw_path = os.path.join(root, "raw")
    clean_path = os.path.join(root, "clean")
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
    for employee in os.listdir(raw_path):
        employee_dir = os.path.join(raw_path, employee)
        if not os.path.exists(os.path.join(clean_path, employee)):
            os.makedirs(os.path.join(clean_path, employee))
        for (i, img_name) in enumerate(os.listdir(employee_dir)):
            if img_name.startswith("."):
                continue
            img_path = os.path.join(employee_dir, img_name)
            im = Image.open(img_path)
            resize_name = "{0}_{1}.jpg".format(employee,i)
            img_out = os.path.join(clean_path, employee, resize_name)
            im.save(img_out)
