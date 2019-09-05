import json
import numpy as np
import re
import os


def get_data_from_json(path, filename, finalpath="/home/m320/robot40human_ws/src/data_collector/rgb_bbox"):
    with open(path) as f:
        data = json.load(f)

    for x in data["shapes"]:
        a = x["points"]
        b = np.asarray(a)
        filepath = os.path.join(finalpath, filename)
        np.savetxt(filepath, b, fmt="%f")


def load_files(path):
    for file in sorted(os.listdir(path)):
        filePath = os.path.join(path, file)
        get_data_from_json(filePath, os.path.splitext(file)[0])


if __name__ == '__main__':
    path = "/home/m320/robot40human_ws/src/data_collector/rgb_json"
    load_files(path)
