import re, os
import numpy as np

def load_pictures(path):
    toReturn = list()
    for file in sorted(os.listdir(path)):
        toReturn.append(os.path.join(path,file))
    return toReturn


def load_translations(path):
    returnTrans= list()
    returnRot = list()
    for translation in sorted(os.listdir(path)):
        filePath = os.path.join(path, "trans")
        b = np.loadtxt(filePath, dtype=float)
        rot  = b[:3,:3]
        rot = np.reshape(rot,newshape=(-1))
        trans = b[:3,3]
        trans = np.reshape(trans, newshape=(-1))
        returnTrans.append(trans)
        returnRot.append(rot)

    return  returnTrans, returnRot

def load_data(path):
    rgb = load_pictures(os.path.join(path,"rgb"))
    depth = load_pictures(os.path.join(path,"depth"))
    trans, rot = load_translations(os.path.join(path,"trans"))
    return [rgb, depth,trans,rot]
