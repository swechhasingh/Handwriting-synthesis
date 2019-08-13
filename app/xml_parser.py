from xml.dom import minidom
import numpy as np 
from pathlib import Path
import sys
import os
sys.path.append('../')
from utils import plot_stroke

# for flask app
def path_string_to_stroke(path, str_len, down_sample=False):
    path_data = path.split(" ")[:-1]
    print(len(path_data))
    stroke = np.zeros((len(path_data), 3))
    i = 0
    while i < len(path_data):
        command = path_data[i][0]
        coord = path_data[i][1:].split(',')
        if command == 'M':
            stroke[i,0] = 1.0
        elif command == 'L':
            stroke[i,0] = 0.0
        stroke[i,1] = float(coord[0])
        stroke[i,2] = -float(coord[1])
        i += 1

    stroke[0,0] = 0.0 
    stroke[-1, 0] = 1. 
    print("initial shape of data: ", stroke.shape)
    cuts = np.where(stroke[:, 0] == 1.)[0]
    print("EOS index:",cuts)

    k = 1
    ratio = len(stroke) // str_len
    print("LPC: ", ratio)
    if ratio > 30 or down_sample:
        k = 2
        print("downsampling by 2")
    
    start = 0
    down_sample_data = []
    for eos in cuts:
        down_sample_data.append(stroke[start:eos:k])
        down_sample_data.append(stroke[eos])
        start = eos + 1
    down_sample_stroke = np.vstack(down_sample_data)
    # convert absolute coordinates into offset 
    down_sample_stroke[1:,1:] = down_sample_stroke[1:,1:] - down_sample_stroke[:-1,1:]
    print("After downsampling shape of data: ",len(down_sample_stroke))
    
    return down_sample_stroke

def svg_xml_parser(svg_path="./mobile/writing_8.svg"):
    """
        Extract path data from svg xml file
        return:
            path_data: list of points in path
    """
    doc = minidom.parse(svg_path)
    path_strings = [path.getAttribute('d') for path
                in doc.getElementsByTagName('path')]
    doc.unlink()
    path = path_strings[0]
    # print(len(path))

    path_data = path.split(" ")[:-1]

    print(len(path_data))
    return path_data


def path_to_stroke(path_data, k=1, save_path="./mobile/"):
    """
        Convert svg path data into stroke data with offset coordinates
        args:
            path_data: list of svg path points
            k: downsample factor, default 1 means no downsampling
            save_path: directory path to save stroke.npy file
    """

    save_path = Path(save_path)
    stroke = np.zeros((len(path_data), 3))
    i = 0
    while i < len(path_data):
        command = path_data[i][0]
        coord = path_data[i][1:].split(',')
        if command == 'M':
            stroke[i,0] = 1.0
        elif command == 'L':
            stroke[i,0] = 0.0
        stroke[i,1] = float(coord[0])
        stroke[i,2] = -float(coord[1])
        i += 1

    stroke[0,0] = 0.0 
    stroke[-1, 0] = 1. 
    print("initial shape of data: ", stroke.shape)
    
    cuts = np.where(stroke[:, 0] == 1.)[0]
    print("EOS index:",cuts)

    start = 0
    down_sample_data = []
    for eos in cuts:
        down_sample_data.append(stroke[start:eos:k])
        down_sample_data.append(stroke[eos])
        start = eos + 1

    down_sample_stroke = np.vstack(down_sample_data)
    # convert absolute coordinates into offset 
    down_sample_stroke[1:,1:] = down_sample_stroke[1:,1:] - down_sample_stroke[:-1,1:]
    print("final shape of data: ", down_sample_stroke.shape)

    plot_stroke(down_sample_stroke, "img.png")
    np.save(save_path, down_sample_stroke, allow_pickle=True)

if __name__ == '__main__':

    path_data = svg_xml_parser(svg_path="./static/mobile/writing_10.svg")
    path_to_stroke(path_data, k=1, save_path="./static/mobile/style_10.npy")
    with open('./static/mobile/inpText_10.txt') as file:
        texts = file.read().splitlines()
    real_text = texts[0]
    print(len(list(real_text)))