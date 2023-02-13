"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np
from collections import deque

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected_coord from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected_coord.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected_coord character.
        h: height of the detected_coord character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected_coord characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    enrollment(characters)

    character_coord = detection(test_img)
    
    names = recognition(test_img, character_coord)
    
    results = []
    for i in range(len(character_coord)):
        results.append( {"bbox": character_coord[i], "name": names[i]} )

    # raise NotImplementedError
    return results

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    
    vectors = {}
    
    siftExtractor = cv2.SIFT_create()

    for ch in characters:
        if ch[0]=='dot':
            continue
        keypts, desp = siftExtractor.detectAndCompute(ch[1], None)
        vectors[ch[0]] = desp.tolist()

    with open('vectors.json', "w") as file:
        json.dump(vectors, file)

def match_vector(descriptor1, descriptor2):
    key_dist = np.zeros((len(descriptor1),len(descriptor2)))
    for i in range(descriptor1.shape[0]):
        for j in range(descriptor2.shape[0]):
            d = np.square(descriptor1[i] - descriptor2[j])
            ssd = np.sum(d)
            key_dist[i][j] = ssd
    
    ratio = []

    for i in range(descriptor1.shape[0]):
        sort_key = np.argsort(key_dist[i])
        bm,sbm = sort_key[0],sort_key[1]
        if key_dist[i][bm] / key_dist[i][sbm] < 0.45:
            ratio.append(bm)
    
    return ratio





def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.

    # implement connected component labeling to detect various candidate characters
    # Total number of characters = 142 + 1 background
    # characters can be between 1/2 and 2x the size of the enrolled images.
    # generate features/ resize to prepare for recogniton

    binary_threshold = np.zeros(test_img.shape)
    rows, columns = test_img.shape
    for i in range(rows):
        for j in range(columns):
            binary_threshold[i][j] = 1 if test_img[i][j]<133 else 0
    
    detected_coord = []
    visited_coord = set()

    def graph_DFS(x,y):
        graph_DFS_queue = deque()
        graph_DFS_queue.append((x,y))
        min_coord_x, min_coord_y, width, height = 9999, 9999, -1000, -1000
        while graph_DFS_queue:
            i,j = graph_DFS_queue.pop()
            min_coord_x = min(min_coord_x, j)
            min_coord_y = min(min_coord_y, i)
            width = max(j-min_coord_x, width)
            height = max(i-min_coord_y, height)
            visited_coord.add((i,j))
            for a,b in [(-1,0), (1,0), (0,1), (0,-1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                if 0<=i+a<rows and 0<=j+b<columns and (i+a,j+b) not in visited_coord and binary_threshold[i+a][j+b]==1:
                    graph_DFS_queue.append((i+a,j+b))

        return [min_coord_x, min_coord_y, width+1, height+1]

    for i in range(rows):
        for j in range(columns):
            if (i,j) not in visited_coord and binary_threshold[i][j] == 1:
                detected_coord.append(graph_DFS(i,j))
            else:
                continue

    return detected_coord


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)

def recognition(test_img, character_coord):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    # using features of the enrollment characters
    # implement matching for all recognition
    
    with open('vectors.json', "r") as dfile:
        vectors = json.load(dfile)

    names = []
    siftExtractor = cv2.SIFT_create()

    for bbox in character_coord:
        x,y,w,h = bbox
        image = test_img[y:y+h,x:x+w].astype('uint8')
        key1,desc1 = siftExtractor.detectAndCompute(image, None)

        if len(key1) == 0:
            names.append('dot')
            continue

        match_desp = []
        for char in vectors:
            desc2 = np.array(vectors[char])
            match_desp.append((char,match_vector(desc1, desc2)))
        match_desp.sort(key= lambda x: len(x[1]), reverse=True)

        if len(match_desp[0][1]) > 0:
            names.append(match_desp[0][0])
        else:
            names.append("UNKNOWN")
    
    return names


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)

if __name__ == "__main__":
    main()