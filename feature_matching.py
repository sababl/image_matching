import cv2
import numpy
import argparse

from functions import *

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Feature Matching between two images.")
    parser.add_argument("-s", "--source", type=str, help="Path to the first image", default='.\images\image_1.jpg')
    parser.add_argument("-d", "--destination", type=str, help="Path to the second image", default='.\images\image_2.jpg')
    
    # Parse the arguments
    args = parser.parse_args()

    img1 = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.destination, cv2.IMREAD_GRAYSCALE)

    # Compute matches and keypoints
    kp1, kp2, matches, affinity_matrix = feature_matching(img1, img2)

    # Show the matches
    show_matches(args.source, args.destination, matches, kp1, kp2)