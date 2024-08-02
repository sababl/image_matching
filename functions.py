import cv2
import numpy as np

def extract_patches_around_corners(image, corners, patch_size=32):
    """
    Extract patches of the image around each detected corner.

    Parameters:
    - image_path (str): Path to the input image.
    - patch_size (int): Size of the patches to extract. Default is 32.

    Returns:
    - patches (list): List of extracted patches.
    """

    patches = []
    half_patch_size = patch_size // 2

    for corner in corners:
        x, y = corner.ravel()

        # Extract patch around the corner
        x_start = max(x - half_patch_size, 0)
        x_end = min(x + half_patch_size, image.shape[1])
        y_start = max(y - half_patch_size, 0)
        y_end = min(y + half_patch_size, image.shape[0])

        patch = image[y_start:y_end, x_start:x_end]
        patches.append(patch)

    return patches


def compute_affinity_matrix(des1, des2):
    """
    Compute the affinity matrix based on the Euclidean distance between descriptors.

    Parameters:
    - des1 (np.ndarray): Descriptors from image 1.
    - des2 (np.ndarray): Descriptors from image 2.

    Returns:
    - affinity_matrix (np.ndarray): The m x n affinity matrix.
    """
    m = des1.shape[0]
    n = des2.shape[0]
    affinity_matrix = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            affinity_matrix[i, j] = np.linalg.norm(des1[i] - des2[j])
    
    return affinity_matrix


def detect_matches(affinity_matrix):
    """
    Detect matches based on the affinity matrix.

    Parameters:
    - affinity_matrix (np.ndarray): The m x n affinity matrix.

    Returns:
    - matches (list of tuples): List of matched indices.
    """
    m, n = affinity_matrix.shape
    matches = []

    for i in range(m):
        min_index = np.argmin(affinity_matrix[i])
        matches.append((i, min_index))

    return matches

def feature_matching(img1, img2):
    """
    Perform feature matching between two images.

    Parameters:
    - image1_path (str): Path to the first image.
    - image2_path (str): Path to the second image.

    Returns:
    - matches (list of tuples): List of matched indices.
    - affinity_matrix (np.ndarray): The m x n affinity matrix.
    """

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Compute the affinity matrix
    affinity_matrix = compute_affinity_matrix(des1, des2)
    
    # Detect matches
    matches = detect_matches(affinity_matrix)
    
    return kp1, kp2, matches, affinity_matrix



def show_matches(image1_path, image2_path, matches, kp1, kp2):
    """
    Visualize the matches between two images.

    Parameters:
    - image1_path (str): Path to the first image.
    - image2_path (str): Path to the second image.
    - matches (list of tuples): List of matched indices from feature_matching.
    - kp1 (list): Keypoints from the first image.
    - kp2 (list): Keypoints from the second image.
    
    Returns:
    - None: Displays the matched images.
    """
    # Read the images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Create DMatch objects for drawing
    dmatches = []
    for i, (img1_idx, img2_idx) in enumerate(matches):
        dmatches.append(cv2.DMatch(_imgIdx=0, _queryIdx=img1_idx, _trainIdx=img2_idx, _distance=0))

    # Draw matches
    matched_image = cv2.drawMatches(img1, kp1, img2, kp2, dmatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesThickness=1)
    
    # Display the image with matches
    cv2.imshow("Matches", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
