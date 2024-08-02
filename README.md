# Feature Matching and Visualization using OpenCV

This project implements a feature matching procedure between two images using OpenCV. The program detects keypoints in both images, computes an affinity matrix based on descriptor distances, identifies the best matches, and visualizes these matches by drawing lines between corresponding keypoints.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The project uses ORB (Oriented FAST and Rotated BRIEF) for feature detection and descriptor extraction. It then computes an affinity matrix between the descriptors of the two images, finds the best matches by minimizing the distance, and visualizes these matches.

## Features

- Detects keypoints and computes descriptors using ORB.
- Computes an affinity matrix based on Euclidean distance between descriptors.
- Finds the best matches by minimizing distances in the affinity matrix.
- Visualizes the matches by drawing lines between corresponding keypoints.

### Prerequisites

- Python 3.6 or higher
- OpenCV

### Install OpenCV

You can install OpenCV using pip:

```bash
pip install opencv-python
```

# Clone the Repository
Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/feature-matching-opencv.git
cd feature-matching-opencv
```

# Usage
You can run the feature matching script directly from the command line. The script accepts optional arguments for the image paths. If not provided, it uses default images.

# Running the Script
```bash
python feature_matching.py -s path_to_image1.jpg -d path_to_image2.jpg

```

# Project Structure

├── feature_matching.py     # Main script for feature matching and visualization
├── README.md               # Project documentation
├── default_image1.jpg      # Default first image (if provided)
├── default_image2.jpg      # Default second image (if provided)
└── images/                 # Directory for storing example images
