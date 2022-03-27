# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, os.path
import csv

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, warp_polar
from skimage.feature import canny, graycomatrix
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from PIL import Image
from pathlib import Path
from skimage.color import rgb2gray
from sklearn.decomposition import PCA

class Iris:
    def __init__(self):
        #####
        #matplotlib.use('Agg')
        pass

    def process_data(self):
        base_path = Path(__file__)
        path = (base_path / "../MMU-Iris-Database/").resolve()
        for f in os.listdir(path):
            for c in os.listdir(os.path.join(path, f)):
                for i in os.listdir(os.path.join(path, f, c)):
                    self.process_image(os.path.join(path, f, c, i))

    def process_image(self, file_path):
        # Load picture and detect edges
        image = np.array(Image.open(file_path), dtype=np.uint8)
        image = rgb2gray(image)
        edges = canny(image, sigma=1, low_threshold=0.1, high_threshold=0.3)

        # Detect two radii
        hough_radii = np.arange(50, 100, 2)
        hough_res = hough_circle(edges, hough_radii)

        # Select the most prominent 3 circles
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                total_num_peaks=1)

        # Draw them
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        image = color.gray2rgb(image)

        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius,
                                            shape=image.shape)
            image[circy, circx] = (220, 20, 20)

        warped = warp_polar(image, (cy[0], cx[0]), radius=radii[0], output_shape=(image.shape[1]*2, image.shape[0]*2), 
            scaling='linear', channel_axis=2)

        warped = list(zip(*warped[::-1]))
        warped_array = np.array(warped)
        warped_image = Image.fromarray((warped_array * 255).astype(np.uint8))

        cropped_warped_image = []

        for x in range(warped_image.height):
            black_percentage = 0
            black_number = 0
            for y in range(warped_image.width):
                if warped_image.getpixel((y,x)) < (45, 45, 45):
                    black_number = black_number + 1

            black_percentage = black_number / warped_image.width
            if black_percentage < 0.5:
                cropped_warped_image.append(warped[x])

        new_file_path = file_path.replace("MMU-Iris-Database", "Prepared-MMU-Iris-Database")

        if not os.path.exists(new_file_path.rsplit('\\',1)[0]):
            os.makedirs(new_file_path.rsplit('\\',1)[0])
        Image.fromarray((np.array(cropped_warped_image) * 255).astype(np.uint8)).save(new_file_path)


    def create_glcm(self):
        base_path = Path(__file__)
        path = (base_path / "../Prepared-MMU-Iris-Database/").resolve()

        features = []

        for f in os.listdir(path):
            for c in os.listdir(os.path.join(path, f)):
                for i in os.listdir(os.path.join(path, f, c)):
                    features.append(self.glcm(os.path.join(path, f, c, i)))

        self.pca(features)

        with open('glcm.csv', 'w', encoding='UTF8', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            for row in features:
                # write a row to the csv file
                writer.writerow(row)


    def glcm(self, file_path):
        image = np.array(Image.open(file_path), dtype=np.uint8)
        image = rgb2gray(image)
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = np.array(image, dtype=np.uint8)
        glcm = graycomatrix(image, distances=[5], angles=[0], levels=256)
        glcm_1D = glcm[:, :, 0, 0].flatten()

        #plt.imshow(glcm[:, :, 0, 0])
        #plt.show()

        return glcm_1D.tolist()

    def pca(self, features):
        pca = PCA()
        pca.fit(features)
        #print(pca.explained_variance_ratio_)
        #print(pca.singular_values_)

    def classification(self, data):
        # X, y ?
        knn = KNeighborsClassifier(n_neighbors=3)
        scores = cross_val_score(knn, X, y, cv=10)
        print(scores)
