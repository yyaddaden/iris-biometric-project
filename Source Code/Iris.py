# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, os.path
import csv

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, warp_polar
from skimage.feature import canny, graycomatrix, graycoprops
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from PIL import Image
from pathlib import Path
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

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

        sample = ["005", "010", "013", "022", "027", "030", "032", "035", "038", "040"]
        GLCMs_1D = []
        Labels = []

        for f in os.listdir(path):
            if f in sample:
                for c in os.listdir(os.path.join(path, f)):
                    for i in os.listdir(os.path.join(path, f, c)):
                        GLCMs_1D.append(self.glcm(os.path.join(path, f, c, i)))
                        Labels.append(f)

        PCA = []
        PCA = self.pca(GLCMs_1D, min(len(GLCMs_1D), len(GLCMs_1D[0])))

        self.classification(PCA, Labels)

        #self.create_glcm_pca_csv(GLCMs_1D, Labels)

    
    def create_glcm_pca_csv(self, GLCMs_1D, Labels):
        number_of_file = 10

        for i in range(number_of_file):
            PCA_components = []
            header = []
            n_components=(i+1)*10
            PCA_components = self.pca(GLCMs_1D, n_components)
        
            for j in range(len(PCA_components[0])):
                header.append("C" + str(j))
        
            header.append("ID")
        
            PCA_components_labels = [[]]*len(PCA_components)
        
            for j in range(len(PCA_components)):
                PCA_components_labels[j] = np.append(PCA_components[j], Labels[j]).tolist()
        
            with open('glcm-sample' + str(i) + '.csv', 'w', encoding='UTF8', newline='') as f:
                # create the csv writer
                writer = csv.writer(f)
        
                writer.writerow(header) 
        
                for row in PCA_components_labels:
                    # write a row to the csv file
                    writer.writerow(row)                


    def glcm(self, file_path):
        image = np.array(Image.open(file_path), dtype=np.uint8)
        image = rgb2gray(image)
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = np.array(image, dtype=np.uint8)
        #glcm = graycomatrix(image, distances=[5], angles=[0], levels=256)
        glcm = graycomatrix(image, distances=[10], angles=[0], levels=256)
        glcm_1D = glcm[:, :, 0, 0].flatten()

        #plt.imshow(glcm[:, :, 0, 0])
        #plt.show()

        return glcm_1D.tolist()

    def pca(self, GLCMs_1D, number_of_components):
        pca = PCA(n_components=number_of_components)
        return pca.fit_transform(GLCMs_1D)

    def classification(self, data, labels):
        X = data
        y = np.array(labels)

        # p=1 for manhattan distance
        knn = KNeighborsClassifier(n_neighbors=2, weights='distance', p=1)
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        k_folds = skf.split(X, y)

        n = 0
        score_sum = 0

        for train_index, test_index in k_folds:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            knn.fit(X_train, y_train)
            print(y_test)
            print(knn.predict(X_test))
            #print(knn.score(X_test, y_test))
            score_sum += knn.score(X_test, y_test)
            n += 1

        print(score_sum/n)
            
    def glcm_features(self):
        base_path = Path(__file__)
        path = (base_path / "../Prepared-MMU-Iris-Database/").resolve()

        sample = ["005", "010", "013", "022", "027", "030", "032", "035", "038", "040"]
        GLCM_features = [[]]
        index = 1

        GLCM_features[0] = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM", "ID"]

        for f in os.listdir(path):
            if f in sample:
                for c in os.listdir(os.path.join(path, f)):
                    for i in os.listdir(os.path.join(path, f, c)):
                        image = np.array(Image.open(os.path.join(path, f, c, i)), dtype=np.uint8)
                        image = rgb2gray(image)
                        image = Image.fromarray((image * 255).astype(np.uint8))
                        image = np.array(image, dtype=np.uint8)
                        glcm = graycomatrix(image, distances=[10], angles=[0], levels=256)

                        GLCM_features.append([])

                        GLCM_features[index].append(graycoprops(glcm, 'contrast')[0][0])
                        GLCM_features[index].append(graycoprops(glcm, 'dissimilarity')[0][0])
                        GLCM_features[index].append(graycoprops(glcm, 'homogeneity')[0][0])
                        GLCM_features[index].append(graycoprops(glcm, 'energy')[0][0])
                        GLCM_features[index].append(graycoprops(glcm, 'correlation')[0][0])
                        GLCM_features[index].append(graycoprops(glcm, 'ASM')[0][0])
                        GLCM_features[index].append(f)

                        index += 1

        with open('glcm_features_sample.csv', 'w', encoding='UTF8', newline='') as f:
                # create the csv writer
                writer = csv.writer(f)
        
                for row in GLCM_features:
                    # write a row to the csv file
                    writer.writerow(row)       

        
