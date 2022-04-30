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
        pass


    def pre_process_data(self):
        matplotlib.use('Agg')

        # Pre-process all images
        base_path = Path(__file__)
        path = (base_path / "../MMU-Iris-Database/").resolve()
        for f in os.listdir(path):
            for c in os.listdir(os.path.join(path, f)):
                for i in os.listdir(os.path.join(path, f, c)):
                    self.pre_process_image(os.path.join(path, f, c, i))


    def pre_process_image(self, file_path):
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

        # Transform iris into a rectangle image
        warped = warp_polar(image, (cy[0], cx[0]), radius=radii[0], output_shape=(image.shape[1]*2, image.shape[0]*2), 
            scaling='linear', channel_axis=2)

        warped = list(zip(*warped[::-1]))
        warped_array = np.array(warped)
        warped_image = Image.fromarray((warped_array * 255).astype(np.uint8))

        cropped_warped_image = []

        # Remove the black portion of the image (pupil)
        for x in range(warped_image.height):
            black_percentage = 0
            black_number = 0
            for y in range(warped_image.width):
                if warped_image.getpixel((y,x)) < (45, 45, 45):
                    black_number = black_number + 1

            black_percentage = black_number / warped_image.width
            if black_percentage < 0.5:
                cropped_warped_image.append(warped[x])

        # Save pre-processed images
        new_file_path = file_path.replace("MMU-Iris-Database", "Prepared-MMU-Iris-Database")

        if not os.path.exists(new_file_path.rsplit('\\',1)[0]):
            os.makedirs(new_file_path.rsplit('\\',1)[0])
        Image.fromarray((np.array(cropped_warped_image) * 255).astype(np.uint8)).save(new_file_path)


    def evaluation(self, bool_sample, n_folds=10, n_knn=2):
        base_path = Path(__file__)
        path = (base_path / "../Prepared-MMU-Iris-Database/").resolve()

        sample = ["005", "010", "013", "022", "027", "030", "032", "035", "038", "040"]
        GLCMs_1D = []
        Labels = []

        # Create 1D GLCM of all images
        for f in os.listdir(path):
            if not bool_sample or f in sample:
                for c in os.listdir(os.path.join(path, f)):
                    for i in os.listdir(os.path.join(path, f, c)):
                        GLCMs_1D.append(self.glcm(os.path.join(path, f, c, i)))
                        Labels.append(f)

        # dimentionality reduction
        PCA = []
        PCA = self.pca(GLCMs_1D, min(len(GLCMs_1D), len(GLCMs_1D[0])))

        # evaluation with cross validation
        self.cross_validation(PCA, Labels, n_folds, n_knn)


    def glcm(self, file_path):
        # Load image
        image = np.array(Image.open(file_path), dtype=np.uint8)
        image = rgb2gray(image)
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = np.array(image, dtype=np.uint8)

        # Create 1D GLCM of the image
        glcm = graycomatrix(image, distances=[10], angles=[0], levels=256)
        glcm_1D = glcm[:, :, 0, 0].flatten()

        return glcm_1D.tolist()


    def pca(self, GLCMs_1D, number_of_components):
        # Apply dimensionality reduction
        pca = PCA(n_components=number_of_components)
        return pca.fit_transform(GLCMs_1D)


    def cross_validation(self, data, labels, n_folds, n_knn):
        X = data
        y = np.array(labels)

        # Create knn classifier
        # p=1 for manhattan distance
        knn = KNeighborsClassifier(n_neighbors=n_knn, weights='distance', p=1)
        

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=10)
        k_folds = skf.split(X, y)

        n = 0
        score_sum = 0

        # Evaluation with cross validation
        for train_index, test_index in k_folds:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            knn.fit(X_train, y_train)
            score_sum += knn.score(X_test, y_test)
            n += 1

        # print recognition rate
        print("Recognition rate: {0:.00%}".format(score_sum/n))


    def export_glcm_pca_to_csv(self, bool_sample):
        base_path = Path(__file__)
        path = (base_path / "../Prepared-MMU-Iris-Database/").resolve()

        sample = ["005", "010", "013", "022", "027", "030", "032", "035", "038", "040"]
        GLCMs_1D = []
        Labels = []

        # Create 1D GLCM of all images
        for f in os.listdir(path):
            if not bool_sample or f in sample:
                for c in os.listdir(os.path.join(path, f)):
                    for i in os.listdir(os.path.join(path, f, c)):
                        GLCMs_1D.append(self.glcm(os.path.join(path, f, c, i)))
                        Labels.append(f)

        # export 1D GLCMs to csv files with different number of components
        self.create_glcm_pca_csv_files(GLCMs_1D, Labels, bool_sample)


    def create_glcm_pca_csv_files(self, GLCMs_1D, Labels, bool_sample):

        if bool_sample:
            number_of_file = 10
        else:
            number_of_file = 9

        for i in range(number_of_file):
            PCA_components = []
            header = []

            if bool_sample:
                n_components=(i+1)*10
            else:
                n_components=(i+1)*50
            
            # Apply dimensionnality reduction
            PCA_components = self.pca(GLCMs_1D, n_components)
        
            for j in range(len(PCA_components[0])):
                header.append("C" + str(j))
        
            header.append("ID")
        
            PCA_components_labels = [[]]*len(PCA_components)
        
            # Add IDs to the PCA array
            for j in range(len(PCA_components)):
                PCA_components_labels[j] = np.append(PCA_components[j], Labels[j]).tolist()

            file_name = "glcm"

            if bool_sample:
                file_name += "_sample"

            # Create csv file
            with open(file_name + str(i) + '.csv', 'w', encoding='UTF8', newline='') as f:
                # create the csv writer
                writer = csv.writer(f)
        
                writer.writerow(header) 
        
                for row in PCA_components_labels:
                    # write a row to the csv file
                    writer.writerow(row)     


    def create_glcm_features_csv_files(self, bool_sample):
        base_path = Path(__file__)
        path = (base_path / "../Prepared-MMU-Iris-Database/").resolve()

        sample = ["005", "010", "013", "022", "027", "030", "032", "035", "038", "040"]
        GLCM_features = [[]]
        index = 1

        GLCM_features[0] = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM", "ID"]

        # Create array of GLCM features for all images
        for f in os.listdir(path):
            if not bool_sample or f in sample:
                for c in os.listdir(os.path.join(path, f)):
                    for i in os.listdir(os.path.join(path, f, c)):
                        # Load image
                        image = np.array(Image.open(os.path.join(path, f, c, i)), dtype=np.uint8)
                        image = rgb2gray(image)
                        image = Image.fromarray((image * 255).astype(np.uint8))
                        image = np.array(image, dtype=np.uint8)

                        # Create GLCM
                        glcm = graycomatrix(image, distances=[10], angles=[0], levels=256)

                        GLCM_features.append([])

                        # Extract GLCM texture features
                        GLCM_features[index].append(graycoprops(glcm, 'contrast')[0][0])
                        GLCM_features[index].append(graycoprops(glcm, 'dissimilarity')[0][0])
                        GLCM_features[index].append(graycoprops(glcm, 'homogeneity')[0][0])
                        GLCM_features[index].append(graycoprops(glcm, 'energy')[0][0])
                        GLCM_features[index].append(graycoprops(glcm, 'correlation')[0][0])
                        GLCM_features[index].append(graycoprops(glcm, 'ASM')[0][0])
                        GLCM_features[index].append(f)

                        index += 1

        file_name = "glcm_features"

        if bool_sample:
            file_name += "_sample"

        # Create csv file
        with open(file_name + '.csv', 'w', encoding='UTF8', newline='') as f:
                # create the csv writer
                writer = csv.writer(f)
        
                for row in GLCM_features:
                    # write a row to the csv file
                    writer.writerow(row)       

        
