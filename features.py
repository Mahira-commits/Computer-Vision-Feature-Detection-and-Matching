import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        # # Convert the image to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # # Detect keypoints using Harris corner detector
        # corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        
        # # Normalize the corner response
        # corners_norm = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1, None)
        
        # # Threshold to get strong corners
        # threshold = 0.01 * corners_norm.max()
        # corner_image = image.copy()
        # keypoints = []
        # for i in range(corners_norm.shape[0]):
        #     for j in range(corners_norm.shape[1]):
        #         if corners_norm[i,j] > threshold:
        #             # Create a cv2.KeyPoint object
        #             kp = cv2.KeyPoint(j, i, 10, _class_id=-1, _response=corners_norm[i,j])
        #             keypoints.append(kp)
        #             # Draw a circle at each keypoint
        #             cv2.circle(corner_image, (j, i), 5, (0,255,0), 2)
        
        # # Display the keypoints
        # cv2.imshow('Corners', corner_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # return keypoints
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # # TODO 1: Compute the harris corner strength for 'srcImage' at
        # # each pixel and store in 'harrisImage'.  See the project page
        # # for direction on how to do this. Also compute an orientation
        # # for each pixel and store it in 'orientationImage.'
        # # TODO-BLOCK-BEGIN
        
        # sobelx = cv2.Sobel(srcImage, cv2.CV_64F, 1, 0, ksize=3)
        # sobely = cv2.Sobel(srcImage, cv2.CV_64F, 0, 1, ksize=3)

        # # Compute components of the Harris matrix
        # Ixx = sobelx ** 2
        # Iyy = sobely ** 2
        # Ixy = sobelx * sobely

        # # Compute sums of squared differences
        # Sxx = cv2.GaussianBlur(Ixx, (3, 3), 0)
        # Syy = cv2.GaussianBlur(Iyy, (3, 3), 0)
        # Sxy = cv2.GaussianBlur(Ixy, (3, 3), 0)

        # # Compute determinant and trace of the Harris matrix
        # det = (Sxx * Syy) - (Sxy ** 2)
        # trace = Sxx + Syy

        # # Compute Harris response
        # harrisImage = det - 0.04 * (trace ** 2)

        # # Compute orientation of gradients
        # orientationImage = np.arctan2(sobely, sobelx) * (180 / np.pi)

        # # Compute x and y gradients of the image
        # Ix = ndimage.sobel(srcImage, axis=1, mode='constant')
        # Iy = ndimage.sobel(srcImage, axis=0, mode='constant')

        # # Compute products of derivatives at every pixel
        # Ixx = ndimage.gaussian_filter(Ix**2, sigma=1)
        # Iyy = ndimage.gaussian_filter(Iy**2, sigma=1)
        # Ixy = ndimage.gaussian_filter(Ix*Iy, sigma=1)

        # # Compute the Harris response for each pixel
        # detM = Ixx * Iyy - Ixy**2
        # traceM = Ixx + Iyy
        # k = 0.1
        # harrisImage = detM - k * traceM**2

        # # Compute orientation of the gradient
        # orientationImage = np.degrees(np.arctan2(Iy, Ix))

        # Compute the gradients using 3x3 Sobel operator and Use reflection for gradient values outside of the image range.
        dx = cv2.Sobel(srcImage, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        dy = cv2.Sobel(srcImage, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

        # Compute products of derivatives
        Ix2 = dx ** 2
        Iy2 = dy ** 2
        Ixy = dx * dy

        # Apply Gaussian filter to the products of derivatives - with 5x5 Gaussian mask with 0.5 sigma
        sigma = 0.5
        Ix2 = cv2.GaussianBlur(Ix2, (5, 5), sigma, borderType=cv2.BORDER_REFLECT)
        Iy2 = cv2.GaussianBlur(Iy2, (5, 5), sigma, borderType=cv2.BORDER_REFLECT)
        Ixy = cv2.GaussianBlur(Ixy, (5, 5), sigma, borderType=cv2.BORDER_REFLECT)

        # Compute the Harris response for each pixel - with k = 0.1
        k = 0.1
        harrisImage = (Ix2 * Iy2 - Ixy ** 2) - k * (Ix2 + Iy2) ** 2

        # Compute the orientation of the gradient
        orientationImage = np.arctan2(dy, dx) * (180 / np.pi)

        # raise Exception("TODO 1: in features.py not implemented")

        # # TODO-BLOCK-END
       
        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, dtype=np.bool)

        # Iterate over each pixel in the harrisImage
        for i in range(harrisImage.shape[0]):
            for j in range(harrisImage.shape[1]):
                # Define the neighborhood
                min_row = max(0, i - 3)
                max_row = min(harrisImage.shape[0], i + 4)
                min_col = max(0, j - 3)
                max_col = min(harrisImage.shape[1], j + 4)

                # Extract the local window
                neighborhood = harrisImage[min_row:max_row, min_col:max_col]

                # Check if the current pixel value is the maximum in its neighborhood
                if harrisImage[i, j] == np.max(neighborhood):
                    destImage[i, j] = True
                else:
                    destImage[i, j] = False

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # # TODO 3: Fill in feature f with location and orientation
                # # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # # f.angle to the orientation in degrees and f.response to
                # # the Harris score
                # # TODO-BLOCK-BEGIN
                # raise Exception("TODO 3: in features.py not implemented")
                # # TODO-BLOCK-END

                # features.append(f)
                f.size = 10
                f.pt = (x, y)
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]

                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        # import pdb; pdb.set_trace()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        # # Convert the image to grayscale
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # # Initialize MOPS descriptor extractor
        # mops = cv2.xfeatures2d.MOPS_create()
        
        # # Compute descriptors
        # keypoints, descriptors = mops.compute(gray_image, keypoints)
        
        # return descriptors
        raise NotImplementedError()


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):
            x, y = int(f.pt[0]), int(f.pt[1])

            # # TODO 4: The simple descriptor is a 5x5 window of intensities
            # # sampled centered on the feature point. Store the descriptor
            # # as a row-major vector. Treat pixels outside the image as zero.
            # # TODO-BLOCK-BEGIN
            # raise Exception("TODO 4: in features.py not implemented")
            # # TODO-BLOCK-END
            # Initialize a 5x5 window around the keypoint
            window = np.zeros((5, 5))

            # Fill the window with intensities from the grayscale image
            for row in range(-2, 3):
                for col in range(-2, 3):
                    # Check if the pixel is within the image boundaries
                    if 0 <= y + row < grayImage.shape[0] and 0 <= x + col < grayImage.shape[1]:
                        window[row + 2][col + 2] = grayImage[y + row, x + col]

            # Store the window intensities as a row-major vector
            desc[i] = window.flatten()

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            # # TODO-BLOCK-BEGIN
            # raise Exception("TODO 5: in features.py not implemented")

            x, y = f.pt  # Get the keypoint position
            orientation = math.radians(f.angle)  # Convert orientation to radians

            # Compute translation to center the feature
            T1 = transformations.get_trans_mx(np.array([-x, -y, 0]))

            # Compute rotation to align the feature orientation to the right
            R = transformations.get_rot_mx(0, 0, -orientation)

            # Compute scaling to resize the window from 40x40 to 8x8
            S = transformations.get_scale_mx(0.2, 0.2, 1)  # Scaling factor is 5 because 40/8 = 5

            # Compute translation to move the top-left corner of the window to the descriptor origin
            T2 = transformations.get_trans_mx(np.array([4, 4, 0]))

            MT1 = np.dot(T2, np.dot(S, np.dot(R, T1)))
            transMx = MT1[:2, (0, 1, 3)]

            # Combine the transformations
            # a = T1 @ S
            # b = 
            # transMx = T2 @ S @ R @ T1  # Note that matrix multiplication is associative
            # transMx = T1 @ R @ S @ T2
            # # TODO-BLOCK-END
            # Compute the transformation matrix based on the keypoint location and orientation
            # new_angle_of_rotation = 90 - f.angle
            # local_rotation_mat = transformations.get_rot_mx(0, 0, np.radians(new_angle_of_rotation))
            # local_scale_matrix = transformations.get_scale_mx(0.2, 0.2, 1)
                
            # local_rotation_mat_final = local_rotation_mat[:2, :2]
            # local_scale_matrix_final = local_scale_matrix[:2, :2]
            # transMx = np.dot(local_rotation_mat_final, local_scale_matrix_final)
            # transMx = np.hstack((transMx, np.zeros((transMx.shape[0], 1), dtype=transMx.dtype)))
            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # # TODO 6: Normalize the descriptor to have zero mean and unit 
            # # variance. If the variance is negligibly small (which we 
            # # define as less than 1e-10) then set the descriptor
            # # vector to zero. Lastly, write the vector to desc.
            # # TODO-BLOCK-BEGIN
            # raise Exception("TODO 6: in features.py not implemented")
            # # TODO-BLOCK-END
            # Normalize the descriptor to have zero mean and unit variance
            
            desc_mean = np.mean(destImage)
            desc_var = np.var(destImage)
            if desc_var < 1e-10:
                destImage = np.zeros((1, windowSize*windowSize))
            else:
                destImage = (destImage - desc_mean) / (desc_var**0.5)
                destImage = destImage.reshape((1, windowSize*windowSize))
            
            desc[i] = destImage
        
        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        # # Convert the image to grayscale
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # # Gaussian blur the grayscale image
        # gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0.5)
        
        # # Window size for the feature descriptor
        # window_size = 8
        
        # # Initialize the descriptor array
        # desc = np.zeros((len(keypoints), window_size * window_size))
        
        # for i, f in enumerate(keypoints):
        #     # Compute the transformation matrix based on the keypoint location and orientation
        #     transMx = np.zeros((2, 3))
        #     transMx[0, 0] = np.cos(np.radians(f.angle))
        #     transMx[0, 1] = -np.sin(np.radians(f.angle))
        #     transMx[1, 0] = np.sin(np.radians(f.angle))
        #     transMx[1, 1] = np.cos(np.radians(f.angle))
        #     transMx[0, 2] = -4 * f.pt[0] * np.cos(np.radians(f.angle)) - 4 * f.pt[1] * np.sin(np.radians(f.angle)) + window_size / 2
        #     transMx[1, 2] = 4 * f.pt[0] * np.sin(np.radians(f.angle)) - 4 * f.pt[1] * np.cos(np.radians(f.angle)) + window_size / 2
            
        #     # Apply the transformation to the grayscale image
        #     destImage = cv2.warpAffine(gray_image, transMx, (window_size, window_size), flags=cv2.INTER_LINEAR)
            
        #     # Normalize the descriptor to have zero mean and unit variance
        #     desc_mean = np.mean(destImage)
        #     desc_std = np.std(destImage)
        #     if desc_std < 1e-10:
        #         desc[i] = 0
        #     else:
        #         desc[i] = (destImage - desc_mean) / desc_std
        
        # return desc
        raise NotImplementedError()


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        # # Create a brute-force matcher
        # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # # Match descriptors
        # matches = bf.match(desc1, desc2)
        
        # # Convert matches to cv2.DMatch objects
        # dmatches = [cv2.DMatch(i, i, 0) for i in range(len(matches))]
        
        # return dmatches
        raise NotImplementedError()

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # # TODO 7: Perform simple feature matching.  This uses the SSD
        # # distance between two feature vectors, and matches a feature in
        # # the first image with the closest feature in the second image.
        # # Note: multiple features from the first image may match the same
        # # feature in the second image.
        # # TODO-BLOCK-BEGIN
        # raise Exception("TODO 7: in features.py not implemented")
        ssd = spatial.distance.cdist(desc1, desc2, metric = 'euclidean')
        for i in range(ssd.shape[0]):
            min_index = np.argmin(ssd[i])
            match = cv2.DMatch(i, min_index, ssd[i][min_index])
            matches.append(match)
        # TODO-BLOCK-END
        # # Compute SSD distances
        # for i in range(len(desc1)):
        #     min_distance = float('inf')
        #     min_index = -1
        #     for j in range(len(desc2)):
        #         distance = np.sum((desc1[i] - desc2[j]) ** 2)
        #         if distance < min_distance:
        #             min_distance = distance
        #             min_index = j
        #     # Create a cv2.DMatch object
        #     match = cv2.DMatch(i, min_index, min_distance)
        #     matches.append(match)

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # # TODO 8: Perform ratio feature matching.
        # # This uses the ratio of the SSD distance of the two best matches
        # # and matches a feature in the first image with the closest feature in the
        # # second image.
        # # Note: multiple features from the first image may match the same
        # # feature in the second image.
        # # You don't need to threshold matches in this function
        # # TODO-BLOCK-BEGIN
        # raise Exception("TODO 8: in features.py not implemented")
        ssd = spatial.distance.cdist(desc1, desc2, metric = 'euclidean')

        for i in range(ssd.shape[0]):
            # distances = ssd[i]
            # distances.sort()
            # ratio = distances[0] / distances[1]
            # match = cv2.DMatch(i, np.argmin(ssd[i]), ratio)
            # matches.append(match)
            min_dist_idx, min_dist2_idx = np.argsort(ssd[i])[:2]
            match_val = cv2.DMatch(i, min_dist_idx, ssd[i, min_dist_idx] / ssd[i, min_dist2_idx])
            matches.append(match_val)
        # # TODO-BLOCK-END
        # # Compute SSD distances and perform ratio test
        # for i in range(len(desc1)):
        #     distances = []
        #     for j in range(len(desc2)):
        #         distance = np.sum((desc1[i] - desc2[j]) ** 2)
        #         distances.append((distance, j))
        #     distances.sort()
        #     if len(distances) > 1:
        #         ratio = distances[0][0] / distances[1][0]
        #         # Apply ratio test
        #         if ratio < 0.75:
        #             match = cv2.DMatch(i, distances[0][1], ratio)
        #             matches.append(match)
        #     elif len(distances) == 1:
        #         match = cv2.DMatch(i, distances[0][1], 0)
        #         matches.append(match)


        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))
