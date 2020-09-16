import time
# import the necessary packages
import numpy as np
import arrayfire as af
import imutils
import cv2



class CODES:
    INFO = "[INFO]"
    ERROR = "[ERROR]"


# Return time elapsed
def timer(start_time=None,msg = None, display=False):
    # Initialize timer
    if start_time is None:
        return time.time()
    if msg is not None and display:
        print(CODES.INFO, msg , ": {:.3f} s".format(time.time() - start_time))
    return time.time() - start_time

class Manual:
    def __init__(self):
        # determine if we are using OpenCV v3.X and initialize the
        # cached homography matrix
        self.isv3 = imutils.is_cv3()
        self.cachedH = None
        self.to_estimate = False
        self.FLANN_INDEX_KDTREE = 2
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE,trees=10)
        self.search_params = dict(checks=100)


    def stitch(self, images, ratio=0.90, reprojThresh=10.0):
        # unpack the images
        (imageB, imageA) = images
        # if the cached homography matrix is None, then we need to
        # apply keypoint matching to construct it
        print("estimate status",self.to_estimate,'\n')
        if self.cachedH is None or self.to_estimate is True:
            # detect keypoints and extract
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB,
                featuresA, featuresB, ratio, reprojThresh)
            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if M is None:
                return None
            # cache the homography matrix
            self.cachedH = M[1]
            # apply a perspective transform to stitch the images together
            # using the cached homography matrix
            result = cv2.warpPerspective(imageA, self.cachedH,
                (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
            if result is not None:
                return False,result
            
        else:
            # apply a perspective transform to stitch the images together
            # using the cached homography matrix
            result = cv2.warpPerspective(imageA, self.cachedH,
                (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
            if result is not None:
                return False,result
        # return the stitched image
        return False,None

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # image to numpy
        # img = np.asarray(image)
        # extract keypoints
        # img = af.Array(img.ctypes.data,img.shape,img.dtype.char)
        # print(af.display(img))

       # fast_features = af.vision.fast(img)
        # print(fast_features)
        fast = cv2.FastFeatureDetector_create()
        kps = fast.detect(image,None)
        # write keypoints
        # img2 = cv2.drawKeypoints(image,kps,None,color=(255,0,0))
        # cv2.imwrite('inputs/fast_true.png',img2)

        # Disable nonmaxSupression
        fast.setNonmaxSuppression(0)
        kp = fast.detect(image,None)
        # img3 = cv2.drawKeypoints(image,kp,None,color=(255,0,0))
        # cv2.imwrite('inputs/fast_false.png',img3)

        # extract features
        br = cv2.BRISK_create()
        kp,features = br.compute(image,kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kp])
        # return a tuple of keypoints and features
        return (kps, np.float32(features))

    def matchKeypoints(self, kpsA, kpsB,featuresA,featuresB,ratio,reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        # matcher = cv2.DescriptorMatcher_create("BruteForce")
        matcher = cv2.FlannBasedMatcher(self.index_params,self.search_params)
        rawMatches = matcher.knnMatch(featuresA, featuresB,k=2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        # otherwise, no homograpy could be computed
        return None