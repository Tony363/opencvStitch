import time
# import the necessary packages
import numpy as np
#import arrayfire as af
import imutils
import cv2
import faulthandler; faulthandler.enable()
class CODES:
    INFO = "[INFO]"
    ERROR = "[ERROR]"
    SAVED = "[SAVED]"


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

def Kseam_work_aspect(K,seam_work_aspect):
    K[0, 0] *= seam_work_aspect
    K[0, 2] *= seam_work_aspect
    K[1, 1] *= seam_work_aspect
    K[1, 2] *= seam_work_aspect
    return K

def Manual(
    left_image,
    right_image,
    cached=None,
    work_megapix=0.00,
    seam_megapix=0.01,
    ba_refine_mask='_____',
    finder=cv2.xfeatures2d_SURF.create(),
    matcher = cv2.detail.BestOf2NearestMatcher_create(False, 0.65),
    blender=cv2.detail.Blender_createDefault(cv2.detail.Blender_NO),
    compensator=cv2.detail.ExposureCompensator_createDefault(2),
    seam_finder=cv2.detail_GraphCutSeamFinder('COST_COLOR'),
    ):
    img_names = np.asarray([left_image,right_image])
    if cached is not None:
        composetime = timer(start_time=None)
        dst_sz,warper, cameras,corners,masks_warped = cached
        blender.prepare(dst_sz)
        for idx, name in enumerate(img_names):
            corner, image_warped = warper.warp(name, cameras[idx].K().astype(np.float32), cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            p, mask_warped = warper.warp(255 * np.ones((name.shape[0], name.shape[1]), np.uint8), cameras[idx].K().astype(np.float32), cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            mask_warped = cv2.bitwise_and(cv2.resize(cv2.dilate(masks_warped[idx], None), (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT), mask_warped)
            blender.feed(cv2.UMat(image_warped.astype(np.int16)), mask_warped, corners[idx])
        result, result_mask = blender.blend(None, None)
        dst = cv2.normalize(src=result, dst=None, alpha=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        timer(composetime)
        return False,dst
    estimation_time = timer(start_time=None)     
    """
    finder = cv2.xfeatures2d_SURF.create() 
    finder = cv2.ORB.create()
    finder = cv2.xfeatures2d_SIFT.create()
    finder = cv2.BRISK_create()
    finder = cv2.AKAZE_create()
    finder = cv2.FastFeatureDetector_create(),
    """
        
    work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (left_image.shape[0] * left_image.shape[1]))) # because both image dimensions should be the same
    seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (left_image.shape[0] * left_image.shape[1])))
    seam_work_aspect = seam_scale / work_scale

    full_img_sizes = np.asarray([(name.shape[1],name.shape[0]) for name in img_names])
    features = np.asarray([cv2.detail.computeImageFeatures2(finder,cv2.resize(src=name, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_LINEAR_EXACT)) for name in img_names])
    images = np.asarray([cv2.resize(src=name, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT) for name in img_names])
    
    # matcher = get_matcher(args)
    p = matcher.apply2(features)
    matcher.collectGarbage()

    indices = cv2.detail.leaveBiggestComponent(features, p, 0.3)
    estimator = cv2.detail_HomographyBasedEstimator()
    b, cameras = estimator.apply(features, p, None)
    
    # cameras = np.asarray([cam.R.astype(np.float32) for cam in cameras])
    for cam in cameras:cam.R = cam.R.astype(np.float32) # need to figure out how to turn back from np to cv::detail::CameraParams object
    
    # can explore more different adjuster matrix params
    adjuster = cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(1)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)
    
    focals = np.asarray([cam.focal for cam in cameras]) # might need np.sort()
    
    warped_image_scale = focals[len(focals) // 2] if len(focals) % 2 == 1 else (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
    
    rmats = np.asarray([np.copy(cam.R)for cam in cameras])
    rmats = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
    for idx, cam in enumerate(cameras):cam.R = rmats[idx] # need to figure out how to vectorize cameras object

    # warper = cv2.PyRotationWarper('spherical', warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    # masks = np.asarray([cv2.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8)) for i in range(img_names.shape[0])])
    # corners = np.asarray([warper.warp(images[idx], Kseam_work_aspect(cameras[idx].K().astype(np.float32),seam_work_aspect), cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)[0] for idx in range(img_names.shape[0])])
    # sizes = np.asarray([warper.warp(images[idx], Kseam_work_aspect(cameras[idx].K().astype(np.float32),seam_work_aspect), cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)[1].shape[1::-1] for idx in range(img_names.shape[0])])
    # images_warped = np.asarray([warper.warp(images[idx], Kseam_work_aspect(cameras[idx].K().astype(np.float32),seam_work_aspect), cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)[1]for idx in range(img_names.shape[0])])
    # masks_warped = np.asarray([warper.warp(masks[idx], Kseam_work_aspect(cameras[idx].K().astype(np.float32),seam_work_aspect), cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)[1].get() for idx in range(img_names.shape[0])])
    # images_warped_f = np.asarray([img.astype(np.float32) for img in images_warped])
    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(img_names.shape[0]):
        um = cv2.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)

    warper = cv2.PyRotationWarper('spherical', warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    for idx in range(img_names.shape[0]):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())

    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)

    # compensator = get_compensator(args)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped) # .tolist()
    # seam_finder = SEAM_FIND_CHOICES[args.seam]
    seam_finder.find(images_warped_f, corners, masks_warped)# .tolist()

    warped_image_scale *= 1/work_scale
    warper = cv2.PyRotationWarper('spherical', warped_image_scale)

    # """calculate corner and size of time step"""
    corners = []
    sizes = []
    for i in range(len(img_names)):
        cameras[i].focal *= 0.9999/work_scale
        cameras[i].ppx *= 1/work_scale
        cameras[i].ppy *= 1/work_scale
        sz = (full_img_sizes[i][0] * 1, full_img_sizes[i][1] * 1)
        K = cameras[i].K().astype(np.float32)
        roi = warper.warpRoi(sz, K, cameras[i].R)
        corners.append(roi[0:2])
        sizes.append(roi[2:4])

    dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
    blender.prepare(dst_sz)

    """Panorama construction step"""
    # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
    for idx, name in enumerate(img_names):
        corner, image_warped = warper.warp(name, cameras[idx].K().astype(np.float32), cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        p, mask_warped = warper.warp(255 * np.ones((name.shape[0], name.shape[1]), np.uint8), cameras[idx].K().astype(np.float32), cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        mask_warped = cv2.bitwise_and(cv2.resize(cv2.dilate(masks_warped[idx], None), (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT), mask_warped)
        blender.feed(cv2.UMat(image_warped.astype(np.int16)), mask_warped, corners[idx])
    
    result, result_mask = blender.blend(None, None)
    dst = cv2.normalize(src=result, dst=None, alpha=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#    if dst.shape[0] in range(2000,2300) and dst.shape[1] in range(4700,5000):
#    cv2.imshow("Stitching result:{} --{}".format(dst.shape,work_megapix),imutils.resize(dst,width=1080))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    cached = (dst_sz,warper,cameras,corners,masks_warped)
    return False,dst,cached
    print("--work_megapix {}, worked | shape {} | estimation time {}".format(work_megapix,dst.shape,timer(estimation_time)))

    return True,dst,None

