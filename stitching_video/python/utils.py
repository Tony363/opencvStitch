import time
# import the necessary packages
import numpy as np
#import arrayfire as af
#from numba import jit
import imutils
import cv2
from stitching_object import Stitcher
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

def get_minimum_total_frame(left_capture, right_capture):
    left_total_frame = int(left_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    right_total_frame = int(right_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frame = min(left_total_frame, right_total_frame)
    print(CODES.INFO, "Total frames set to {}".format(total_frame))
    return total_frame

def status_check(status):
    if status != cv2.Stitcher_OK:
        print("Can't stitch images, error code = {}".format(status))
        return False
    return True
    
def estimateTransform(left_image,right_image,work_megapix):
    start_time = timer(start_time=None)
    compose = Stitcher(left_image,right_image,work_megapix=work_megapix)
    compose.match_features()
    compose.refineMask()
    compose.fine_adjustments()
    compose.corners_masks_sizes()
    compose.images_warpedf()
    compose.get_warper()
    compose.calculate_corners()
    pano = compose.composePanorama()
    print("--work_megapix {}, worked | shape {} | estimation time {}".format(work_megapix,pano.shape,timer(start_time)))
    return False,pano,compose


def Kseam_work_aspect(K,seam_work_aspect):
    K[0, 0] *= seam_work_aspect
    K[0, 2] *= seam_work_aspect
    K[1, 1] *= seam_work_aspect
    K[1, 2] *= seam_work_aspect
    return K

def composePanorama(img_names,blender,compensator,cached):
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
    return False,dst

def calculate_corners(img_names,cameras,full_img_sizes,work_scale,warper):
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
    return corners,sizes


def corners_masks_sizes(warped_image_scale,seam_work_aspect,img_names,images,cameras):
    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    warper = cv2.PyRotationWarper('spherical', warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    for idx in range(img_names.shape[0]):
        um = cv2.UMat(255 * np.ones((images[idx].shape[0], images[idx].shape[1]), np.uint8))
        masks.append(um)
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
    return corners,masks_warped,images_warped,sizes,masks

def refine_mask(ba_refine_mask):
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
    return refine_mask

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
        status,pano = composePanorama(img_names,blender,compensator,cached)
        timer(composetime)
        return status,pano
       
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
    adjuster.setRefinementMask(refine_mask(ba_refine_mask))
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
    # images_warped = np.asarray([w@jit(nopython=False)arper.warp(images[idx], Kseam_work_aspect(cameras[idx].K().astype(np.float32),seam_work_aspect), cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)[1]for idx in range(img_names.shape[0])])
    # masks_warped = np.asarray([warper.warp(masks[idx], Kseam_work_aspect(cameras[idx].K().astype(np.float32),seam_work_aspect), cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)[1].get() for idx in range(img_names.shape[0])])
    # images_warped_f = np.asarray([img.astype(np.float32) for img in images_warped])

    corners,masks_warped,images_warped,sizes,masks = corners_masks_sizes(warped_image_scale,seam_work_aspect,img_names,images,cameras)

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

    """calculate corner and size of time step"""
    corners,sizes = calculate_corners(img_names,cameras,full_img_sizes,work_scale,warper)
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

    if dst.shape[0] in range(2000,2300) and dst.shape[1] in range(4700,5000):
        cv2.imshow("Stitching result:{} --{}".format(dst.shape,work_megapix),imutils.resize(dst,width=1080))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cached = (dst_sz,warper,cameras,corners,masks_warped)
        return False,dst,cached
    print("--work_megapix {}, worked | shape {} | estimation time {}".format(work_megapix,dst.shape,timer(estimation_time)))

    return True,dst,None

