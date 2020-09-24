"""
Stitching sample (advanced)
===========================

Show how to use Stitcher API from python.
"""

# Python 2/3 compatibility
from __future__ import print_function

import argparse
from collections import OrderedDict

import cv2 
import numpy as np
from numba import jit
import imutils
from utils import timer
import faulthandler; faulthandler.enable()

EXPOS_COMP_CHOICES = OrderedDict()
EXPOS_COMP_CHOICES['gain_blocks'] = cv2.detail.ExposureCompensator_GAIN_BLOCKS
EXPOS_COMP_CHOICES['gain'] = cv2.detail.ExposureCompensator_GAIN
EXPOS_COMP_CHOICES['channel'] = cv2.detail.ExposureCompensator_CHANNELS
EXPOS_COMP_CHOICES['channel_blocks'] = cv2.detail.ExposureCompensator_CHANNELS_BLOCKS
EXPOS_COMP_CHOICES['no'] = cv2.detail.ExposureCompensator_NO

BA_COST_CHOICES = OrderedDict()
BA_COST_CHOICES['ray'] = cv2.detail_BundleAdjusterRay
BA_COST_CHOICES['reproj'] = cv2.detail_BundleAdjusterReproj
BA_COST_CHOICES['affine'] = cv2.detail_BundleAdjusterAffinePartial
BA_COST_CHOICES['no'] = cv2.detail_NoBundleAdjuster

FEATURES_FIND_CHOICES = OrderedDict()
try:
    FEATURES_FIND_CHOICES['surf'] = cv2.xfeatures2d_SURF.create
except AttributeError:
    print("SURF not available")
# if SURF not available, ORB is default
FEATURES_FIND_CHOICES['orb'] = cv2.ORB.create
try:
    FEATURES_FIND_CHOICES['sift'] = cv2.xfeatures2d_SIFT.create
except AttributeError:
    print("SIFT not available")
try:
    FEATURES_FIND_CHOICES['brisk'] = cv2.BRISK_create
except AttributeError:
    print("BRISK not available")
try:
    FEATURES_FIND_CHOICES['akaze'] = cv2.AKAZE_create
except AttributeError:
    print("AKAZE not available")

SEAM_FIND_CHOICES = OrderedDict()
SEAM_FIND_CHOICES['gc_color'] = cv2.detail_GraphCutSeamFinder('COST_COLOR')
SEAM_FIND_CHOICES['gc_colorgrad'] = cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
SEAM_FIND_CHOICES['dp_color'] = cv2.detail_DpSeamFinder('COLOR')
SEAM_FIND_CHOICES['dp_colorgrad'] = cv2.detail_DpSeamFinder('COLOR_GRAD')
SEAM_FIND_CHOICES['voronoi'] = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
SEAM_FIND_CHOICES['no'] = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO)

ESTIMATOR_CHOICES = OrderedDict()
ESTIMATOR_CHOICES['homography'] = cv2.detail_HomographyBasedEstimator
ESTIMATOR_CHOICES['affine'] = cv2.detail_AffineBasedEstimator

WARP_CHOICES = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)

WAVE_CORRECT_CHOICES = ('horiz', 'no', 'vert',)

BLEND_CHOICES = ('multiband', 'feather', 'no',)

def arguments():
    parser = argparse.ArgumentParser(
        prog="stitching_detailed.py", description="Rotation model images stitcher"
    )
    parser.add_argument(
        '--img_names', nargs='+',
        help="Files to stitch", type=str
    )
    parser.add_argument(
        '--try_cuda',
        action='store',
        default=False,
        help="Try to use CUDA. The default value is no. All default values are for CPU mode.",
        type=bool, dest='try_cuda'
    )

    parser.add_argument(
        '--features', action='store', default=list(FEATURES_FIND_CHOICES.keys())[0],
        help="Type of features used for images matching. The default is '%s'." % FEATURES_FIND_CHOICES.keys(),
        choices=FEATURES_FIND_CHOICES.keys(),
        type=str, dest='features'
    )
    parser.add_argument(
        '--matcher', action='store', default='homography',
        help="Matcher used for pairwise image matching.",
        choices=('homography', 'affine'),
        type=str, dest='matcher'
    )
    parser.add_argument(
        '--match_conf', action='store',
        help="Confidence for feature matching step. The default is 0.3 for ORB and 0.65 for other feature types.",
        type=float, dest='match_conf'
    )
    parser.add_argument(
        '--ba_refine_mask', action='store', default='xxxxx',
        help="Set refinement mask for bundle adjustment. It looks like 'x_xxx', "
            "where 'x' means refine respective parameter and '_' means don't refine, "
            "and has the following format:<fx><skew><ppx><aspect><ppy>. "
            "The default mask is 'xxxxx'. "
            "If bundle adjustment doesn't support estimation of selected parameter then "
            "the respective flag is ignored.",
        type=str, dest='ba_refine_mask'
    )
    parser.add_argument(
        '--seam', action='store', default=list(SEAM_FIND_CHOICES.keys())[0],
        help="Seam estimation method. The default is '%s'." % list(SEAM_FIND_CHOICES.keys())[0],
        choices=SEAM_FIND_CHOICES.keys(),
        type=str, dest='seam'
    )
    parser.add_argument(
        '--expos_comp', action='store', default=list(EXPOS_COMP_CHOICES.keys())[0],
        help="Exposure compensation method. The default is '%s'." % list(EXPOS_COMP_CHOICES.keys())[0],
        choices=EXPOS_COMP_CHOICES.keys(),
        type=str, dest='expos_comp'
    )
    parser.add_argument(
        '--expos_comp_nr_feeds', action='store', default=1,
        help="Number of exposure compensation feed.",
        type=np.int32, dest='expos_comp_nr_feeds'
    )
    parser.add_argument(
        '--expos_comp_nr_filtering', action='store', default=2,
        help="Number of filtering iterations of the exposure compensation gains.",
        type=float, dest='expos_comp_nr_filtering'
    )
    parser.add_argument(
        '--expos_comp_block_size', action='store', default=32,
        help="BLock size in pixels used by the exposure compensator. The default is 32.",
        type=np.int32, dest='expos_comp_block_size'
    )
    return parser,parser.parse_args()

def Kseam_work_aspect(K,seam_work_aspect):
    K[0, 0] *= seam_work_aspect
    K[0, 2] *= seam_work_aspect
    K[1, 1] *= seam_work_aspect
    K[1, 2] *= seam_work_aspect
    return K

def cam_focal_ppx_ppy(cameras,i,work_scale):
    cameras[i].focal *= 1/work_scale
    cameras[i].ppx *= 1/work_scale
    cameras[i].ppy *= 1/work_scale


def get_matcher(args):
    try_cuda = args.try_cuda
    matcher_type = args.matcher
    if args.match_conf is None:
        if args.features == 'orb':
            match_conf = 0.3
        else:
            match_conf = 0.65
    else:
        match_conf = args.match_conf
    matcher = cv2.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
    return matcher


def get_compensator(args):
    expos_comp_type = EXPOS_COMP_CHOICES[args.expos_comp]
    expos_comp_nr_feeds = args.expos_comp_nr_feeds
    expos_comp_block_size = args.expos_comp_block_size
    # expos_comp_nr_filtering = args.expos_comp_nr_filtering
    if expos_comp_type == cv2.detail.ExposureCompensator_CHANNELS:
        compensator = cv2.detail_ChannelsCompensator(expos_comp_nr_feeds)
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    elif expos_comp_type == cv2.detail.ExposureCompensator_CHANNELS_BLOCKS:
        compensator = cv2.detail_BlocksChannelsCompensator(
            expos_comp_block_size, expos_comp_block_size,
            expos_comp_nr_feeds
        )
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    else:
        compensator = cv2.detail.ExposureCompensator_createDefault(expos_comp_type)
    return compensator

# @jit(nopython=False)
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
    return dst

# @jit(nopython=False)
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

# @jit(nopython=False)
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

# @jit(nopython=True)
def images_warpedf(images_warped):
    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)
    return images_warped_f

def match_features(matcher,features):
    # matcher = get_matcher(args)
    p = matcher.apply2(features)
    matcher.collectGarbage()

    indices = cv2.detail.leaveBiggestComponent(features, p, 0.3)
    estimator = cv2.detail_HomographyBasedEstimator()
    b, cameras = estimator.apply(features, p, None)

    # cameras = np.asarray([cam.R.astype(np.float32) for cam in cameras])
    for cam in cameras:cam.R = cam.R.astype(np.float32) # need to figure out how to turn back from np to cv::detail::CameraParams object
    return p,cameras

def fine_adjustments(features,ba_refine_mask,p,cameras):
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
    return warped_image_scale,cameras  

def get_warper(compensator,seam_finder,corners,images_warped,masks_warped,images_warped_f,warped_image_scale,work_scale):
    # compensator = get_compensator(args)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped) # .tolist()
    # seam_finder = SEAM_FIND_CHOICES[args.seam]
    seam_finder.find(images_warped_f, corners, masks_warped)# .tolist()

    warped_image_scale *= 1/work_scale
    warper = cv2.PyRotationWarper('spherical', warped_image_scale)
    return warper  
class Manual:
    def __init__(self,left_image,right_image,cached):
        
        """initial params"""
        self.img_names = np.asarray([left_image,right_image])
        self.cached = None
        self.work_megapix = 0.00
        self.seam_megapix = 0.01
        self.ba_refine_mask = '_____'
        self.finder = cv2.xfeatures2d_SURF.create()
        self.matcher = cv2.detail.BestOf2NearestMatcher_create(False, 0.65)
        self.blender=cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
        self.compensator=cv2.detail.ExposureCompensator_createDefault(2)
        self.seam_finder=cv2.detail_GraphCutSeamFinder('COST_COLOR')

        """core transform matrix"""
        self.dst_sz = None
        self.warper = None
        self.p = None
        self.cameras = None
        self.corners = None
        self.masks_warped = None

        """aspect ratios"""
        self.work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (left_image.shape[0] * left_image.shape[1]))) # because both image dimensions should be the same
        self.seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (left_image.shape[0] * left_image.shape[1])))
        self.work_aspect = self.work_scale / self.seam_scale
        self.warped_image_scale = None

        """features and keypoints"""
        self.full_img_sizes = np.asarray([(name.shape[1],name.shape[0]) for name in img_names])
        self.features = np.asarray([cv2.detail.computeImageFeatures2(finder,cv2.resize(src=name, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_LINEAR_EXACT)) for name in img_names])
        self.images = np.asarray([cv2.resize(src=name, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT) for name in img_names])
        self.refine_mask = None

        """constructor transform"""
        self.corners = None
        self.masks_warped = None 
        self.images_warped = None
        self.sizes = None
        self.masks = None
        self.images_warped_f = None

    def match_features(self):
        # matcher = get_matcher(args)
        p = self.matcher.apply2(self.features)
        self.matcher.collectGarbage()

        indices = cv2.detail.leaveBiggestComponent(self.features, p, 0.3)
        estimator = cv2.detail_HomographyBasedEstimator()
        b, cameras = estimator.apply(self.features, p, None)
        
        # cameras = np.asarray([cam.R.astype(np.float32) for cam in cameras])
        for cam in cameras:
            cam.R = cam.R.astype(np.float32) # need to figure out how to turn back from np to cv::detail::CameraParams object
        self.cameras = cameras
        self.p = p
        return p,cameras

    def refine_mask(self):
        refine_mask = np.zeros((3, 3), np.uint8)
        if self.ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if self.ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if self.ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if self.ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if self.ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        self.refine_mask = refine_mask
        return refine_mask

    def fine_adjustments(self):
        # can explore more different adjuster matrix params
        adjuster = cv2.detail_BundleAdjusterRay()
        adjuster.setConfThresh(1)
        adjuster.setRefinementMask(self.refine_mask(self.ba_refine_mask))
        b, cameras = adjuster.apply(self.refine_mask)
        focals = np.asarray([cam.focal for cam in self.cameras]) # might need np.sort()
        warped_image_scale = focals[len(focals) // 2] if len(focals) % 2 == 1 else (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
        rmats = np.asarray([np.copy(cam.R)for cam in self.cameras])
        rmats = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
        for idx, cam in enumerate(self.cameras):
            cam.R = rmats[idx] # need to figure out how to vectorize cameras object
        self.cameras = cameras
        self.warped_image_scale = warped_image_scale
        return warped_image_scale,cameras  
    
    def corners_masks_sizes(self):
        corners = []
        masks_warped = []
        images_warped = []
        sizes = []
        masks = []
        warper = cv2.PyRotationWarper('spherical', self.warped_image_scale * self.seam_work_aspect)  # warper could be nullptr?
        for idx in range(self.img_names.shape[0]):
            um = cv2.UMat(255 * np.ones((self.images[idx].shape[0], self.images[idx].shape[1]), np.uint8))
            masks.append(um)
            K = self.cameras[idx].K().astype(np.float32)
            K[0, 0] *= self.seam_work_aspect
            K[0, 2] *= self.seam_work_aspect
            K[1, 1] *= self.seam_work_aspect
            K[1, 2] *= self.seam_work_aspect
            corner, image_wp = warper.warp(self.images[idx], K, self.cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            corners.append(corner)
            sizes.append((image_wp.shape[1], image_wp.shape[0]))
            images_warped.append(image_wp)
            p, mask_wp = warper.warp(masks[idx], K, self.cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            masks_warped.append(mask_wp.get())
        self.corners = corners
        self.masks_warped = masks_warped
        self.images_warped = images_warped
        self.sizes = sizes
        self.masks = masks
        return corners,masks_warped,images_warped,sizes,masks
    
    def images_warpedf(self):
        images_warped_f = []
        for img in images_warped:
            imgf = img.astype(np.float32)
            images_warped_f.append(imgf)
        return images_warped_f

    def composePanorama(self,cached):
        dst_sz,warper, cameras,corners,masks_warped = cached
        self.blender.prepare(dst_sz)
        for idx, name in enumerate(self.img_names):
            corner, image_warped = warper.warp(name, cameras[idx].K().astype(np.float32), cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            p, mask_warped = warper.warp(255 * np.ones((name.shape[0], name.shape[1]), np.uint8), cameras[idx].K().astype(np.float32), cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            self.compensator.apply(idx, corners[idx], image_warped, mask_warped)
            mask_warped = cv2.bitwise_and(cv2.resize(cv2.dilate(masks_warped[idx], None), (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT), mask_warped)
            self.blender.feed(cv2.UMat(image_warped.astype(np.int16)), mask_warped, corners[idx])
        result, result_mask = self.blender.blend(None, None)
        dst = cv2.normalize(src=result, dst=None, alpha=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return dst
    
    def calculate_corners(self,cameras,full_img_sizes,work_scale,warper):
        corners = []
        sizes = []
        for i in range(len(self.img_names)):
            cameras[i].focal *= 0.9999/work_scale
            cameras[i].ppx *= 1/work_scale
            cameras[i].ppy *= 1/work_scale
            sz = (full_img_sizes[i][0] * 1, full_img_sizes[i][1] * 1)
            K = cameras[i].K().astype(np.float32)
            roi = warper.warpRoi(sz, K, cameras[i].R)
            corners.append(roi[0:2])
            sizes.append(roi[2:4])
        return corners,sizes
    
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
    # if cached is not None:
    #     composetime = timer(start_time=None)
    #     status,pano = composePanorama(img_names,blender,compensator,cached)
    #     timer(composetime)
    #     return status,pano
       
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
    
    p,cameras = match_features(matcher,features)
    
    warped_image_scale,cameras = fine_adjustments(features,ba_refine_mask,p,cameras)

    # warper = cv2.PyRotationWarper('spherical', warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    # masks = np.asarray([cv2.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8)) for i in range(img_names.shape[0])])
    # corners = np.asarray([warper.warp(images[idx], Kseam_work_aspect(cameras[idx].K().astype(np.float32),seam_work_aspect), cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)[0] for idx in range(img_names.shape[0])])
    # sizes = np.asarray([warper.warp(images[idx], Kseam_work_aspect(cameras[idx].K().astype(np.float32),seam_work_aspect), cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)[1].shape[1::-1] for idx in range(img_names.shape[0])])
    # images_warped = np.asarray([warper.warp(images[idx], Kseam_work_aspect(cameras[idx].K().astype(np.float32),seam_work_aspect), cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)[1]for idx in range(img_names.shape[0])])
    # masks_warped = np.asarray([warper.warp(masks[idx], Kseam_work_aspect(cameras[idx].K().astype(np.float32),seam_work_aspect), cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)[1].get() for idx in range(img_names.shape[0])])
    # images_warped_f = np.asarray([img.astype(np.float32) for img in images_warped])

    corners,masks_warped,images_warped,sizes,masks = corners_masks_sizes(warped_image_scale,seam_work_aspect,img_names,images,cameras)
    images_warped_f = images_warpedf(images_warped)

    warper = get_warper(compensator,seam_finder,corners,images_warped,masks_warped,images_warped_f,warped_image_scale, work_scale)

    """calculate corner and size of time step"""
    corners,sizes = calculate_corners(img_names,cameras,full_img_sizes,work_scale,warper)
    dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
    blender.prepare(dst_sz)

    """Panorama construction step"""
    # # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
    dst = composePanorama(img_names,blender,compensator,(dst_sz,warper, cameras,corners,masks_warped))

    if dst.shape[0] in range(2000,2300) and dst.shape[1] in range(4700,5000):
        cv2.imshow("Stitching result:{} --{}".format(dst.shape,work_megapix),imutils.resize(dst,width=1080))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cached = (dst_sz,warper,cameras,corners,masks_warped)
        return False,dst,cached
    print("--work_megapix {}, worked | shape {} | estimation time {}".format(work_megapix,dst.shape,timer(estimation_time)))
    return True,dst,None

if __name__ == '__main__':
    parser,args = arguments()
    __doc__ += '\n' + parser.format_help()
    print(__doc__)
    left_image,right_image = cv2.imread(args.img_names[0]),cv2.imread(args.img_names[1])
    """
    May tennis :good stitches around 2100 - 4700 in shape
    June tennis :good stitches around 2200 - 4900 in shape;--work_megapix 0.38
    """
    # status,pano,cached = Manual(left_image,right_image,work_megapix=0.66)
    for i in np.arange(0.0,1.0,0.01):
        try:
            status,pano,cached = Manual(left_image,right_image,work_megapix=i)
        except Exception as e:
            print(e)
            continue
    cv2.destroyAllWindows()

