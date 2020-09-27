import cv2 
import numpy as np
# from numba import jit
import imutils
# from utils import timer

class Stitcher:
    def __init__(self,left_image,right_image,work_megapix):
        
        """initial params"""
        self.img_names = np.asarray([left_image,right_image])
        self.cached = None
        self.work_megapix = work_megapix
        self.seam_megapix = 0.01
        self.ba_refine_mask = '_____'
        self.finder = cv2.xfeatures2d_SURF.create()
        self.matcher = cv2.detail.BestOf2NearestMatcher_create(False, 0.65)
        self.blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
        self.compensator = cv2.detail.ExposureCompensator_createDefault(2)
        self.seam_finder = cv2.detail_GraphCutSeamFinder('COST_COLOR')

        """core transform matrix"""
        self.dst_sz = None
        self.warper = None
        self.p = None
        self.cameras = None
        self.corners = None
        self.masks_warped = None

        """aspect ratios"""
        self.work_scale = min(1.0, np.sqrt(self.work_megapix * 1e6 / (left_image.shape[0] * left_image.shape[1]))) # because both image dimensions should be the same
        self.seam_scale = min(1.0, np.sqrt(self.seam_megapix * 1e6 / (left_image.shape[0] * left_image.shape[1])))
        self.seam_work_aspect = self.work_scale / self.seam_scale
        self.warped_image_scale = None

        """features and keypoints"""
        self.full_img_sizes = np.asarray([(name.shape[1],name.shape[0]) for name in self.img_names])
        self.features = np.asarray([cv2.detail.computeImageFeatures2(self.finder,cv2.resize(src=name, dsize=None, fx=self.work_scale, fy=self.work_scale, interpolation=cv2.INTER_LINEAR_EXACT)) for name in self.img_names])
        self.images = np.asarray([cv2.resize(src=name, dsize=None, fx=self.seam_scale, fy=self.seam_scale, interpolation=cv2.INTER_LINEAR_EXACT) for name in self.img_names])

        """constructor transform"""
        self.refine_mask = None
        self.corners = None
        self.masks_warped = None 
        self.images_warped = None
        self.sizes = None
        self.masks = None
        self.images_warped_f = None

    def new_frame(self,left_image,right_image):
        self.img_names = np.asarray([left_image,right_image])

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

    def refineMask(self):
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
        adjuster.setRefinementMask(self.refine_mask)
        b, cameras = adjuster.apply(self.features,self.p,self.cameras)
        # focals = np.asarray([cam.focal for cam in self.cameras]) # might need np.sort()
        focals = []
        for cam in cameras:
            focals.append(cam.focal)
        focals.sort()
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
        for img in self.images_warped:
            imgf = img.astype(np.float32)
            images_warped_f.append(imgf)
        self.images_warped_f = images_warped_f
        return images_warped_f
    
    def get_warper(self):
        # compensator = get_compensator(args)
        self.compensator.feed(corners=self.corners, images=self.images_warped, masks=self.masks_warped) # .tolist()
        # seam_finder = SEAM_FIND_CHOICES[args.seam]
        self.seam_finder.find(self.images_warped_f, self.corners, self.masks_warped)# .tolist()

        self.warped_image_scale *= 1/self.work_scale
        self.warper = cv2.PyRotationWarper('spherical', self.warped_image_scale)
        return self.warper  

    def calculate_corners(self):
        corners = []
        sizes = []
        for i in range(len(self.img_names)):
            self.cameras[i].focal *= 0.9999/self.work_scale
            self.cameras[i].ppx *= 1/self.work_scale
            self.cameras[i].ppy *= 1/self.work_scale
            sz = (self.full_img_sizes[i][0] * 1, self.full_img_sizes[i][1] * 1)
            K = self.cameras[i].K().astype(np.float32)
            roi = self.warper.warpRoi(sz, K, self.cameras[i].R)
            corners.append(roi[0:2])
            sizes.append(roi[2:4])
        self.corners = corners
        self.sizes = sizes
        self.dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
        return corners,sizes

    def composePanorama(self):
        # dst_sz,warper, cameras,corners,masks_warped = cached
        print(self.corners,type(self.corners))
        self.blender.prepare(self.dst_sz)
        for idx, name in enumerate(self.img_names):
            corner, image_warped = self.warper.warp(name, self.cameras[idx].K().astype(np.float32), self.cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            p, mask_warped = self.warper.warp(255 * np.ones((name.shape[0], name.shape[1]), np.uint8), self.cameras[idx].K().astype(np.float32), self.cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            self.compensator.apply(idx, self.corners[idx], image_warped, mask_warped)
            mask_warped = cv2.bitwise_and(cv2.resize(cv2.dilate(self.masks_warped[idx], None), (mask_warped.shape[1],mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT), mask_warped)
            self.blender.feed(cv2.UMat(image_warped.astype(np.int16)), mask_warped, self.corners[idx])
        result, result_mask = self.blender.blend(None, None)
        dst = cv2.normalize(src=result, dst=None, alpha=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return dst

def main(left_image,right_image,work_megapix):
    start_time = timer(start_time=None)
    stitcher = Stitcher(left_image,right_image,work_megapix)
    stitcher.work_megapix = work_megapix
    stitcher.match_features()
    stitcher.refineMask()
    stitcher.fine_adjustments()
    stitcher.corners_masks_sizes()
    stitcher.images_warpedf()
    stitcher.get_warper()
    stitcher.calculate_corners()
    dst = stitcher.composePanorama()
    print("--work_megapix {}, worked | shape {} | estimation time {}".format(work_megapix,dst.shape,timer(start_time)))

    if dst.shape[0] in range(2000,2300) and dst.shape[1] in range(4700,5000):
        cv2.imshow("Stitching result:{} --{}".format(dst.shape,work_megapix),imutils.resize(dst,width=1080))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    for i in np.arange(0.0,1.0,0.01):
        try:
            main(cv2.imread('inputs/left.png'),cv2.imread('inputs/right.png'),i)
        except Exception as e:
            print(e)
            continue
