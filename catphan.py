from abc import abstractmethod
from collections import OrderedDict
from functools import lru_cache
from os import path as osp
import os
import zipfile
import csv
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import filters, measure, segmentation
from matplotlib.pyplot import cm

from . import image
from .geometry import Point, Line
from .profile import CollapsedCircleProfile, SingleProfile
from .roi import DiskROI, RectangleROI, LowContrastDiskROI


class CatPhanBase:
    """分析CatPhan 604模体DICOM CT图像，图像来源于CT或者CBCT扫描
    分析模块包括: Uniformity (CTP719), Low-Contrast module (CTP730), Resolution geometry test module (CTP732).
    """
    # 不同密度插件半径（mm）
    air_bubble_radius_mm = 7
    # 插件距图像中心距离（mm）
    localization_radius = 59

    def __init__(self, folderpath):
        self.origin_slice = 0
        self.catphan_roll = 0
        if not osp.isdir(folderpath):
            raise NotADirectoryError("Path given was not a Directory/Folder")
        self.dicom_stack = image.DicomImageStack(folderpath)
        self.localize()

    @classmethod
    def from_zip(cls, zip_file):
        with TemporaryZipDirectory(zip_file) as temp_zip:
            obj = cls(temp_zip)
        return obj

    def plot_analyzed_image(self, hu=True, uniformity=True, spatial_res=True, low_contrast=True, show=True):
        """Plot the images used in the calculate and summary data.

        Parameters
        ----------
        hu : bool
            Whether to show the HU linearity circles.
        uniformity : bool
            Whether to show the uniformity ROIs.
        spatial_res : bool
            Whether to show the spatial resolution circle outline.
        low_contrast: bool
            Whether to show the low contrast ROIs.
        show : bool
            Whether to plot the image or not.
        """
        # set up grid and axes
        plt.figure(1)
        grid_size = (2, 4)
        unif_ax = plt.subplot2grid(grid_size, (0, 0))
        hu_ax = plt.subplot2grid(grid_size, (0, 1))
        sr_ax = plt.subplot2grid(grid_size, (1, 0))
        locon_ax = plt.subplot2grid(grid_size, (1, 1))
        hu_lin_ax = plt.subplot2grid(grid_size, (0, 2))
        mtf_ax = plt.subplot2grid(grid_size, (0, 3))
        unif_prof_ax = plt.subplot2grid(grid_size, (1, 2), colspan=2)

        # plot the images
        show_section = [uniformity, hu, spatial_res, low_contrast]
        axes = (unif_ax, hu_ax, sr_ax, locon_ax)
        modules = ['ctp486', 'ctp404', 'ctp528', 'ctp515']
        titles = ('Uniformity', 'HU & Geometry', 'Spatial Resolution', 'Low Contrast')
        for show_unit, axis, title, module in zip(show_section, axes, titles, modules):
            if show_unit:
                klass = getattr(self, module)
                axis.imshow(klass.image.array, cmap=cm.gray)
                klass.plot_rois(axis)
                axis.autoscale(tight=True)
                axis.set_title(title)
            else:
                axis.set_frame_on(False)
            axis.axis('off')

        # plot the other sections
        self.ctp404.plot_linearity(hu_lin_ax)
        self.ctp486.plot_profiles(unif_prof_ax)
        self.ctp528.plot_mtf(mtf_ax)

        # finish up
        plt.tight_layout()
        if show:
            plt.show()

    def plot_single_frame(self):
        modules = ['ctp486', 'ctp404', 'ctp528', 'ctp515']
        titles = ('Uniformity', 'HU & Geometry', 'Spatial Resolution', 'Low Contrast')
        for title, module in zip(titles, modules):
            plt.figure(title, figsize=(8, 8))
            klass = getattr(self, module)
            plt.imshow(klass.image.array, cmap=cm.gray)
            plt.show()

    def plot_center_hu(self, filename, **kwargs):
        plt.figure('HU & Genometry')
        plt.axis('off')
        plt.imshow(self.ctp404.image, cmap=cm.gray)
        center = self.ctp404.phan_center
        circle_prof = CollapsedCircleProfile(center, radius=self.localization_radius / self.mm_per_pixel,
                                              image_array=self.ctp404.image, width_ratio=0.02, num_profiles=2)
        ax = plt.gca()
        circle_prof.plot2axes(ax)
        plt.autoscale(tight=True)
        """
        for roi in klass.hu_rois.values():
            roi.plot2axes(ax, edgecolor=roi.plot_color)
        for roi in klass.bg_hu_rois.values():
            roi.plot2axes(ax, edgecolor='blue')
        ax.axis('off')
        """
        plt.savefig(filename, **kwargs)
        plt.show()

    def plot_center_hu_profile(self):
        slice = Slice(self, self.origin_slice, combine=False)
        center = slice.phan_center
        circle_prof = CollapsedCircleProfile(center, radius=self.localization_radius / self.mm_per_pixel,
                                             image_array=slice.image, width_ratio=0.05, num_profiles=5)
        # circle_prof.plot()
        prof = circle_prof.values
        plt.figure('profile', figsize=(8, 8))
        plt.plot(prof)
        ax = plt.gca()
        ax.axis('off')
        with open('profile.csv', "a") as csvfile:
            writer = csv.writer(csvfile)
            # 写入多行用writerows
            writer.writerow(prof)

    def save_analyzed_image(self, filename, **kwargs):
        """Save the analyzed summary plot.

        Parameters
        ----------
        filename : str, file object
            The name of the file to save the image to.
        kwargs :
            Any valid matplotlib kwargs.
        """
        self.plot_analyzed_image(show=False)
        plt.savefig(filename, **kwargs)

    def plot_analyzed_subimage(self, subimage='hu', delta=True, show=True):
        """Plot a specific component of the CBCT analysis.

        Parameters
        ----------
        subimage : {'hu', 'un', 'sp', 'lc', 'mtf', 'lin', 'prof'}
            The subcomponent to plot. Values must contain one of the following letter combinations.
            E.g. ``linearity``, ``linear``, and ``lin`` will all draw the HU linearity values.

            * ``hu`` draws the HU linearity image.
            * ``un`` draws the HU uniformity image.
            * ``sp`` draws the Spatial Resolution image.
            * ``mtf`` draws the RMTF plot.
            * ``lin`` draws the HU linearity values. Used with ``delta``.
            * ``prof`` draws the HU uniformity profiles.
        delta : bool
            Only for use with ``lin``. Whether to plot the HU delta or actual values.
        interactive : bool
            If False (default), the figure is plotted statically as a matplotlib figure.
            If True, the figure is plotted as an HTML page in the default browser that can have tooltips.
            This setting is only applicable for ``mtf``, ``lin``, and ``prof``, i.e. the "regular" plots. Image
            plotting does not work with mpld3.

            .. note:: mpld3 must be installed to use this feature.

        show : bool
            Whether to actually show the plot.
        """
        subimage = subimage.lower()
        plt.clf()
        plt.axis('off')

        if 'hu' in subimage:  # HU, GEO & thickness objects
            plt.imshow(self.ctp404.image.array, cmap=cm.gray)
            self.ctp404.plot_rois(plt.gca())
            plt.autoscale(tight=True)
        elif 'un' in subimage:  # uniformity
            plt.imshow(self.ctp486.image.array, cmap=cm.gray)
            self.ctp486.plot_rois(plt.gca())
            plt.autoscale(tight=True)
        elif 'sp' in subimage:  # SR objects
            plt.imshow(self.ctp528.image.array, cmap=cm.gray)
            self.ctp528.plot_rois(plt.gca())
            plt.autoscale(tight=True)
        elif 'mtf' in subimage:
            subimage = 'mtf'
            plt.axis('on')
            points = self.ctp528.plot_mtf(plt.gca())
        elif 'lc' in subimage:
            plt.imshow(self.ctp515.image.array, cmap=cm.gray)
            self.ctp515.plot_rois(plt.gca())
            plt.autoscale(tight=True)
        elif 'lin' in subimage:
            subimage = 'lin'
            plt.axis('on')
            points = self.ctp404.plot_linearity(plt.gca(), delta)
        elif 'prof' in subimage:
            subimage = 'prof'
            plt.axis('on')
            self.ctp486.plot_profiles(plt.gca())
        else:
            raise ValueError("Subimage parameter {0} not understood".format(subimage))

        if show:
            plt.show()

    def save_analyzed_subimage(self, filename, subimage='hu', **kwargs):
        """Save a component image to file.

        Parameters
        ----------
        filename : str, file object
            The file to write the image to.
        subimage : str
            See :meth:`~pylinac.cbct.CBCT.plot_analyzed_subimage` for parameter info.
        interactive : bool
            If False (default), saves a matplotlib figure as a .png image.
            If True, saves an html file, which can be opened in a browser, etc.

            .. note:: mpld3 must be installed to use this feature.
        """
        self.plot_analyzed_subimage(subimage, show=False)
        plt.savefig(filename, **kwargs)
        if isinstance(filename, str):
            print("CatPhan subimage figure saved to {0}".format(osp.abspath(filename)))

    def _return_results(self):
        """Helper function to spit out values that will be tested."""
        print(self.return_results())
        print("Phantom roll: {0}".format(self.find_phantom_roll()))
        print("Origin slice: {}".format(self.find_origin_slice()))
        mtfs = {}
        for mtf in (30, 50, 80):
            mtfval = self.ctp528.mtf(mtf)
            mtfs[mtf] = mtfval
        print('MTFs: {}'.format(mtfs))

    def localize(self):
        # 找到HU线性模块中心断层图像
        self.origin_slice = self.find_origin_slice()
        # 找到Catphan模体旋转角度
        self.catphan_roll = self.find_phantom_roll()

    @property
    def mm_per_pixel(self):
        """The millimeters per pixel of the DICOM images."""
        return self.dicom_stack.metadata.PixelSpacing[0]

    @lru_cache(maxsize=1)
    def find_origin_slice(self):
        """Using a brute force search of the images, find the median HU linearity slice.

        This method walks through all the images and takes a collapsed circle profile where the HU
        linearity ROIs are. If the profile contains both low (<800) and high (>800) HU values and most values are the same
        (i.e. it's not an artifact), then
        it can be assumed it is an HU linearity slice. The median of all applicable slices is the
        center of the HU slice.

        Returns
        -------
        int
            The middle slice of the HU linearity module.
        """
        hu_slices = []
        # use a machine-learning classifier
        if not hu_slices:
            for image_number in range(0, self.num_images, 2):
                slice = Slice(self, image_number, combine=False)
                try:
                    center = slice.phan_center
                except ValueError:  # a slice without the phantom in view
                    pass
                else:
                    circle_prof = CollapsedCircleProfile(center, radius=self.localization_radius/self.mm_per_pixel, image_array=slice.image, width_ratio=0.05, num_profiles=5)
                    prof = circle_prof.values
                    # determine if the profile contains both low and high values and that most values are the same
                    low_end, high_end = np.percentile(prof, [2, 98])
                    median = np.median(prof)
                    if (low_end < median - 400) and (high_end > median + 400) and (
                                    np.percentile(prof, 80) - np.percentile(prof, 20) < 100):
                        hu_slices.append(image_number)

        if not hu_slices:
            raise ValueError("No slices were found that resembled the HU linearity module")
        hu_slices = np.array(hu_slices)
        c = int(round(np.median(hu_slices)))
        ln = len(hu_slices)
        # drop slices that are way far from median
        hu_slices = hu_slices[((c + ln/2) > hu_slices) & (hu_slices > (c - ln/2))]
        center_hu_slice = int(round(np.median(hu_slices)))
        if self._is_within_image_extent(center_hu_slice):
            return center_hu_slice

    @lru_cache(maxsize=1)
    def find_phantom_roll(self):
        """Determine the "roll" of the phantom.

        This algorithm uses the two air bubbles in the HU slice and the resulting angle between them.

        Returns
        -------
        float : the angle of the phantom in **degrees**.
        """
        def is_right_area(region):
            thresh = np.pi * ((self.air_bubble_radius_mm / self.mm_per_pixel) ** 2)
            return thresh * 1.5 > region.filled_area > thresh / 1.5

        def is_right_eccentricity(region):
            return region.eccentricity < 0.5

        # get edges and make ROIs from it
        slice = Slice(self, self.origin_slice)
        larr, regions, _ = get_regions(slice)
        # find appropriate ROIs and grab the two most centrally positioned ones
        hu_bubbles = [r for r in regions if (is_right_area(r) and is_right_eccentricity(r))]
        central_bubbles = sorted(hu_bubbles, key=lambda x: abs(x.centroid[1] - slice.phan_center.x))[:2]
        sorted_bubbles = sorted(central_bubbles, key=lambda x: x.centroid[0])  # top, bottom
        y_dist = sorted_bubbles[1].centroid[0] - sorted_bubbles[0].centroid[0]
        x_dist = sorted_bubbles[1].centroid[1] - sorted_bubbles[0].centroid[1]
        phan_roll = np.arctan2(y_dist, x_dist)
        anglroll = np.rad2deg(phan_roll) - 90
        return anglroll

    @property
    def num_images(self):
        """Return the number of images loaded.

        Returns
        -------
        float
        """
        return len(self.dicom_stack)

    def _is_within_image_extent(self, image_num):
        """Determine if the image number is beyond the edges of the images (negative or past last image)."""
        if self.num_images - 1 > image_num > 1:
            return True
        else:
            raise ValueError("The determined image number is beyond the image extent. Either the entire dataset "
                             "wasn't loaded or the entire phantom wasn't scanned.")

    @property
    def catphan_size(self):
        """The expected size of the phantom in pixels, based on a 20cm wide phantom.

        Returns
        -------
        float
        """
        phan_area = np.pi*(self.catphan_radius_mm**2)
        return phan_area/(self.mm_per_pixel**2)

    def _zip_images(self):
        """Compress the raw images into a ZIP archive and remove the uncompressed images."""
        zip_name = "{}\CBCT - {}.zip".format(osp.dirname(self.dicom_stack[0].path), self.dicom_stack[0].date_created(format="%A, %I-%M-%S, %B %d, %Y"))
        with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zfile:
            for image in self.dicom_stack:
                zfile.write(image.path, arcname=osp.basename(image.path))
        for image in self.dicom_stack:
            try:
                os.remove(image.path)
            except:
                pass


class CatPhan604(CatPhanBase):
    """A class for loading and analyzing CT DICOM files of a CatPhan 604. Can be from a CBCT or CT scanner
    Analyzes: Uniformity (CTP486), High-Contrast Spatial Resolution (CTP528),
    Image Scaling & HU Linearity (CTP404), and Low contrast (CTP515).
    """
    _classifier_model = '604'
    catphan_radius_mm = 101
    offsets = {
        'Uniformity': -70,
        'Low contrast': -40,
        'Spatial Resolution': 40,
    }

    def return_results(self):
        """Return the results of the analysis as a string. Use with print()."""
        string = ('\n - CatPhan 604 QA Test - \n'
                  'HU Linearity ROIs: {}\n'
                  'HU Passed?: {}\n'
                  'Uniformity ROIs: {}\n'
                  'Uniformity index: {}\n'
                  'Integral non-uniformity: {}\n'
                  'Uniformity Passed?: {}\n'
                  'MTF 30% (lp/mm): {}\n'
                  'MTF 50% (lp/mm): {}\n'
                  'MTF 80% (lp/mm): {}\n'
                  'Low contrast ROIs "seen": {}\n'
                  'Low contrast visibility: {}\n'
                  'Geometric Line Average (mm): {}\n'
                  'Geometry Passed?: {}\n'
                  'Slice Thickness (mm): {}\n'
                  'Slice Thickness Passed? {}\n').format(self.ctp404.hu_roi_vals, self.ctp404.passed_hu,
                                                         self.ctp486.get_ROI_vals(), self.ctp486.uniformity_index,
                                                         self.ctp486.integral_non_uniformity, self.ctp486.overall_passed,
                                                         self.ctp528.mtf(30),
                                                         self.ctp528.mtf(50),
                                                         self.ctp528.mtf(80),
                                                         self.ctp515.rois_visible, self.ctp404.lcv,
                                                         self.ctp404.avg_line_length, self.ctp404.passed_geometry,
                                                         self.ctp404.meas_slice_thickness,
                                                         self.ctp404.passed_thickness)
        return string

    def write_csv(self, filename="test.csv"):
        """Return the results of the analysis as a list. Write to CSV file. by Jinyan-hu"""
        list = []
        """
        for key, value in self.ctp404.hu_rois.items():
            list.append(value.pixel_value)
        #for key, value in self.ctp486.rois.items():
        #    list.append(value.pixel_value)
        for key, value in self.ctp528.mtfs.items():
            list.append(key)
            list.append(value)
        list.append(self.ctp486.uniformity_index)
        #list.append(self.ctp486.integral_non_uniformity)
        #list.append(self.ctp528.mtf(50))
        list.append(self.ctp515.rois_visible)
        list.append(self.ctp404.lcv)
        list.append(self.ctp404.avg_line_length)
        list.append(self.ctp404.meas_slice_thickness)
        # print(list)
        """
        for key, value in self.ctp528.mtfs.items():
            list.append(key)
            list.append(value)
        with open(filename, "a") as csvfile:
            writer = csv.writer(csvfile)
            # 写入多行用writerows
            writer.writerow(list)

    def analyze(self, hu_tolerance=40, scaling_tolerance=1, thickness_tolerance=0.2,
                low_contrast_tolerance=1, cnr_threshold=15, zip_after=False):
        """Single-method full analysis of CBCT DICOM files.

        Parameters
        ----------
        hu_tolerance : int
            The HU tolerance value for both HU uniformity and linearity.
        scaling_tolerance : float, int
            The scaling tolerance in mm of the geometric nodes on the HU linearity slice (CTP404 module).
        thickness_tolerance : float, int
            The tolerance of the thickness calculation in mm, based on the wire ramps in the CTP404 module.

            .. warning:: Thickness accuracy degrades with image noise; i.e. low mAs images are less accurate.

        low_contrast_tolerance : int
            The number of low-contrast bubbles needed to be "seen" to pass.
        cnr_threshold : float, int
            The threshold for "detecting" low-contrast image. See RTD for calculation info.
        zip_after : bool
            If the CT images were not compressed before analysis and this is set to true, pylinac will compress
            the analyzed images into a ZIP archive.
        """
        self.ctp404 = CTP404(self, offset=0, hu_tolerance=hu_tolerance, thickness_tolerance=thickness_tolerance,
                             scaling_tolerance=scaling_tolerance)
        self.ctp486 = CTP486(self, offset=self.offsets['Uniformity'], tolerance=hu_tolerance)
        self.ctp528 = CTP528(self, tolerance=None, offset=self.offsets['Spatial Resolution'])
        self.ctp515 = CTP515(self, tolerance=low_contrast_tolerance, cnr_threshold=cnr_threshold, offset=self.offsets['Low contrast'])
        if zip_after and not self.was_from_zip:
            self._zip_images()


class HUDiskROI(DiskROI):
    """An HU ROI object. Represents a circular area measuring either HU sample (Air, Poly, ...)
    or HU uniformity (bottom, left, ...).
    """
    def __init__(self, array, angle, roi_radius, dist_from_center, phantom_center, nominal_value=None, tolerance=None,
                 background_median=None, background_std=None):
        """
        Parameters
        ----------
        nominal_value : int
            The nominal pixel value of the HU ROI.
        tolerance : int
            The roi pixel value tolerance.
        """
        super().__init__(array, angle, roi_radius, dist_from_center, phantom_center)
        self.nominal_val = nominal_value
        self.tolerance = tolerance
        self.background_median = background_median
        self.background_std = background_std

    @property
    def cnr(self):
        """The contrast-to-noise value of the HU disk"""
        return 2*abs(self.pixel_value - self.background_median) / (self.std + self.background_std)

    @property
    def value_diff(self):
        """The difference in HU between measured and nominal."""
        return abs(self.pixel_value - self.nominal_val)

    @property
    def passed(self):
        """Boolean specifying if ROI pixel value was within tolerance of the nominal value."""
        return self.value_diff <= self.tolerance

    @property
    def plot_color(self):
        """Return one of two colors depending on if ROI passed."""
        return 'green' if self.passed else 'red'


class ThicknessROI(RectangleROI):
    """A rectangular ROI that measures the angled wire rod in the HU linearity slice which determines slice thickness."""

    @property
    @lru_cache(maxsize=1)
    def long_profile(self):
        """The profile along the axis perpendicular to ramped wire."""
        img = image.load(self.pixel_array)
        img.filter(size=1, kind='gaussian')
        prof = SingleProfile(img.array.max(axis=np.argmin(img.shape)))
        return prof

    @property
    @lru_cache(maxsize=1)
    def wire_fwhm(self):
        """The FWHM of the wire in pixels."""
        return self.long_profile.fwxm(x=50, interpolate=True)

    @property
    def plot_color(self):
        """The plot color."""
        return 'blue'


class ROIManagerMixin:
    """Class for handling multiple ROIs. Used for the HU linearity, Uniformity, Geometry, Low-contrast, and Thickness slices.

    Attributes
    ----------
    dist2rois_mm : int, float
        The distance from the phantom center to the ROIs, in mm.
    roi_radius_mm : int, float
        The radius of the ROIs, in mm.
    roi_names : list
        The names of the ROIs.
    roi_nominal_angles : list
        The nominal angles of the ROIs; must be same order as ``roi_names``.
    """
    dist2rois_mm = 0
    roi_radius_mm = 0
    roi_names = []
    roi_nominal_angles = []

    @property
    def roi_angles(self):
        """The ROI angles, corrected for phantom roll."""
        return np.array(self.roi_nominal_angles) + self.catphan_roll

    @property
    def dist2rois(self):
        """Distance from the phantom center to the ROIs, corrected for pixel spacing."""
        return self.dist2rois_mm / self.mm_per_pixel

    @property
    def roi_radius(self):
        """ROI radius, corrected for pixel spacing."""
        return self.roi_radius_mm / self.mm_per_pixel

    def get_ROI_vals(self):
        """Return a dict of the HU values of the HU ROIs."""
        return {key: val.pixel_value for key, val in self.rois.items()}

    def plot_rois(self, axis, threshold=None):
        """Plot the ROIs to the axis."""
        for roi in self.rois.values():
            if not threshold:
                roi.plot2axes(axis, edgecolor=roi.plot_color)
            else:
                roi.plot2axes(axis, edgecolor=roi.plot_color_cnr)


class Slice:
    """Base class for analyzing specific slices of a CBCT dicom set."""
    def __init__(self, catphan, slice_num=None, combine=True, combine_method='mean', num_slices=0):
        """
        Parameters
        ----------
        catphan : `~pylinac.cbct.CatPhanBase` instance.
        slice_num : int
            The slice number of the DICOM array desired. If None, will use the ``slice_num`` property of subclass.
        combine : bool
            If True, combines the slices +/- ``num_slices`` around the slice of interest to improve signal/noise.
        combine_method : {'mean', 'max'}
            How to combine the slices if ``combine`` is True.
        num_slices : int
            The number of slices on either side of the nominal slice to combine to improve signal/noise; only
            applicable if ``combine`` is True.
        """
        if slice_num is not None:
            self.slice_num = slice_num
        if combine:
            array = combine_surrounding_slices(catphan.dicom_stack, self.slice_num, mode=combine_method, slices_plusminus=num_slices)
        else:
            array = catphan.dicom_stack[self.slice_num].array
        self.image = image.load(array)
        self.catphan_size = catphan.catphan_size
        self.mm_per_pixel = catphan.mm_per_pixel

    @property
    def __getitem__(self, item):
        return self.image.array[item]

    @property
    @lru_cache(maxsize=1)
    def phan_center(self):
        """Determine the location of the center of the phantom.

        The image is analyzed to see if:
        1) the CatPhan is even in the image (if there were any ROIs detected)
        2) an ROI is within the size criteria of the catphan
        3) the ROI area that is filled compared to the bounding box area is close to that of a circle

        Raises
        ------
        ValueError
            If any of the above conditions are not met.
        """
        # convert the slice to binary and label ROIs
        edges = filters.scharr(self.image.as_type(np.float))
        if np.max(edges) < 0.1:
            raise ValueError("Unable to locate Catphan")
        larr, regionprops, num_roi = get_regions(self, fill_holes=True, threshold='mean')
        # check that there is at least 1 ROI
        if num_roi < 1 or num_roi is None:
            raise ValueError("Unable to locate the CatPhan")
        catphan_region = sorted(regionprops, key=lambda x: np.abs(x.filled_area - self.catphan_size))[0]
        if (self.catphan_size * 1.2 < catphan_region.filled_area) or (catphan_region.filled_area < self.catphan_size / 1.2):
            raise ValueError("Unable to locate Catphan")
        center_pixel = catphan_region.centroid
        return Point(center_pixel[1], center_pixel[0])


class CatPhanModule(Slice, ROIManagerMixin):
    """Base class for a CTP module.
    """
    combine_method = 'mean'
    num_slices = 0

    def __init__(self, catphan, tolerance, offset=0):
        """
        Parameters
        ----------
        catphan : `~pylinac.cbct.CatPhanBase` instance.
        tolerance : float
        offset : int, float
        """
        self._offset = offset
        self.origin_slice = catphan.origin_slice
        self.tolerance = tolerance
        self.slice_thickness = catphan.dicom_stack.metadata.SliceThickness
        self.catphan_roll = catphan.catphan_roll
        self.mm_per_pixel = catphan.mm_per_pixel
        Slice.__init__(self, catphan, combine_method=self.combine_method, num_slices=self.num_slices)
        self.preprocess(catphan)
        self._setup_rois()

    def preprocess(self, catphan):
        """A preprocessing step before analyzing the CTP module.

        Parameters
        ----------
        catphan : `~pylinac.cbct.CatPhanBase` instance.
        """
        pass

    @property
    def slice_num(self):
        """The slice number of the spatial resolution module.

        Returns
        -------
        float
        """
        return int(self.origin_slice+round(self._offset/self.slice_thickness))

    @abstractmethod
    def _setup_rois(self):
        pass


class CTP404(CatPhanModule):
    """Class for analysis of the HU linearity, geometry, and slice thickness regions of the CTP404.
    """

    def __init__(self, catphan, offset, hu_tolerance, thickness_tolerance, scaling_tolerance):
        """
        Parameters
        ----------
        catphan : `~pylinac.cbct.CatPhanBase` instance.
        offset : float
        hu_tolerance : float
        thickness_tolerance : float
        scaling_tolerance : float
        """
        self.mm_per_pixel = catphan.mm_per_pixel
        self.hu_tolerance = hu_tolerance
        self.thickness_tolerance = thickness_tolerance
        self.scaling_tolerance = scaling_tolerance
        if isinstance(catphan, CatPhan604):
            self.hu= {
                'distance to ROIs': 58/self.mm_per_pixel,
                'ROI radius': 5/self.mm_per_pixel,
                'ROIs': {
                    'Air': {'value': -1000, 'angle': -90},
                    'PMP': {'value': -200, 'angle': -120},
                    '50Bone': {'value': 725, 'angle': -150},
                    'LDPE': {'value': -100, 'angle': 180},
                    'Poly': {'value': -35, 'angle': 120},
                    'Acrylic': {'value': 120, 'angle': 60},
                    '20Bone': {'value': 240, 'angle': 30},
                    'Delrin': {'value': 340, 'angle': 0},
                    'Teflon': {'value': 990, 'angle': -60},
                },
                'background ROI angles': [-30, -210]
            }
        self.bg_hu_rois = OrderedDict()
        self.hu_rois = OrderedDict()

        # thickness
        height = 40
        width = 10
        self.thickness = {
            'distance to ROIs': 38/self.mm_per_pixel,
            'ROIs': {
                'Left': {'angle': 180, 'width': width, 'height': height},
                'Bottom': {'angle': 90, 'width': height, 'height': width},
                'Right': {'angle': 0, 'width': width, 'height': height},
                'Top': {'angle': -90, 'width': height, 'height': width},
            }
        }
        self.thickness_rois = OrderedDict()

        # geometry
        self.geometry = {
            'line_assignments': {'Top-Horizontal': (0, 1),
                                 'Bottom-Horizontal': (2, 3),
                                 'Left-Vertical': (0, 2),
                                 'Right-Vertical': (1, 3)},
            'Image size': 35/self.mm_per_pixel,
        }
        self.lines = OrderedDict()
        super().__init__(catphan, tolerance=None, offset=offset)

    def preprocess(self, catphan):
        # for the thickness analysis image, combine thin slices or just use one slice if slices are thick
        if float(catphan.dicom_stack.metadata.SliceThickness) < 3.5:
            self.pad = 1
        else:
            self.pad = 0
        self.thickness_image = Slice(catphan, combine_method='mean', num_slices=self.num_slices+self.pad, slice_num=self.slice_num).image

    def _setup_rois(self):
        self._setup_hu_rois()
        self._setup_thickness_rois()
        self._setup_geometry_rois()

    def _setup_hu_rois(self):
        # background ROIs
        for idx, angle in enumerate(self.hu['background ROI angles']):
            self.bg_hu_rois[idx] = HUDiskROI(self.image, angle+self.catphan_roll,
                                             self.hu['ROI radius'],
                                             self.hu['distance to ROIs'],
                                             self.phan_center)
        # # center background ROI
        self.bg_hu_rois[idx+1] = HUDiskROI(self.image, 0, self.hu['ROI radius'], 0, self.phan_center)
        bg_median = np.mean([roi.pixel_value for roi in self.bg_hu_rois.values()])
        bg_std = np.std([roi.pixel_value for roi in self.bg_hu_rois.values()])
        # actual HU linearity ROIs
        for name, values in self.hu['ROIs'].items():
            self.hu_rois[name] = HUDiskROI(self.image, values['angle']+self.catphan_roll, self.hu['ROI radius'], self.hu['distance to ROIs'],
                                           self.phan_center, values['value'], self.hu_tolerance, background_median=bg_median,
                                           background_std=bg_std)

    def _setup_thickness_rois(self):
        for name, value in self.thickness['ROIs'].items():
            self.thickness_rois[name] = ThicknessROI(self.thickness_image, value['width']/self.mm_per_pixel,
                                                     value['height']/self.mm_per_pixel, value['angle'],
                                                     self.thickness['distance to ROIs'], self.phan_center)

    def _setup_geometry_rois(self):
        boxsize = self.geometry['Image size']
        xbounds = (int(self.phan_center.x-boxsize), int(self.phan_center.x+boxsize))
        ybounds = (int(self.phan_center.y-boxsize), int(self.phan_center.y+boxsize))
        geo_img = self.image[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        larr, regionprops, num_roi = get_regions(geo_img, fill_holes=True, clear_borders=False)
        # check that there is at least 1 ROI
        if num_roi < 4:
            raise ValueError("Unable to locate the Geometric nodes")
        elif num_roi > 4:
            regionprops = sorted(regionprops, key=lambda x: x.filled_area, reverse=True)[:4]
        sorted_regions = sorted(regionprops, key=lambda x: (2*x.centroid[0]+x.centroid[1]))
        centers = [Point(r.weighted_centroid[1]+xbounds[0], r.weighted_centroid[0]+ybounds[0]) for r in sorted_regions]
        # # setup the geometric lines
        for name, order in self.geometry['line_assignments'].items():
            self.lines[name] = GeometricLine(centers[order[0]], centers[order[1]], self.mm_per_pixel, self.scaling_tolerance)

    @property
    def lcv(self):
        """The low-contrast visibility"""
        return 2 * abs(self.hu_rois['LDPE'].pixel_value - self.hu_rois['Poly'].pixel_value) / (self.hu_rois['LDPE'].std + self.hu_rois['Poly'].std)

    def plot_linearity(self, axis=None, plot_delta=True):
        """Plot the HU linearity values to an axis.

        Parameters
        ----------
        axis : None, matplotlib.Axes
            The axis to plot the values on. If None, will create a new figure.
        plot_delta : bool
            Whether to plot the actual measured HU values (False), or the difference from nominal (True).
        """
        nominal_x_values = [roi.nominal_val for roi in self.hu_rois.values()]
        if axis is None:
            fig, axis = plt.subplots()
        if plot_delta:
            values = [roi.value_diff for roi in self.hu_rois.values()]
            nominal_measurements = [0]*len(values)
            ylabel = 'HU Delta'
        else:
            values = [roi.pixel_value for roi in self.hu_rois.values()]
            nominal_measurements = nominal_x_values
            ylabel = 'Measured Values'
        points = axis.plot(nominal_x_values, values, 'g+', markersize=15, mew=2)
        axis.plot(nominal_x_values, nominal_measurements)
        axis.plot(nominal_x_values, np.array(nominal_measurements) + self.hu_tolerance, 'r--')
        axis.plot(nominal_x_values, np.array(nominal_measurements) - self.hu_tolerance, 'r--')
        axis.margins(0.05)
        axis.grid('on')
        axis.set_xlabel("Nominal Values")
        axis.set_ylabel(ylabel)
        axis.set_title("HU linearity")
        return points

    @property
    def passed_hu(self):
        """Boolean specifying whether all the ROIs passed within tolerance."""
        return all(roi.passed for roi in self.hu_rois.values())

    @property
    def hu_roi_vals(self):
        return {key: value.pixel_value for key, value in self.hu_rois.items()}

    def plot_rois(self, axis):
        """Plot the ROIs onto the image, as well as the background ROIs"""
        # plot HU linearity ROIs
        #for roi in self.hu_rois.values():
        #    roi.plot2axes(axis, edgecolor=roi.plot_color)
        #for roi in self.bg_hu_rois.values():
        #    roi.plot2axes(axis, edgecolor='blue')
        # plot thickness ROIs
        for roi in self.thickness_rois.values():
            roi.plot2axes(axis, edgecolor='blue')
        # plot geometry lines
        for line in self.lines.values():
            line.plot2axes(axis, color=line.pass_fail_color)

    @property
    def passed_thickness(self):
        """Whether the slice thickness was within tolerance from nominal."""
        return self.slice_thickness-self.thickness_tolerance<self.meas_slice_thickness<self.slice_thickness+self.thickness_tolerance

    @property
    def meas_slice_thickness(self):
        """The average slice thickness for the 4 wire measurements in mm."""
        print(sorted(roi.wire_fwhm*self.mm_per_pixel*0.42 for roi in self.thickness_rois.values()))
        return np.mean(sorted(roi.wire_fwhm*self.mm_per_pixel*0.42 for roi in self.thickness_rois.values()))/(1+2*self.pad)

    @property
    def avg_line_length(self):
        return np.mean([line.length_mm for line in self.lines.values()])

    @property
    def passed_geometry(self):
        """Returns whether all the line lengths were within tolerance."""
        return all(line.passed for line in self.lines.values())


class CTP486(CatPhanModule):
    """Class for analysis of the Uniformity slice of the CTP module. Measures 5 ROIs around the slice that
    should all be close to the same value.
    """
    dist2rois_mm = 53
    roi_radius_mm = 10
    roi_data = {
        'Top': {'angle': 90},
        'Right': {'angle': 0},
        'Bottom': {'angle': -90},
        'Left': {'angle': 180},
        'Center': {'angle': 0},
    }

    def _setup_rois(self):
        self.rois = OrderedDict()
        for name, data in self.roi_data.items():
            distance = self.dist2rois if name != 'Center' else 0
            self.rois[name] = HUDiskROI(self.image, data['angle']+self.catphan_roll, self.roi_radius, distance,
                                        self.phan_center, 0, self.tolerance)

    def plot_profiles(self, axis=None):
        """Plot the horizontal and vertical profiles of the Uniformity slice.

        Parameters
        ----------
        axis : None, matplotlib.Axes
            The axis to plot on; if None, will create a new figure.
        """
        if axis is None:
            fig, axis = plt.subplots()
        horiz_data = self.image[int(self.phan_center.y), :]
        vert_data = self.image[:, int(self.phan_center.x)]
        axis.plot(horiz_data, 'g', label='Horizontal')
        axis.plot(vert_data, 'b', label='Vertical')
        axis.autoscale(tight=True)
        # TODO: replace .plot() calls with .axhline() calls when mpld3 fixes functionality
        axis.plot([i for i in range(len(horiz_data))], [self.tolerance] * len(horiz_data), 'r-', linewidth=3)
        axis.plot([i for i in range(len(horiz_data))], [-self.tolerance] * len(horiz_data), 'r-', linewidth=3)
        axis.grid('on')
        axis.set_ylabel("HU")
        axis.legend(loc=8, fontsize='small', title="")
        axis.set_title("Uniformity Profiles")

    @property
    def overall_passed(self):
        """Boolean specifying whether all the ROIs passed within tolerance."""
        return all(roi.passed for roi in self.rois.values())

    @property
    def uniformity_index(self):
        """The Uniformity Index"""
        center = self.rois['Center']
        uis = [100*((roi.pixel_value-center.pixel_value)/(center.pixel_value+1000)) for roi in self.rois.values()]
        abs_uis = np.abs(uis)
        return uis[np.argmax(abs_uis)]

    @property
    def integral_non_uniformity(self):
        """The Integral Non-Uniformity"""
        maxhu = max(roi.pixel_value for roi in self.rois.values())
        minhu = min(roi.pixel_value for roi in self.rois.values())
        return (maxhu - minhu)/(maxhu + minhu + 2000)


class CTP528(CatPhanModule):
    """Class for analysis of the Spatial Resolution slice of the CBCT dicom data set.

    A collapsed circle profile is taken of the line-pair region. This profile is search for
    peaks and valleys. The MTF is calculated from those peaks & valleys.

    Attributes
    ----------
    radius2linepairs_mm : float
        The radius in mm to the line pairs.
    """
    radius2linepairs_mm = 47

    def _setup_rois(self):
        pass

    @property
    def sr_rois(self):
        """Spatial resolution ROI characteristics.

        Returns
        -------
        dict
        """
        rois = OrderedDict()
        rois['region 1'] = {'start': 0, 'end': 0.12, 'num peaks': 2, 'num valleys': 1, 'peak spacing': 0.021, 'gap size (cm)': 0.5, 'lp/mm': 0.2}
        rois['region 2'] = {'start': 0.10, 'end': 0.183, 'num peaks': 3, 'num valleys': 2, 'peak spacing': 0.01, 'gap size (cm)': 0.25, 'lp/mm': 0.4}
        rois['region 3'] = {'start': 0.183, 'end': 0.245, 'num peaks': 4, 'num valleys': 3, 'peak spacing': 0.006, 'gap size (cm)': 0.167, 'lp/mm': 0.6}
        rois['region 4'] = {'start': 0.245, 'end': 0.288, 'num peaks': 4, 'num valleys': 3, 'peak spacing': 0.00557, 'gap size (cm)': 0.125, 'lp/mm': 0.8}
        rois['region 5'] = {'start': 0.288, 'end': 0.3367, 'num peaks': 4, 'num valleys': 3, 'peak spacing': 0.004777, 'gap size (cm)': 0.1, 'lp/mm': 1.0}
        rois['region 6'] = {'start': 0.3367, 'end': 0.3885, 'num peaks': 5, 'num valleys': 4, 'peak spacing': 0.00398, 'gap size (cm)': 0.083, 'lp/mm': 1.2}
        rois['region 7'] = {'start': 0.3885, 'end': 0.4355, 'num peaks': 5, 'num valleys': 4, 'peak spacing': 0.00358, 'gap size (cm)': 0.071, 'lp/mm': 1.4}
        rois['region 8'] = {'start': 0.4355, 'end': 0.4801, 'num peaks': 5, 'num valleys': 4, 'peak spacing': 0.0027866, 'gap size (cm)': 0.063, 'lp/mm': 1.6}
        return rois

    @property
    def lp_freq(self):
        """Line pair frequencies in line pair/mm.

        Returns
        -------
        list
        """
        return [v['lp/mm'] for v in self.sr_rois.values()]

    @property
    @lru_cache(maxsize=1)
    def mtfs(self):
        """The Relative MTF of the line pairs, normalized to the first region.

        Returns
        -------
        dict
        """
        mtfs = OrderedDict()
        for key, value in self.sr_rois.items():
            max_values = self.circle_profile.find_peaks(min_distance=value['peak spacing'], max_number=value['num peaks'],
                                                        search_region=(value['start'], value['end']), kind='value')
            # check that the right number of peaks were found before continuing, otherwise stop searching for regions
            if len(max_values) != value['num peaks']:
                continue
            upper_mean = max_values.mean()
            max_indices = self.circle_profile.find_peaks(min_distance=value['peak spacing'], max_number=value['num peaks'],
                                                         search_region=(value['start'], value['end']), kind='index')
            lower_mean = self.circle_profile.find_valleys(min_distance=value['peak spacing'], max_number=value['num valleys'],
                                                          search_region=(min(max_indices), max(max_indices)), kind='value').mean()
            mtfs[key] = (upper_mean - lower_mean) / (upper_mean + lower_mean)
        if not mtfs:
            raise ValueError("Did not find any spatial resolution pairs to analyze. File an issue on github (https://github.com/jrkerns/pylinac/issues) if this is a valid dataset.")

        # normalize mtf
        norm = mtfs['region 1']
        for key, value in mtfs.items():
            mtfs[key] = value/norm
        return mtfs

    @property
    def radius2linepairs(self):
        """Radius from the phantom center to the line-pair region, corrected for pixel spacing."""
        return self.radius2linepairs_mm / self.mm_per_pixel

    def plot_rois(self, axis):
        """Plot the circles where the profile was taken within."""
        self.circle_profile.plot2axes(axis, edgecolor='blue', plot_peaks=False)

    def preprocess(self, catphan):
        if isinstance(catphan, CatPhan604):
            self.start_angle = np.pi
            self.ccw = True

    @property
    @lru_cache(maxsize=1)
    def circle_profile(self):
        """Calculate the median profile of the Line Pair region.

        Returns
        -------
        :class:`pylinac.core.profile.CollapsedCircleProfile` : A 1D profile of the Line Pair region.
        """
        circle_profile = CollapsedCircleProfile(self.phan_center, self.radius2linepairs, image_array=self.image,
                                                start_angle=self.start_angle + np.deg2rad(self.catphan_roll),
                                                width_ratio=0.04, sampling_ratio=2, ccw=self.ccw)
        circle_profile.filter(0.001, kind='gaussian')
        circle_profile.ground()
        return circle_profile

    def mtf(self, percent=None, region=None):
        """Return the MTF value of the spatial resolution. Only one of the two parameters may be used.

        Parameters
        ----------
        percent : int, float
            The percent relative MTF; i.e. 0-100.
        region : int
            The line-pair region desired (1-6).

        Returns
        -------
        float : the line-pair resolution at the given MTF percent or region.
        """
        if (region is None and percent is None) or (region is not None and percent is not None):
            raise ValueError("Must pass in either region or percent")
        if percent is not None:
            y_vals = list(self.mtfs.values())
            x_vals_intrp = np.arange(self.lp_freq[0], self.lp_freq[len(y_vals)-1], 0.01)
            x_vals = self.lp_freq[:len(y_vals)]
            y_vals_intrp = np.interp(x_vals_intrp, x_vals, y_vals)
            mtf_percent = x_vals_intrp[np.argmin(np.abs(y_vals_intrp - (percent / 100)))]
            return self.simple_round(mtf_percent, 2)
        elif region is not None:
            return self.line_pair_mtfs[region]

    def simple_round(self, number, decimals=0):
        """Round a number to the given number of decimals. Fixes small floating number errors."""
        num = int(round(number * 10 ** decimals))
        num /= 10 ** decimals
        return num

    def plot_mtf(self, axis=None):
        """Plot the Relative MTF.

        Parameters
        ----------
        axis : None, matplotlib.Axes
            The axis to plot the MTF on. If None, will create a new figure.
        """
        if axis is None:
            fig, axis = plt.subplots()
        mtf_vals = list(self.mtfs.values())
        points = axis.plot(self.lp_freq[:len(mtf_vals)], mtf_vals, marker='o')
        axis.margins(0.05)
        axis.grid('on')
        axis.set_xlabel('Line pairs / mm')
        axis.set_ylabel("Relative MTF")
        axis.set_title('RMTF')
        return points


class GeometricLine(Line):
    """Represents a line connecting two nodes/ROIs on the Geometry Slice.

    Attributes
    ----------
    nominal_length_mm : int, float
        The nominal distance between the geometric nodes, in mm.
    """
    nominal_length_mm = 50

    def __init__(self, geo_roi1, geo_roi2, mm_per_pixel, tolerance):
        """
        Parameters
        ----------
        geo_roi1 : GEO_ROI
            One of two ROIs representing one end of the line.
        geo_roi2 : GEO_ROI
            The other ROI which is the other end of the line.
        mm_per_pixel : float
            The mm/pixel value.
        tolerance : int, float
            The tolerance of the geometric line, in mm.
        """
        super().__init__(geo_roi1, geo_roi2)
        self.mm_per_pixel = mm_per_pixel
        self.tolerance = tolerance

    @property
    def passed(self):
        """Whether the line passed tolerance."""
        return self.nominal_length_mm - self.tolerance < self.length_mm < self.nominal_length_mm + self.tolerance

    @property
    def pass_fail_color(self):
        """Plot color for the line, based on pass/fail status."""
        return 'green' if self.passed else 'red'

    @property
    def length_mm(self):
        """Return the length of the line in mm."""
        return self.length*self.mm_per_pixel


class CTP515(CatPhanModule):
    """Class for analysis of the low contrast slice of the CTP module. Low contrast is measured by obtaining
    the average pixel value of the contrast ROIs and comparing that value to the average background value. To obtain
    a more "human" detection level, the contrast (which is largely the same across different-sized ROIs) is multiplied
    by the diameter. This value is compared to the contrast threshold to decide if it can be "seen".
    """
    num_slices = 3
    dist2rois_mm = 50
    bg_roi_radius_mm = 4
    inner_bg_dist_mm = [37, 39, 40, 40.5, 41.5, 41.5]
    outer_bg_dist_mm = [63, 61, 60, 59.5, 58.5, 58.5]
    roi_radius_mm = [6, 3.5, 3, 2.5, 2, 1.5]
    roi_names = ['15', '9', '8', '7', '6', '5']
    roi_nominal_angles = [-87.4, -69.1, -52.7, -38.5, -25.1, -12.9]

    def __init__(self, catphan, tolerance, cnr_threshold, offset=0):
        self.cnr_threshold = cnr_threshold
        super().__init__(catphan, tolerance=tolerance, offset=offset)

    def _setup_rois(self):
        self.rois = OrderedDict()
        self.inner_bg_rois = OrderedDict()
        self.outer_bg_rois = OrderedDict()
        for idx, (name, angle, radius) in enumerate(zip(self.roi_names, self.roi_angles, self.roi_radius)):
            self.inner_bg_rois[name] = LowContrastDiskROI(self.image, angle, self.bg_roi_radius, self.inner_bg_dist[idx],
                                                          self.phan_center, cnr_threshold=self.cnr_threshold)
            self.outer_bg_rois[name] = LowContrastDiskROI(self.image, angle, self.bg_roi_radius, self.outer_bg_dist[idx],
                                                          self.phan_center, cnr_threshold=self.cnr_threshold)
            background_val = np.mean([self.inner_bg_rois[name].pixel_value, self.outer_bg_rois[name].pixel_value])
            self.rois[name] = LowContrastDiskROI(self.image, angle, radius, self.dist2rois,
                                                 self.phan_center, background=background_val, cnr_threshold=self.cnr_threshold)

    @property
    def inner_bg_dist(self):
        return np.array(self.inner_bg_dist_mm) / self.mm_per_pixel

    @property
    def outer_bg_dist(self):
        return np.array(self.outer_bg_dist_mm) / self.mm_per_pixel

    @property
    def bg_roi_radius(self):
        """A list of the ROI radii, scaled to pixels."""
        return self.bg_roi_radius_mm / self.mm_per_pixel

    @property
    def roi_radius(self):
        """A list of the ROI radii, scaled to pixels."""
        return [radius / self.mm_per_pixel for radius in self.roi_radius_mm]

    @property
    def rois_visible(self):
        """The number of ROIs "visible"."""
        return sum(roi.passed_cnr_constant for roi in self.rois.values())

    def plot_rois(self, axis):
        """Plot the ROIs to an axis."""
        super().plot_rois(axis, threshold='constant')
        for roi in self.inner_bg_rois.values():
            roi.plot2axes(axis, 'green')
        for roi in self.outer_bg_rois.values():
            roi.plot2axes(axis, 'green')

    @property
    def overall_passed(self):
        """Whether there were enough low contrast ROIs "seen"."""
        return sum(roi.passed_constant for roi in self.rois.values()) >= self.tolerance

    def plot_contrast(self, axis=None):
        """Plot the contrast constant.
        Parameters
        ----------
        axis : None, matplotlib.Axes
            The axis to plot the contrast on. If None, will create a new figure.
        """
        if axis is None:
            fig, axis = plt.subplots()
        else:
            axis = axis.twinx().twiny()
        sizes = np.array(list(self.rois.keys()), dtype=int)
        contrasts = [roi.contrast_constant for roi in self.rois.values()]
        points = axis.plot(sizes, contrasts)
        axis.margins(0.05)
        axis.grid('on')
        axis.set_xlabel('ROI size (mm)')
        axis.set_ylabel("Contrast * Diameter")
        return points


def get_regions(slice_or_arr, fill_holes=False, clear_borders=True, threshold='otsu'):
    """Get the skimage regions of a black & white image."""
    if threshold == 'otsu':
        thresmeth = filters.threshold_otsu
    elif threshold == 'mean':
        thresmeth = np.mean
    if isinstance(slice_or_arr, Slice):
        edges = filters.scharr(slice_or_arr.image.array.astype(np.float))
        center = slice_or_arr.image.center
    elif isinstance(slice_or_arr, np.ndarray):
        edges = filters.scharr(slice_or_arr)
        center = (int(edges.shape[1]/2), int(edges.shape[0]/2))
    edges = filters.gaussian(edges, sigma=1)
    if isinstance(slice_or_arr, Slice):
        box_size = 110/slice_or_arr.mm_per_pixel
        thres_img = edges[int(center.y-box_size):int(center.y+box_size),
                          int(center.x-box_size):int(center.x+box_size)]
        thres = thresmeth(thres_img)
    else:
        thres = thresmeth(edges)
    bw = edges > thres
    if clear_borders:
        segmentation.clear_border(bw, buffer_size=int(0.01*bw.shape[0]), in_place=True)
    if fill_holes:
        bw = ndimage.binary_fill_holes(bw)
    labeled_arr, num_roi = measure.label(bw, return_num=True)
    regionprops = measure.regionprops(labeled_arr, edges)
    return labeled_arr, regionprops, num_roi


def combine_surrounding_slices(dicomstack, nominal_slice_num, slices_plusminus=1, mode='mean'):
    """Return an array that is the combination of a given slice and a number of slices surrounding it.

    Parameters
    ----------
    dicomstack : `~pylinac.core.image.DicomImageStack`
        The CBCT DICOM stack.
    nominal_slice_num : int
        The slice of interest (along 3rd dim).
    slices_plusminus: int
        How many slices plus and minus to combine (also along 3rd dim).
    mode : {'mean', 'median', 'max}
        Specifies the method of combination.

    Returns
    -------
    combined_array : numpy.array
        The combined array of the DICOM stack slices.
    """
    slices = range(nominal_slice_num - slices_plusminus, nominal_slice_num + slices_plusminus + 1)
    arrays = tuple(dicomstack[s].array for s in slices)
    array_stack = np.dstack(arrays)
    if mode == 'mean':
        combined_array = np.mean(array_stack, 2)
    elif mode == 'median':
        combined_array = np.median(array_stack, 2)
    else:
        combined_array = np.max(array_stack, 2)
    return combined_array


class TemporaryZipDirectory(TemporaryDirectory):
    """Creates a temporary directory that unpacks a ZIP archive."""

    def __init__(self, zfile):
        """
        Parameters
        ----------
        zfile : str
            String that points to a ZIP archive.
        """
        super().__init__()
        zfiles = zipfile.ZipFile(zfile)
        zfiles.extractall(path=self.name)
