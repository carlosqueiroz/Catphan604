from .geometry import Circle, Point, Rectangle
import numpy as np
from functools import lru_cache


class DiskROI(Circle):
    """An class representing a disk-shaped Region of Interest."""
    def __init__(self, array, angle, roi_radius, dist_from_center, phantom_center):
        """
        Parameters
        ----------
        array : ndarray
            The 2D array representing the image the disk is on.
        angle : int, float
            The angle of the ROI in degrees from the phantom center.
        roi_radius : int, float
            The radius of the ROI from the center of the phantom.
        dist_from_center : int, float
            The distance of the ROI from the phantom center.
        phantom_center : tuple
            The location of the phantom center.
        """
        center = self._get_shifted_center(angle, dist_from_center, phantom_center)
        super().__init__(center_point=center, radius=roi_radius)
        self._array = array

    @property
    def std(self):
        """The standard deviation of the pixel values."""
        masked_img = self.circle_mask()
        return np.nanstd(masked_img)

    @staticmethod
    def _get_shifted_center(angle, dist_from_center, phantom_center):
        """The center of the ROI; corrects for phantom dislocation and roll."""
        y_shift = np.sin(np.deg2rad(angle)) * dist_from_center
        x_shift = np.cos(np.deg2rad(angle)) * dist_from_center
        return Point(phantom_center.x + x_shift, phantom_center.y + y_shift)

    @property
    def pixel_value(self):
        """The median pixel value of the ROI."""
        masked_img = self.circle_mask()
        return np.nanmedian(masked_img)

    @lru_cache(maxsize=1)
    def circle_mask(self):
        """Return a mask of the image, only showing the circular ROI."""
        # http://scikit-image.org/docs/dev/auto_examples/plot_camera_numpy.html
        masked_array = np.copy(self._array).astype(np.float)
        l_x, l_y = self._array.shape[0], self._array.shape[1]
        X, Y = np.ogrid[:l_x, :l_y]
        outer_disk_mask = (X - self.center.y) ** 2 + (Y - self.center.x) ** 2 > self.radius ** 2
        masked_array[outer_disk_mask] = np.NaN
        return masked_array


class LowContrastDiskROI(DiskROI):
    """A class for analyzing the low-contrast disks."""

    def __init__(self, array, angle, roi_radius, dist_from_center, phantom_center, contrast_threshold=None, background=None,
                 cnr_threshold=None):
        """
        Parameters
        ----------
        contrast_threshold : float, int
            The threshold for considering a bubble to be "seen".
        """
        super().__init__(array, angle, roi_radius, dist_from_center, phantom_center)
        self.contrast_threshold = contrast_threshold
        self.cnr_threshold = cnr_threshold
        self.background = background

    @property
    def contrast_to_noise(self):
        """The contrast to noise ratio of the bubble: (Signal - Background)/Stdev."""
        return abs(self.pixel_value - self.background) / self.std

    @property
    def cnr_constant(self):
        """The contrast-to-noise value times the bubble diameter."""
        return self.contrast_to_noise * self.diameter

    @property
    def passed_cnr_constant(self):
        """Boolean specifying if ROI pixel value was within tolerance of the nominal value."""
        return self.cnr_constant > self.cnr_threshold

    @property
    def plot_color_cnr(self):
        """Return one of two colors depending on if ROI passed."""
        return 'blue' if self.passed_cnr_constant else 'red'


class RectangleROI(Rectangle):
    """Class that represents a rectangular ROI."""

    def __init__(self, array, width, height, angle, dist_from_center, phantom_center):
        y_shift = np.sin(np.deg2rad(angle)) * dist_from_center
        x_shift = np.cos(np.deg2rad(angle)) * dist_from_center
        center = Point(phantom_center.x + x_shift, phantom_center.y + y_shift)
        super().__init__(width, height, center, as_int=True)
        self._array = array

    @property
    def pixel_array(self):
        """The pixel array within the ROI."""
        return self._array[self.bl_corner.x:self.tr_corner.x, self.bl_corner.y:self.tr_corner.y]