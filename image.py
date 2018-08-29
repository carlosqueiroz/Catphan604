"""This module holds classes for image loading and manipulation."""
from io import BytesIO
import os.path as osp
import os

import pydicom
from pydicom.errors import InvalidDicomError
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.pyplot import cm

from .geometry import Point

ARRAY = 'Array'
DICOM = 'DICOM'


def load(path, **kwargs):
    """Load a DICOM image, or numpy 2D array."""
    if isinstance(path, BaseImage):
        return path

    if _is_array(path):
        return ArrayImage(path, **kwargs)
    elif _is_dicom(path):
        return DicomImage(path, **kwargs)
    else:
        raise TypeError("The argument `{0}` was not found to be a valid DICOM file, Image file, or array".format(path))


def _is_dicom(path):
    """Whether the file is a readable DICOM file via pydicom."""
    try:
        ds = pydicom.read_file(path, stop_before_pixels=True, force=True)
        ds.SOPClassUID
        return True
    except:
        return False


def _is_array(obj):
    """Whether the object is a numpy array."""
    return isinstance(obj, np.ndarray)


class BaseImage:
    """Base class for the Image classes.

    Attributes
    ----------
    path : str
        The path to the image file.
    array : numpy.ndarray
        The actual image pixel array.
    """

    def __init__(self, path):
        if not osp.isfile(path):
            raise FileExistsError("File `{0}` does not exist. Verify the file path name.".format(path))
        self.path = path

    @property
    def center(self):
        """Return the center position of the image array as a Point."""
        x_center = self.shape[1] / 2
        y_center = self.shape[0] / 2
        return Point(x_center, y_center)

    def plot(self, ax=None, show=True, clear_fig=False, **kwargs):
        """Plot the image. """
        if ax is None:
            fig, ax = plt.subplots()
        if clear_fig:
            plt.clf()
        ax.imshow(self.array, cmap=cm.gray, **kwargs)
        if show:
            plt.show()
        return ax

    def filter(self, size=0.05, kind='median'):
        """Filter the profile.

        Parameters
        ----------
        size : int, float
            Size of the median filter to apply.
            If a float, the size is the ratio of the length. Must be in the range 0-1.
            E.g. if size=0.1 for a 1000-element array, the filter will be 100 elements.
            If an int, the filter is the size passed.
        kind : {'median', 'gaussian'}
            The kind of filter to apply. If gaussian, *size* is the sigma value.
        """
        if isinstance(size, float):
            if 0 < size < 1:
                size *= len(self.array)
                size = max(size, 1)
            else:
                raise TypeError("Float was passed but was not between 0 and 1")

        if kind == 'median':
            self.array = ndimage.median_filter(self.array, size=size)
        elif kind == 'gaussian':
            self.array = ndimage.gaussian_filter(self.array, sigma=size)

    def as_type(self, dtype):
        return self.array.astype(dtype)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        return self.array[item]


class DicomImage(BaseImage):
    """An image from a DICOM RTImage file.

    Attributes
    ----------
    metadata : pydicom Dataset
        The dataset of the file as returned by pydicom without pixel data.
    """

    def __init__(self, path, *, dtype=None):
        super().__init__(path)
        # read the file once to get just the DICOM metadata
        self.metadata = pydicom.read_file(path, force=True)
        self._original_dtype = self.metadata.pixel_array.dtype
        # read a second time to get pixel data
        if isinstance(path, BytesIO):
            path.seek(0)
        ds = pydicom.read_file(path, force=True)
        if dtype is not None:
            self.array = ds.pixel_array.astype(dtype)
        else:
            self.array = ds.pixel_array
        # convert values to proper HU: real_values = slope * raw + intercept
        if self.metadata.SOPClassUID.name == 'CT Image Storage':
            self.array = int(self.metadata.RescaleSlope)*self.array + int(self.metadata.RescaleIntercept)


class ArrayImage(BaseImage):
    """An image constructed solely from a numpy array."""

    def __init__(self, array, dtype=None):
        if dtype is not None:
            self.array = np.array(array, dtype=dtype)
        else:
            self.array = array


class DicomImageStack:
    """加载DICOM图像，维护DICOM图像栈(e.g. a CT dataset)"""
    def __init__(self, folder, dtype=None):
        self.images = []
        # 按检索顺序加载图像
        for pdir, sdir, files in os.walk(folder):
            for file in files:
                path = osp.join(pdir, file)
                if self.is_CT_slice(path):
                    img = DicomImage(path, dtype=dtype)
                    self.images.append(img)

        # 至少加载一辐图像
        if len(self.images) < 1:
            raise FileNotFoundError("No files were found in the specified location: {0}".format(folder))

        # 按图像物理位置排序
        self.images.sort(key=lambda x: x.metadata.ImagePositionPatient[-1])

    @staticmethod
    def is_CT_slice(file):
        """测试文件是否为DICOM CT存储文件"""
        try:
            ds = pydicom.read_file(file, force=True, stop_before_pixels=True)
            return ds.SOPClassUID.name == 'CT Image Storage'
        except (InvalidDicomError, AttributeError, MemoryError):
            return False

    def plot(self, slice=0):
        self.images[slice].plot()

    @property
    def metadata(self):
        """返回第一辐图像metadata，只有图像栈通用属性才能由其返回"""
        return self.images[0].metadata

    def __getitem__(self, item):
        return self.images[item]

    def __setitem__(self, key, value):
        self.images[key] = value

    def __len__(self):
        return len(self.images)
