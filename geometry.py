from itertools import zip_longest
import numpy as np
from collections import Iterable
from matplotlib.patches import Circle as mpl_Circle
from matplotlib.patches import Rectangle as mpl_Rectangle
import math


class Point:
    """A geometric point with x, y, and z coordinates/attributes."""
    _attr_list = ['x', 'y', 'z', 'idx', 'value']
    _coord_list = ['x', 'y', 'z']

    def __init__(self, x=0, y=0, z=0, idx=None, value=None, as_int=False):
        """
        Parameters
        ----------
        x : number-like, Point, iterable
            x-coordinate or iterable type containing all coordinates. If iterable, values are assumed to be in order: (x,y,z).
        y : number-like, optional
            y-coordinate
        idx : int, optional
            Index of point. Useful for sequential coordinates; e.g. a point on a circle profile is sometimes easier to describe
            in terms of its index rather than x,y coords.
        value : number-like, optional
            value at point location (e.g. pixel value of an image)
        as_int : boolean
            If True, coordinates are converted to integers.
        """
        if isinstance(x, Point):
            for attr in self._attr_list:
                item = getattr(x, attr, None)
                setattr(self, attr, item)
        elif isinstance(x, Iterable):
            for attr, item in zip_longest(self._attr_list, x, fillvalue=0):
                setattr(self, attr, item)
        else:
            self.x = x
            self.y = y
            self.z = z
            self.idx = idx
            self.value = value

        if as_int:
            self.x = int(round(self.x))
            self.y = int(round(self.y))
            self.z = int(round(self.z))

    def distance_to(self, thing):
        """Calculate the distance to the given point.

        Parameters
        ----------
        thing : Circle, Point, 2 element iterable
            The other thing to calculate distance to.
        """
        if isinstance(thing, Circle):
            return abs(np.sqrt((self.x - thing.center.x) ** 2 + (self.y - thing.center.y) ** 2) - thing.radius)
        p = Point(thing)
        return math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2 + (self.z - p.z) ** 2)


class Line:
    """A line that is represented by two points or by m*x+b.

    Notes
    -----
    Calculations of slope, etc are from here:
    http://en.wikipedia.org/wiki/Linear_equation
    and here:
    http://www.mathsisfun.com/algebra/line-equation-2points.html
    """
    def __init__(self, point1, point2):
        """
        Parameters
        ----------
        point1 : Point, optional
            One point of the line
        point2 : Point, optional
            Second point along the line.
        """
        self.point1 = Point(point1)
        self.point2 = Point(point2)

    @property
    def length(self):
        """Return length of the line, if finite."""
        return self.point1.distance_to(self.point2)

    def plot2axes(self, axes, width=1, color='w'):
        """Plot the line to an axes.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            An MPL axes to plot to.
        color : str
            The color of the line.
        """
        axes.plot((self.point1.x, self.point2.x), (self.point1.y, self.point2.y), linewidth=width, color=color)


class Circle:
    """A geometric circle with center Point, radius, and diameter."""

    def __init__(self, center_point=None, radius=None):
        """
        Parameters
        ----------
        center_point : Point, optional
            Center point of the wobble circle.
        radius : float, optional
            Radius of the wobble circle.
        """
        if center_point is None:
            center_point = Point()
        elif isinstance(center_point, Point) or isinstance(center_point, Iterable):
            center_point = Point(center_point)
        else:
            raise TypeError("Circle center must be of type Point or iterable")

        self.center = center_point
        self.radius = radius

    @property
    def diameter(self):
        """Get the diameter of the circle."""
        return self.radius * 2

    def plot2axes(self, axes, edgecolor='black', fill=False):
        """Plot the Circle on the axes.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            An MPL axes to plot to.
        edgecolor : str
            The color of the circle.
        fill : bool
            Whether to fill the circle with color or leave hollow.
        """
        axes.add_patch(mpl_Circle((self.center.x, self.center.y), edgecolor=edgecolor, radius=self.radius, fill=fill))


class Rectangle:
    """A rectangle with width, height, center Point, top-left corner Point, and bottom-left corner Point."""

    def __init__(self, width, height, center, as_int=False):
        """
        Parameters
        ----------
        width : number
            Width of the rectangle.
        height : number
            Height of the rectangle.
        center : Point, iterable, optional
            Center point of rectangle.
        as_int : bool
            If False (default), inputs are left as-is. If True, all inputs are converted to integers.
        """
        if as_int:
            self.width = int(np.round(width))
            self.height = int(np.round(height))
        else:
            self.width = width
            self.height = height
        self._as_int = as_int
        self.center = Point(center, as_int=as_int)

    @property
    def bl_corner(self):
        """The location of the bottom left corner."""
        return Point(self.center.x - self.width / 2, self.center.y - self.height / 2, as_int=self._as_int)

    def plot2axes(self, axes, edgecolor='black', angle=0.0, fill=False, alpha=1, facecolor='g'):
        """Plot the Rectangle to the axes.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            An MPL axes to plot to.
        edgecolor : str
            The color of the circle.
        angle : float
            Angle of the rectangle.
        fill : bool
            Whether to fill the rectangle with color or leave hollow.
        """
        axes.add_patch(mpl_Rectangle((self.bl_corner.x, self.bl_corner.y),
                                     width=self.width,
                                     height=self.height,
                                     angle=angle,
                                     edgecolor=edgecolor,
                                     alpha=alpha,
                                     facecolor=facecolor,
                                     fill=fill))

    @property
    def br_corner(self):
        """The location of the bottom right corner."""
        return Point(self.center.x + self.width / 2, self.center.y - self.height / 2, as_int=self._as_int)

    @property
    def bl_corner(self):
        """The location of the bottom left corner."""
        return Point(self.center.x - self.width / 2, self.center.y - self.height / 2, as_int=self._as_int)

    @property
    def tl_corner(self):
        """The location of the top left corner."""
        return Point(self.center.x - self.width / 2, self.center.y + self.height / 2, as_int=self._as_int)

    @property
    def tr_corner(self):
        """The location of the top right corner."""
        return Point(self.center.x + self.width / 2, self.center.y + self.height / 2, as_int=self._as_int)
