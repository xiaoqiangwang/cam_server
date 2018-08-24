import tempfile
from logging import getLogger

import math
import numba
import numpy
import scipy
import scipy.misc
import scipy.optimize
import scipy.stats

from matplotlib import cm

from cam_server import config

_logging = getLogger(__name__)


@numba.njit(parallel=True)
def remove_background(image, background, threshold):
    y = image.shape[0]
    x = image.shape[1]
    
    for i in numba.prange(y):
        for j in range(x):
            v = image[i,j]
            b = background[i,j]

            v -= b
            if v < threshold:
                v = 0

            image[i,j] = v


def subtract_background(image, background_image, threshold):
    # We do not want negative numbers int the image.
    if image.shape != background_image.shape:
        raise RuntimeError("Invalid background_image size %s compared to image %s" % (background_image.shape,
                                                                                      image.shape))

    numpy.subtract(image, background_image, out=image)
    image[image < int(threshold)] = 0


def get_region_of_interest(image, offset_x, size_x, offset_y, size_y):
    return image[offset_y:offset_y + size_y, offset_x:offset_x + size_x]


def apply_threshold(image, threshold=1):
    image[image < int(threshold)] = 0


@numba.njit(parallel=True)
def get_statistics(image):
    """Return the minimum/maximum, x/y profile"""
    y = image.shape[0]
    x = image.shape[1]

    yp = numpy.zeros(y)
    xp = numpy.zeros(x)

    min = 0
    max = 0
    total = 0

    for i in numba.prange(y):
        for j in range(x):
            v = image[i, j]

            yp[i] += v
            xp[j] += v
            total += v

            if v < min:
                min = v
            elif v > max:
                max = v
 
    return min, max, xp, yp, total


def find_index(axis, item):
    """ Find the index of the axis value that corresponds to the passed value/item"""

    ascending = axis[1] > axis[0]  # if true axis looks like this [0, 1, 2, 3, 4, 5]

    # Ascending order -> [6, 7, 8, 9]
    if ascending:
        # Item value 5 -> go to first section.
        if item < axis[0]:
            return 0

        # use 'right' side so that index is always one higher
        return numpy.searchsorted(axis, item, 'right') - 1

    # Descending order -> [9, 8, 7, 6]
    else:
        # Item value 5 -> go to last section.
        if item < axis[-1]:
            return len(axis) - 1

        # Negate the array and number to search from the right.
        return numpy.searchsorted(-axis, -item)

def get_good_region_profile(profile, threshold=0.3, gfscale=1.8):
    profile_min = profile.min()
    # The center of the good region is defined by the index of the max value of the profile
    index_maximum = profile.argmax()
    profile_max = profile[index_maximum]

    threshold_value = (profile_max - profile_min) * threshold + profile_min

    index_start = index_maximum
    index_end = index_maximum

    for i in range(index_maximum, 0, -1):
        if profile[i] < threshold_value:
            index_start = i
            break

    for i in range(index_maximum, profile.size):
        if profile[i] < threshold_value:
            index_end = i
            break

    # Extend the good region based on gfscale
    gf_extend = (index_end - index_start) * gfscale - (index_end - index_start)

    index_start -= gf_extend / 2
    index_end += gf_extend / 2

    index_start = max(index_start, 0)
    index_end = min(index_end, profile.size - 1)

    return int(index_start), int(index_end)  # Start and end index of the good region


def center_of_mass(profile, axis):
    sum = profile.sum()
    center_of_mass = numpy.dot(axis, profile) / sum
    rms = numpy.sqrt(numpy.abs(numpy.dot(axis**2, profile) / sum - center_of_mass ** 2))
    return center_of_mass, rms


@numba.vectorize([numba.float64(numba.float64,numba.float64,numba.float64,numba.float64,numba.float64)], nopython=True)
def _gauss_function(x, offset, amplitude, center, standard_deviation):
    return offset + amplitude * math.exp(-(x - center) ** 2 / (2 * standard_deviation ** 2))


def gauss_fit(profile, axis):
    if axis.shape[0] != profile.shape[0]:
        raise RuntimeError("Invalid axis passed %d %d" % (axis.shape[0], profile.shape[0]))

    offset = profile.min()  # Minimum is good estimation of offset
    amplitude = profile.max() - offset  # Max value is a good estimation of amplitude
    center = numpy.dot(axis, profile) / profile.sum() # Center of mass is a good estimation of center (mu)
    surface = numpy.trapz((profile - offset), x=axis) # Consider gaussian integral is amplitude * sigma * sqrt(2*pi)
    standard_deviation = surface / (amplitude * numpy.sqrt(2 * numpy.pi))

    try:
        optimal_parameter, _ = scipy.optimize.curve_fit(
                _gauss_function, axis, profile.astype("float32"),
                p0=[offset, amplitude, center, standard_deviation],
                jac=_gauss_deriv,
                col_deriv=1)
        offset, amplitude, center, standard_deviation = optimal_parameter
    except BaseException as e:
        _logging.exception("COULD NOT CONVERGE!")

    gauss_function = _gauss_function(axis, offset, amplitude, center, standard_deviation)

    return gauss_function, offset, amplitude, center, abs(standard_deviation)


@numba.njit
def _gauss_deriv(x, offset, amplitude, center, standard_deviation):
    fac = numpy.exp(-(x - center) ** 2 / (2 * standard_deviation ** 2))

    result = numpy.empty((4, x.size), dtype=x.dtype)
    result[0, :] = 1.0
    result[1, :] = fac
    result[2, :] = amplitude * fac * (x - center) / (standard_deviation**2)
    result[3, :] = amplitude * fac * ((x-center)**2) / (standard_deviation**3)

    return result


def slice_image(image, number_of_slices=1, vertical=False):
    """
    :param image:
    :param number_of_slices:
    :param vertical:            if vertical the axis to use is y, if not vertical the axis to use is x
    :return:
    """

    if vertical:
        image = image.T  # transpose

    slice_size = image.shape[0] / number_of_slices
    slices = numpy.empty((number_of_slices, image.shape[1]))

    for i in range(number_of_slices):
        slices[i] = image[i * slice_size:(i + 1) * slice_size, :].sum(0)

    return slices


def calculate_slices(axis, center, standard_deviation, scaling=2, number_of_slices=9):
    """ Calculate index list for slices based on the given axis """

    if number_of_slices % 2 == 0:
        raise ValueError("Number of slices must be odd.")

    size_slice = scaling * standard_deviation / number_of_slices

    index_center = find_index(axis, center)
    index_half_slice = find_index(axis, center + size_slice / 2)
    n_pixel_half_slice = abs(index_half_slice - index_center)

    if n_pixel_half_slice < 1:
        _logging.info('Calculated number of pixel of a slice size [%d] is less than 1 - default to 1',
                      n_pixel_half_slice)
        n_pixel_half_slice = 1

    n_pixel_slice = 2 * n_pixel_half_slice

    # Add middle slice - located half/half on center
    start_index = index_center - n_pixel_half_slice
    end_index = index_center + n_pixel_half_slice

    list_slices_indexes = []
    slice_length = None

    number_of_elements_axis = len(axis)

    if start_index >= 0 and end_index < number_of_elements_axis:

        list_slices_indexes.append(start_index)
        list_slices_indexes.append(end_index)

        # The slice length is the difference in axis value from the start to the end of the axis.
        slice_length = abs(axis[start_index] - axis[end_index])

        # We subtract 1 because we already added the middle slice.
        counter_slices = number_of_slices - 1

        # Calculate outer slices
        while counter_slices > 0:
            start_index -= n_pixel_slice
            end_index += n_pixel_slice
            if start_index < 0 or end_index >= number_of_elements_axis:
                _logging.info('Stopping slice calculation as they are out of range ...')
                # Start index cannot be smaller than 0 and end index cannot e larger than len(axis)
                break
            list_slices_indexes.insert(0, start_index)
            list_slices_indexes.append(end_index)

            counter_slices -= 2

    return list_slices_indexes, n_pixel_half_slice, slice_length


def get_x_slices_data(image, x_axis, y_axis, x_center, x_standard_deviation, scaling=2, number_of_slices=11):
    """
    Calculate slices and their statistics
    :return: <center [x,y]>, <standard deviation>, <intensity>
    """

    list_slices, n_pixel_half_slice, slice_length = calculate_slices(x_axis, x_center, x_standard_deviation, scaling,
                                                                     number_of_slices)

    slice_data = []

    for i in range(len(list_slices) - 1):
        if list_slices[i] < image.shape[-1] and list_slices[i + 1] < image.shape[-1]:
            # slices are within good region
            slice_n = image[:, list_slices[i]:list_slices[i + 1]]

            slice_y_profile = slice_n.sum(1)
            pixel_intensity = slice_n.sum()

            # Does x need to be the middle of slice? - currently it is
            center_x = x_axis[list_slices[i] + n_pixel_half_slice]

            gauss_function, offset, amplitude, center_y, standard_deviation = gauss_fit(slice_y_profile, y_axis)
            slice_data.append(([center_x, center_y], standard_deviation, pixel_intensity))
        else:
            _logging.info('Drop slice')

    return slice_data, slice_length


def get_y_slices_data(image, x_axis, y_axis, y_center, y_standard_deviation, scaling=2, number_of_slices=11):
    """
    Calculate slices and their statistics
    :return: <center [x,y]>, <standard deviation>, <intensity>
    """

    list_slices, n_pixel_half_slice, slice_length = calculate_slices(y_axis, y_center, y_standard_deviation, scaling,
                                                                     number_of_slices)

    slice_data = []

    for i in range(len(list_slices) - 1):
        if list_slices[i] < image.shape[0] and list_slices[i + 1] < image.shape[0]:
            # slices are within good region
            slice_n = image[list_slices[i]:list_slices[i + 1], :]

            slice_x_profile = slice_n.sum(0)
            pixel_intensity = slice_n.sum()

            gauss_function, offset, amplitude, center_x, standard_deviation = gauss_fit(slice_x_profile, x_axis)

            # Does x need to be the middle of slice? - currently it is
            slice_data.append(([center_x, y_axis[list_slices[i] + n_pixel_half_slice]], standard_deviation,
                               pixel_intensity))
        else:
            _logging.info('Drop slice')

    return slice_data, slice_length


def linear_fit(x, y):
    slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
    return slope, intercept


def _quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c


def quadratic_fit(x, y):
    optimal_parameter, covariance = scipy.optimize.curve_fit(_quadratic_function, x, y)

    return optimal_parameter


def get_png_from_image(image_raw_bytes, scale=None, min_value=None, max_value=None, colormap_name=None):
    """
    Generate an image from the provided camera.
    :param image_raw_bytes: Image bytes to turn into PNG
    :param scale: Scale the image.
    :param min_value: Min cutoff value.
    :param max_value: Max cutoff value.
    :param colormap_name: Colormap to use. See http://matplotlib.org/examples/color/colormaps_reference.html
    :return: PNG image.
    """

    image_raw_bytes = image_raw_bytes.astype("float64")

    if scale:
        shape_0 = int(image_raw_bytes.shape[0] * scale)
        shape_1 = int(image_raw_bytes.shape[1] * scale)
        sh = shape_0, image_raw_bytes.shape[0] // shape_0, shape_1, image_raw_bytes.shape[1] // shape_1
        image_raw_bytes = image_raw_bytes.reshape(sh).mean(-1).mean(1)

    if min_value:
        image_raw_bytes -= min_value
        image_raw_bytes[image_raw_bytes < 0] = 0

    if max_value:
        image_raw_bytes[image_raw_bytes > max_value] = max_value

    try:
        colormap_name = colormap_name or config.DEFAULT_CAMERA_IMAGE_COLORMAP
        # Available colormaps http://matplotlib.org/examples/color/colormaps_reference.html
        colormap = getattr(cm, colormap_name)

        # http://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
        # normalize image to range 0.0-1.0
        image_raw_bytes *= 1.0 / image_raw_bytes.max()

        image = numpy.uint8(colormap(image_raw_bytes) * 255)
    except:
        raise ValueError("Unable to apply colormap '%s'. "
                         "See http://matplotlib.org/examples/color/colormaps_reference.html for available colormaps." %
                         colormap_name)

    n_image = scipy.misc.toimage(image)

    tmp_file = tempfile.TemporaryFile()

    # https://github.com/python-pillow/Pillow/issues/1211
    # We do not use any compression for speed reasons
    # n_image.save('your_file.png', compress_level=0)
    n_image.save(tmp_file, 'png', compress_level=0)
    # n_image.save(tmp_file, 'jpeg', compress_level=0)  # jpeg seems to be faster

    tmp_file.seek(0)
    content = tmp_file.read()
    tmp_file.close()

    return content
