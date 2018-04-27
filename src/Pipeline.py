# Pipeline.py
# Author: Marcus D. Bloice <https://github.com/mdbloice>
# Licensed under the terms of the MIT Licence.
"""
The Pipeline module is the user facing API for the Augmentor package. It
contains the :class:`~Augmentor.Pipeline.Pipeline` class which is used to
create pipeline objects, which can be used to build an augmentation pipeline
by adding operations to the pipeline object.

For a good overview of how to use Augmentor, along with code samples and
example images, can be seen in the :ref:`mainfeatures` section.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import *

from .Operations import *
from .ImageUtilities import scan_directory, scan, AugmentorImage

import os
import sys
import random
import uuid
import warnings
import numbers
import numpy as np

from tqdm import tqdm
from PIL import Image


class Pipeline(object):
    """
    The Pipeline class handles the creation of augmentation pipelines
    and the generation of augmented data by applying operations to
    this pipeline.
    """

    # Some class variables we use often
    _probability_error_text = "The probability argument must be between 0 and 1."
    _threshold_error_text = "The value of threshold must be between 0 and 255."
    _valid_formats = ["PNG", "BMP", "GIF", "JPEG"]
    _legal_filters = ["NEAREST", "BICUBIC", "ANTIALIAS", "BILINEAR"]

    def __init__(self, source_directory=None, output_directory="output", save_format=None):
        """
        Create a new Pipeline object pointing to a directory containing your
        original image dataset.

        Create a new Pipeline object, using the :attr:`source_directory`
        parameter as a source directory where your original images are
        stored. This folder will be scanned, and any valid file files
        will be collected and used as the original dataset that should
        be augmented. The scan will find any image files with the extensions
        JPEG/JPG, PNG, and GIF (case insensitive).

        :param source_directory: A directory on your filesystem where your
         original images are stored.
        :param output_directory: Specifies where augmented images should be
         saved to the disk. Default is the directory **output** relative to
         the path where the original image set was specified. If it does not
         exist it will be created.
        :param save_format: The file format to use when saving newly created,
         augmented images. Default is JPEG. Legal options are BMP, PNG, and
         GIF.
        :return: A :class:`Pipeline` object.
        """
        random.seed()

        # TODO: Allow a single image to be added when initialising.
        # Initialise some variables for the Pipeline object.
        self.image_counter = 0
        self.augmentor_images = []
        self.distinct_dimensions = set()
        self.distinct_formats = set()
        self.save_format = save_format
        self.operations = []
        self.class_labels = []
        self.process_ground_truth_images = False

        # Now we populate some fields, which we may need to do again later if another
        # directory is added, so we place it all in a function of its own.
        if source_directory is not None:
            self._populate(source_directory=source_directory,
                           output_directory=output_directory,
                           ground_truth_directory=None,
                           ground_truth_output_directory=output_directory)

    def _populate(self, source_directory, output_directory, ground_truth_directory, ground_truth_output_directory):
        """
        Private method for populating member variables with AugmentorImage
        objects for each of the images found in the source directory
        specified by the user. It also populates a number of fields such as
        the :attr:`output_directory` member variable, used later when saving
        images to disk.

        This method is used by :func:`__init__`.

        :param source_directory: The directory to scan for images.
        :param output_directory: The directory to set for saving files.
         Defaults to a directory named output relative to
         :attr:`source_directory`.
        :param ground_truth_directory: A directory containing ground truth
         files for the associated images in the :attr:`source_directory`
         directory.
        :param ground_truth_output_directory: A path to a directory to store
         the output of the operations on the ground truth data set.
        :type source_directory: String
        :type output_directory: String
        :type ground_truth_directory: String
        :type ground_truth_output_directory: String
        :return: None
        """

        # Check if the source directory for the original images to augment exists at all
        if not os.path.exists(source_directory):
            raise IOError("The source directory you specified does not exist.")

        # If a ground truth directory is being specified we will check here if the path exists at all.
        if ground_truth_directory:
            if not os.path.exists(ground_truth_directory):
                raise IOError("The ground truth source directory you specified does not exist.")

        # Get absolute path for output
        abs_output_directory = os.path.join(source_directory, output_directory)

        # Scan the directory that user supplied.
        self.augmentor_images, self.class_labels = scan(source_directory, abs_output_directory)

        # Make output directory/directories
        if len(set(self.class_labels)) <= 1:  # Fixed bad bug by adding set() function here.
            if not os.path.exists(abs_output_directory):
                try:
                    os.makedirs(abs_output_directory)
                except IOError:
                    print("Insufficient rights to read or write output directory (%s)" % abs_output_directory)
        else:
            for class_label in self.class_labels:
                if not os.path.exists(os.path.join(abs_output_directory, str(class_label[0]))):
                    try:
                        os.makedirs(os.path.join(abs_output_directory, str(class_label[0])))
                    except IOError:
                        print("Insufficient rights to read or write output directory (%s)" % abs_output_directory)

        # Check the images, read their dimensions, and remove them if they cannot be read
        # TODO: Do not throw an error here, just remove the image and continue.
        for augmentor_image in self.augmentor_images:
            try:
                with Image.open(augmentor_image.image_path) as opened_image:
                    self.distinct_dimensions.add(opened_image.size)
                    self.distinct_formats.add(opened_image.format)
            except IOError as e:
                print("There is a problem with image %s in your source directory: %s" % (augmentor_image.image_path, e.message))
                self.augmentor_images.remove(augmentor_image)

        sys.stdout.write("Initialised with %s image(s) found.\n" % len(self.augmentor_images))
        sys.stdout.write("Output directory set to %s." % abs_output_directory)

    def _execute(self, augmentor_image, save_to_disk=True):
        """
        Private method. Used to pass an image through the current pipeline,
        and return the augmented image.

        The returned image can then either be saved to disk or simply passed
        back to the user. Currently this is fixed to True, as Augmentor
        has only been implemented to save to disk at present.

        :param augmentor_image: The image to pass through the pipeline.
        :param save_to_disk: Whether to save the image to disk. Currently
         fixed to true.
        :type augmentor_image: :class:`ImageUtilities.AugmentorImage`
        :type save_to_disk: Boolean
        :return: The augmented image.
        """
        # self.image_counter += 1  # TODO: See if I can remove this...

        images = []

        if augmentor_image.image_path is not None:
            images.append(Image.open(augmentor_image.image_path))

        if augmentor_image.ground_truth is not None:
            if isinstance(augmentor_image.ground_truth, list):
                for image in augmentor_image.ground_truth:
                    images.append(Image.open(image))
            else:
                images.append(Image.open(augmentor_image.ground_truth))

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images = operation.perform_operation(images)

        if save_to_disk:
            file_name = str(uuid.uuid4())
            try:
                # TODO: Add a 'coerce' parameter to force conversion to RGB for PNGA->JPEG saving.
                # if image.mode != "RGB":
                #     image = image.convert("RGB")
                for i in range(len(images)):
                    if i == 0:
                        save_name = augmentor_image.class_label + "_original_" + file_name \
                                    + "." + (self.save_format if self.save_format else augmentor_image.file_format)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
                    else:
                        save_name = "_groundtruth_(" + str(i) + ")_" + augmentor_image.class_label + "_" + file_name \
                                    + "." + (self.save_format if self.save_format else augmentor_image.file_format)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
            except IOError as e:
                print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
                print("You can change the save format using the set_save_format(save_format) function.")
                print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")

        # Currently we return only the first image if it is a list
        # for the generator functions.  This will be fixed in a future
        # version.
        return images[0]

    def _execute_with_array(self, image):
        """
        Private method used to execute a pipeline on array or matrix data.
        :param image: The image to pass through the pipeline.
        :type image: Array like object.
        :return: The augmented image.
        """

        pil_image = [Image.fromarray(image)]

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                pil_image = operation.perform_operation(pil_image)

        numpy_array = np.asarray(pil_image[0])

        return numpy_array

    def set_save_format(self, save_format):
        """
        Set the save format for the pipeline. Pass the value
        :attr:`save_format="auto"` to allow Augmentor to choose
        the correct save format based on each individual image's
        file extension.

        If :attr:`save_format` is set to, for example,
        :attr:`save_format="JPEG"` or :attr:`save_format="JPG"`,
        Augmentor will attempt to save the files using the
        JPEG format, which may result in errors if the file cannot
        be saved in this format, such as PNG images with an alpha
        channel.

        :param save_format: The save format to save the images
         when writing to disk.
        :return: None
        """

        if save_format == "auto":
            self.save_format = None
        else:
            self.save_format = save_format

    def sample(self, n):
        """
        Generate :attr:`n` number of samples from the current pipeline.

        This function samples from the pipeline, using the original images
        defined during instantiation. All images generated by the pipeline
        are by default stored in an ``output`` directory, relative to the
        path defined during the pipeline's instantiation.

        :param n: The number of new samples to produce.
        :type n: Integer
        :return: None
        """
        if len(self.augmentor_images) == 0:
            raise IndexError("There are no images in the pipeline. "
                             "Add a directory using add_directory(), "
                             "pointing it to a directory containing images.")

        if len(self.operations) == 0:
            raise IndexError("There are no operations associated with this pipeline.")

        sample_count = 1

        progress_bar = tqdm(total=n, desc="Executing Pipeline", unit=' Samples', leave=False)
        while sample_count <= n:
            for augmentor_image in self.augmentor_images:
                if sample_count <= n:
                    self._execute(augmentor_image)
                    file_name_to_print = os.path.basename(augmentor_image.image_path)
                    # This is just to shorten very long file names which obscure the progress bar.
                    if len(file_name_to_print) >= 30:
                        file_name_to_print = file_name_to_print[0:10] + "..." + \
                                             file_name_to_print[-10: len(file_name_to_print)]
                    progress_bar.set_description("Processing %s" % file_name_to_print)
                    progress_bar.update(1)
                sample_count += 1
        progress_bar.close()

    def sample_with_array(self, image_array, save_to_disk=False):
        """
        Generate images using a single image in array-like format.

        .. seealso::
         See :func:`keras_image_generator_without_replacement()` for

        :param image_array: The image to pass through the pipeline.
        :param save_to_disk: Whether to save to disk or not (default).
        :return:
        """
        a = AugmentorImage(image_path=None, output_directory=None)
        a.image_PIL = Image.fromarray(image_array)

        return self._execute(a, save_to_disk)

    def add_operation(self, operation):
        """
        Add an operation directly to the pipeline. Can be used to add custom
        operations to a pipeline.

        To add custom operations to a pipeline, subclass from the
        Operation abstract base class, overload its methods, and insert the
        new object into the pipeline using this method.

         .. seealso:: The :class:`.Operation` class.

        :param operation: An object of the operation you wish to add to the
         pipeline. Will accept custom operations written at run-time.
        :type operation: Operation
        :return: None
        """
        if isinstance(operation, Operation):
            self.operations.append(operation)
        else:
            raise TypeError("Must be of type Operation to be added to the pipeline.")

    def remove_operation(self, operation_index=-1):
        """
        Remove the operation specified by :attr:`operation_index`, if
        supplied, otherwise it will remove the latest operation added to the
        pipeline.

         .. seealso:: Use the :func:`status` function to find an operation's
          index.

        :param operation_index: The index of the operation to remove.
        :type operation_index: Integer
        :return: The removed operation. You can reinsert this at end of the
         pipeline using :func:`add_operation` if required.
        """

        # Python's own List exceptions can handle erroneous user input.
        self.operations.pop(operation_index)

    def random_distortion(self, probability, grid_width, grid_height, magnitude):
        """
        Performs a random, elastic distortion on an image.

        This function performs a randomised, elastic distortion controlled
        by the parameters specified. The grid width and height controls how
        fine the distortions are. Smaller sizes will result in larger, more
        pronounced, and less granular distortions. Larger numbers will result
        in finer, more granular distortions. The magnitude of the distortions
        can be controlled using magnitude. This can be random or fixed.

        *Good* values for parameters are between 2 and 10 for the grid
        width and height, with a magnitude of between 1 and 10. Using values
        outside of these approximate ranges may result in unpredictable
        behaviour.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param grid_width: The number of rectangles in the grid's horizontal
         axis.
        :param grid_height: The number of rectangles in the grid's vertical
         axis.
        :param magnitude: The magnitude of the distortions.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(Distort(probability=probability, grid_width=grid_width,
                                       grid_height=grid_height, magnitude=magnitude))

    def gaussian_distortion(self, probability, grid_width, grid_height, magnitude, corner, method, mex=0.5, mey=0.5,
                            sdx=0.05, sdy=0.05):
        """
        Performs a random, elastic gaussian distortion on an image.

        This function performs a randomised, elastic gaussian distortion controlled
        by the parameters specified. The grid width and height controls how
        fine the distortions are. Smaller sizes will result in larger, more
        pronounced, and less granular distortions. Larger numbers will result
        in finer, more granular distortions. The magnitude of the distortions
        can be controlled using magnitude. This can be random or fixed.

        *Good* values for parameters are between 2 and 10 for the grid
        width and height, with a magnitude of between 1 and 10. Using values
        outside of these approximate ranges may result in unpredictable
        behaviour.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param grid_width: The number of rectangles in the grid's horizontal
         axis.
        :param grid_height: The number of rectangles in the grid's vertical
         axis.
        :param magnitude: The magnitude of the distortions.
        :param corner: which corner of picture to distort.
         Possible values: "bell"(circular surface applied), "ul"(upper left),
         "ur"(upper right), "dl"(down left), "dr"(down right).
        :param method: possible values: "in"(apply max magnitude to the chosen
         corner), "out"(inverse of method in).
        :param mex: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param mey: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdx: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdy: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        :type corner: String
        :type method: String
        :type mex: Float
        :type mey: Float
        :type sdx: Float
        :type sdy: Float
        :return: None

        For values :attr:`mex`, :attr:`mey`, :attr:`sdx`, and :attr:`sdy` the
        surface is based on the normal distribution:

        .. math::

         e^{- \Big( \\frac{(x-\\text{mex})^2}{\\text{sdx}} + \\frac{(y-\\text{mey})^2}{\\text{sdy}} \Big) }
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(GaussianDistortion(probability=probability, grid_width=grid_width,
                                                  grid_height=grid_height,
                                                  magnitude=magnitude, corner=corner,
                                                  method=method,  mex=mex,
                                                  mey=mey, sdx=sdx, sdy=sdy))


    def skew_left_right(self, probability, magnitude=1):
        """
        Skew an image by tilting it left or right by a random amount. The
        magnitude of this skew can be set to a maximum using the
        magnitude parameter. This can be either a scalar representing the
        maximum tilt, or vector representing a range.

        To see examples of the various skews, see :ref:`perspectiveskewing`.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param magnitude: The maximum tilt, which must be value between 0.1
         and 1.0, where 1 represents a tilt of 45 degrees.
        :type probability: Float
        :type magnitude: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_operation(Skew(probability=probability, skew_type="TILT_LEFT_RIGHT", magnitude=magnitude))

    def skew_top_bottom(self, probability, magnitude=1):
        """
        Skew an image by tilting it forwards or backwards by a random amount.
        The magnitude of this skew can be set to a maximum using the
        magnitude parameter. This can be either a scalar representing the
        maximum tilt, or vector representing a range.

        To see examples of the various skews, see :ref:`perspectiveskewing`.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param magnitude: The maximum tilt, which must be value between 0.1
         and 1.0, where 1 represents a tilt of 45 degrees.
        :type probability: Float
        :type magnitude: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_operation(Skew(probability=probability,
                                    skew_type="TILT_TOP_BOTTOM",
                                    magnitude=magnitude))

    def skew_tilt(self, probability, magnitude=1):
        """
        Skew an image by tilting in a random direction, either forwards,
        backwards, left, or right, by a random amount. The magnitude of
        this skew can be set to a maximum using the magnitude parameter.
        This can be either a scalar representing the maximum tilt, or
        vector representing a range.

        To see examples of the various skews, see :ref:`perspectiveskewing`.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param magnitude: The maximum tilt, which must be value between 0.1
         and 1.0, where 1 represents a tilt of 45 degrees.
        :type probability: Float
        :type magnitude: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_operation(Skew(probability=probability,
                                    skew_type="TILT",
                                    magnitude=magnitude))

    def skew_corner(self, probability, magnitude=1):
        """
        Skew an image towards one corner, randomly by a random magnitude.

        To see examples of the various skews, see :ref:`perspectiveskewing`.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param magnitude: The maximum skew, which must be value between 0.1
         and 1.0.
        :return:
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_operation(Skew(probability=probability,
                                    skew_type="CORNER",
                                    magnitude=magnitude))

    def skew(self, probability, magnitude=1):
        """
        Skew an image in a random direction, either left to right,
        top to bottom, or one of 8 corner directions.

        To see examples of all the skew types, see :ref:`perspectiveskewing`.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param magnitude: The maximum skew, which must be value between 0.1
         and 1.0.
        :type probability: Float
        :type magnitude: Float
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < magnitude <= 1:
            raise ValueError("The magnitude argument must be greater than 0 and less than or equal to 1.")
        else:
            self.add_operation(Skew(probability=probability,
                                    skew_type="RANDOM",
                                    magnitude=magnitude))

    def shear(self, probability, max_shear_left, max_shear_right):
        """
        Shear the image by a specified number of degrees.

        In practice, shear angles of more than 25 degrees can cause
        unpredictable behaviour. If you are observing images that are
        incorrectly rendered (e.g. they do not contain any information)
        then reduce the shear angles.

        :param probability: The probability that the operation is performed.
        :param max_shear_left: The max number of degrees to shear to the left.
         Cannot be larger than 25 degrees.
        :param max_shear_right: The max number of degrees to shear to the
         right. Cannot be larger than 25 degrees.
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not 0 < max_shear_left <= 25:
            raise ValueError("The max_shear_left argument must be between 0 and 25.")
        elif not 0 < max_shear_right <= 25:
            raise ValueError("The max_shear_right argument must be between 0 and 25.")
        else:
            self.add_operation(Shear(probability=probability,
                                     max_shear_left=max_shear_left,
                                     max_shear_right=max_shear_right))


    def ground_truth(self, ground_truth_directory):
        """
        Specifies a directory containing corresponding images that
        constitute respective ground truth images for the images
        in the current pipeline.

        This function will search the directory specified by
        :attr:`ground_truth_directory` and will associate each ground truth
        image with the images in the pipeline by file name.

        Therefore, an image titled ``cat321.jpg`` will match with the
        image ``cat321.jpg`` in the :attr:`ground_truth_directory`.
        The function respects each image's label, therefore the image
        named ``cat321.jpg`` with the label ``cat`` will match the image
        ``cat321.jpg`` in the subdirectory ``cat`` relative to
        :attr:`ground_truth_directory`.

        Typically used to specify a set of ground truth or gold standard
        images that should be augmented alongside the original images
        of a dataset, such as image masks or semantic segmentation ground
        truth images.

        A number of such data sets are openly available, see for example
        `https://arxiv.org/pdf/1704.06857.pdf <https://arxiv.org/pdf/1704.06857.pdf>`_
        (Garcia-Garcia et al., 2017).

        :param ground_truth_directory: A directory containing the
         ground truth images that correspond to the images in the
         current pipeline.
        :type ground_truth_directory: String
        :return: None.
        """

        num_of_ground_truth_images_added = 0

        # Progress bar
        progress_bar = tqdm(total=len(self.augmentor_images), desc="Processing", unit=' Images', leave=False)

        if len(self.class_labels) == 1:
            for augmentor_image_idx in range(len(self.augmentor_images)):
                ground_truth_image = os.path.join(ground_truth_directory,
                                                  self.augmentor_images[augmentor_image_idx].image_file_name)
                if os.path.isfile(ground_truth_image):
                    self.augmentor_images[augmentor_image_idx].ground_truth = ground_truth_image
                    num_of_ground_truth_images_added += 1
        else:
            for i in range(len(self.class_labels)):
                for augmentor_image_idx in range(len(self.augmentor_images)):
                    ground_truth_image = os.path.join(ground_truth_directory,
                                                      self.augmentor_images[augmentor_image_idx].class_label,
                                                      self.augmentor_images[augmentor_image_idx].image_file_name)
                    if os.path.isfile(ground_truth_image):
                        if self.augmentor_images[augmentor_image_idx].class_label == self.class_labels[i][0]:
                            # Check files are the same size. There may be a better way to do this.
                            original_image_dimensions = \
                                Image.open(self.augmentor_images[augmentor_image_idx].image_path).size
                            ground_image_dimensions = Image.open(ground_truth_image).size
                            if original_image_dimensions == ground_image_dimensions:
                                self.augmentor_images[augmentor_image_idx].ground_truth = ground_truth_image
                                num_of_ground_truth_images_added += 1
                                progress_bar.update(1)

        progress_bar.close()

        # May not be required after all, check later.
        if num_of_ground_truth_images_added != 0:
            self.process_ground_truth_images = True

        print("%s ground truth image(s) found." % num_of_ground_truth_images_added)

    def get_ground_truth_paths(self):
        """
        Returns a list of image and ground truth image path pairs. Used for
        verification purposes to ensure the ground truth images match to the
        images containing in the pipeline.

        :return: A list of tuples containing the image path and ground truth
         path pairs.
        """
        paths = []

        for augmentor_image in self.augmentor_images:
            print("Image path: %s\nGround truth path: %s\n---\n" % (augmentor_image.image_path, augmentor_image.ground_truth))
            paths.append((augmentor_image.image_path, augmentor_image.ground_truth))

        return paths
