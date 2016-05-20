"""A helper object to concatenate a number of re sampling objects and
streamline the re-sampling process."""
from __future__ import print_function
from __future__ import division


class Pipeline(object):
    """A helper object to concatenate a number of re sampling objects and
    streamline the re-sampling process.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----

    """

    def __init__(self, x, y):
        """
        :param x:
            Feature matrix.

        :param y:
            Target vectors.
        """

        self.x = x
        self.y = y

    def pipeline(self, list_methods):
        """
        :param list_methods:
            Pass the methods to be used in a list, in the order they will be
            used.

        :return:
            The re-sampled dataset.
        """

        # Initialize with the original dataset.
        x, y = self.x, self.y

        # Go through the list of methods and fit_transform each to the result
        # of the last.
        for met in list_methods:
            x, y = met.fit_transform(x, y)
            print(x.shape)

        # Return the re-sampled dataset.
        return x, y
