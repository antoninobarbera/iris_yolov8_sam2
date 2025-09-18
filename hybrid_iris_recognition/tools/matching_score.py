import numpy as np
from tools.utils import to_polar, normalize_r, is_within_one_std


class matching_score_class():
    '''
    Class to compute the matching score between two irises based on keypoints.

    Attributes:
    - size: The number of points processed.
    - r_diff_list: List to store the radial differences between points.
    - theta_diff_list: List to store the angular differences between points.
    - config: Configuration parameters for matching.
    - centre_1, pupil_radius_1, iris_radius_1: The attributes (center, pupil and iris radii) of the first iris.
    - centre_2, pupil_radius_2, iris_radius_2: The attributes (center, pupil and iris radii) of the second iris.
    '''
    __slots__ = ['size', 'r_diff_list', 'theta_diff_list', 'config',
                'centre_1', 'pupil_radius_1', 'iris_radius_1',
                'centre_2', 'pupil_radius_2', 'iris_radius_2',]


    def __init__(self, iris_1, iris_2, config):
        '''
        Initializes the matching score class with two iris objects and a configuration.

        Args:
        - iris_1: The first iris object to compare.
        - iris_2: The second iris object to compare.
        - config: The configuration parameters for the matching process.
        '''
        self.size = 0
        self.r_diff_list = []
        self.theta_diff_list = []
        self.config = config.matching
        self.centre_1, self.pupil_radius_1, self.iris_radius_1 = iris_1.get_attributes()
        self.centre_2, self.pupil_radius_2, self.iris_radius_2 = iris_2.get_attributes()
         
         
    def __add__(self, p_1, p_2):
        '''
        Adds a new point pair to the matching score comparison.

        Args:
        - p_1: The first point in polar coordinates.
        - p_2: The second point in polar coordinates.
        '''
        r_1, theta_1 = to_polar(p_1, pole=(self.centre_1))
        r_2, theta_2 = to_polar(p_2, pole=(self.centre_2))
        r_1 = normalize_r(r_1, self.pupil_radius_1, self.iris_radius_1)
        r_2 = normalize_r(r_2, self.pupil_radius_2, self.iris_radius_2)
        r_diff = r_1 - r_2
        theta_diff = theta_1 - theta_2
        self.r_diff_list.append(r_diff)
        self.theta_diff_list.append(theta_diff)
        self.size += 1


    def __call__(self):
        '''
        Calculates the total matching score by evaluating the points collected.

        Returns:
        - int: The total score, calculated by counting valid points.
        '''
        if self.size == 0: 
           return 0
        points = 0
        median_r_diff = np.median(np.array(self.r_diff_list))
        median_theta_diff = np.median(np.array(self.theta_diff_list))
        for i in range (0, self.size):
            r_diff_is_valid = is_within_one_std(self.r_diff_list[i], median_r_diff, self.config.stdev_r_diff)
            theta_diff_is_valid = is_within_one_std(self.theta_diff_list[i], median_theta_diff, self.config.stdev_theta_diff)
            if r_diff_is_valid and theta_diff_is_valid: 
                points += 1
        return points
    