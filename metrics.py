# -*- coding:utf-8 -*-
# Author: K.D. Xiu
# Create Date: 10/16/2023
# Description:
#  This file contains the metrics used in the project, which will also be used in my future password security related research.
#  The metrics include:
#   1. Distance-like functions
#   2. Edit-distance-like functions
#   3. Entropy-like functions
#   4. Frequency-like functions
#   5. Alignment-like functions
#   6. Token-based distance functions
#  The functions in this file are all based on the password string.


import math
import numpy as np
import utils as U

class Metric:
    """ The abstract class of all metrics """
    def __init__(self, name):
        self.name = name
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class MahattanDistance(Metric):
    def __init__(self, name = "Mahattan"):
        super().__init__(name)
    
    def ord_dist(self, s1, s2):
        """ Mahattan distance """
        return sum([abs(ord(s1[i]) - ord(s2[i])) for i in range(len(s1))])
    
    def __call__(self, s1, s2):
        """ Mahattan distance """
        vec1, vec2 = U.calc_vec1D(s1, s2)
        dist = np.sum(np.abs(np.array(vec1) - np.array(vec2)))
        dist = round(1 / (1 + dist), 6)
        return dist


class EuclideanDistance(Metric):
    def __init__(self, name = "EuclideanDistance"):
        super().__init__(name)
    
    def ord_dist(self, s1, s2):
        """ Euclidean distance """
        return math.sqrt(sum([(ord(s1[i]) - ord(s2[i])) ** 2 for i in range(len(s1))]))
    
    def __call__(self, s1, s2):
        """ Euclidean distance """
        vec1, vec2 = U.calc_vec1D(s1, s2)
        dist = np.sqrt(np.sum(np.square(np.array(vec1) - np.array(vec2))))
        dist = round(1 / (1 + dist), 6)
        return dist


class ChebyshevDistance(Metric):
    def __init__(self, name = "ChebyshevDistance"):
        super().__init__(name)
    
    def ord_dist(self, s1, s2):
        """ Chebyshev distance """
        return max([abs(ord(s1[i]) - ord(s2[i])) for i in range(len(s1))])
    
    def __call__(self, s1, s2):
        """ Chebyshev distance """
        vec1, vec2 = U.calc_vec1D(s1, s2)
        dist = np.max(np.abs(np.array(vec1) - np.array(vec2)))
        dist = round(1 / (1 + dist), 6)
        return dist


class MinkowskiDistance(Metric):
    def __init__(self, name = "MinkowskiDistance"):
        super().__init__(name)
    
    def ord_dist(self, s1, s2, p):
        """ Minkowski distance """
        return sum([abs(ord(s1[i]) - ord(s2[i])) ** p for i in range(len(s1))]) ** (1 / p)
    
    def __call__(self, s1, s2, p = 3):
        """ Minkowski distance """
        vec1, vec2 = U.calc_vec1D(s1, s2)
        dist = np.sum(np.power(np.abs(np.array(vec1) - np.array(vec2)), p)) ** (1 / p)
        dist = round(1 / (1 + dist), 6)
        return dist


class CanberraDistance(Metric):
    def __init__(self, name = "CanberraDistance"):
        super().__init__(name)
    
    def ord_dist(self, s1, s2):
        """ Canberra distance """
        return sum([abs(ord(s1[i]) - ord(s2[i])) / (abs(ord(s1[i])) + abs(ord(s2[i]))) for i in range(len(s1))])
    
    def __call__(self, s1, s2):
        """ Canberra distance """
        vec1, vec2 = U.calc_vec1D(s1, s2)
        dist = np.sum(np.abs(np.array(vec1) - np.array(vec2)) / (np.abs(np.array(vec1)) + np.abs(np.array(vec2))))
        dist = round(1 / (1 + dist), 6)
        return dist


class HammingDistance(Metric):
    def __init__(self, name = "HammingDistance"):
        super().__init__(name)
    
    def ord_dist(self, s1, s2):
        """ Hamming distance """
        return sum([ord(s1[i]) != ord(s2[i]) for i in range(len(s1))])
    
    def __call__(self, s1, s2):
        """ Hamming distance """
        vec1, vec2 = U.calc_vec1D(s1, s2)
        dist = np.sum(np.array(vec1) != np.array(vec2))
        dist = round(1 / (1 + dist), 6)
        return dist


class JaccardDistance(Metric):
    def __init__(self, name = "JaccardDistance"):
        super().__init__(name)
    
    def ord_dist(self, s1, s2):
        """ Jaccard distance """
        return sum([ord(s1[i]) == ord(s2[i]) for i in range(len(s1))]) / sum([ord(s1[i]) != ord(s2[i]) for i in range(len(s1))])
    
    def __call__(self, s1, s2):
        """ Jaccard distance """
        vec1, vec2 = U.calc_vec1D(s1, s2)
        dist = np.sum(np.array(vec1) == np.array(vec2)) / np.sum(np.array(vec1) != np.array(vec2))
        dist = round(1 / (1 + dist), 6)
        return dist


class CosineSimilarity(Metric):
    def __init__(self, name = "CosineSimilarity"):
        super().__init__(name)
    
    def ord_dist(self, s1, s2):
        """ Cosine similarity """
        return sum([ord(s1[i]) * ord(s2[i]) for i in range(len(s1))]) / (math.sqrt(sum([ord(s1[i]) ** 2 for i in range(len(s1))])) * math.sqrt(sum([ord(s2[i]) ** 2 for i in range(len(s2))])))
    
    def __call__(self, s1, s2):
        """ Cosine similarity """
        vec1, vec2 = U.calc_vec1D(s1, s2)
        dist = np.sum(np.array(vec1) * np.array(vec2)) / (np.sqrt(np.sum(np.square(np.array(vec1)))) * np.sqrt(np.sum(np.square(np.array(vec2)))))
        dist = round(1 / (1 + dist), 6)
        return dist


class LevenshteinEditDistance(Metric):
    def __init__(self, name = "LevenshteinEditDistance"):
        super().__init__(name)
    
    def ord_dist(self, s1, s2):
        """ Levenshtein edit distance """
        dp = [[0] * len(s1 + 1) for _ in range(len(s2) + 1)]
        for i in range(len(s1) + 1):
            dp[0][i] = i
        for i in range(len(s2) + 1):
            dp[i][0] = i
        for i in range(1, len(s2) + 1):
            for j in range(1, len(s1) + 1):
                if s1[j] == s2[i]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min([dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]]) + 1
        return dp[-1][-1]
    
    def __call__(self, s1, s2):
        """ Levenshtein edit distance """
        dp = [[0] * (len(s1) + 1) for _ in range(len(s2) + 1)]
        for i in range(len(s1) + 1):
            dp[0][i] = i
        for i in range(len(s2) + 1):
            dp[i][0] = i
        for i in range(1, len(s2) + 1):
            for j in range(1, len(s1) + 1):
                if s1[j] == s2[i]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min([dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]]) + 1
        dist = round(1 / (1 + dp[-1][-1]), 6)
        return dist


class DamerauLevenshteinEditDistance(Metric):
    def __init__(self, name = "DamerauLevenshteinEditDistance"):
        super().__init__(name)
    
    def ord_dist(self, s1, s2):
        dp = [[0] * (len(s1) + 1) for _ in range(len(s2) + 1)]
        for i in range(len(s1) + 1):
            dp[0][i] = i
        for i in range(len(s2) + 1):
            dp[i][0] = i
        
        for i in range(1, len(s2) + 1):
            for j in range(1, len(s1) + 1):
                if s1[j] == s2[i]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min([dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]]) + 1
                    if i > 1 and j > 1 and s1[j-1:j+1] == s2[i-1:i+1:-1]:
                        dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
        return dp[-1][-1]
    
    def __call__(self, s1, s2):
        dp = [[0] * (len(s1) + 1) for _ in range(len(s2) + 1)]
        for i in range(len(s1) + 1):
            dp[0][i] = i
        for i in range(len(s2) + 1):
            dp[i][0] = i
        
        for i in range(1, len(s2) + 1):
            for j in range(1, len(s1) + 1):
                if s1[j] == s2[i]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min([dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]]) + 1
                    if i > 1 and j > 1 and s1[j-1:j+1] == s2[i-1:i+1:-1]:
                        dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
        dist = round(1 / (1 + dp[-1][-1]), 6)
        return dist


#TODO: implement the following metrics
class SmithWaterman(Metric):
    """ The wrapper of Smith-Waterman algorithm """
    def __init__(self, name = "SmithWaterman"):
        super().__init__(name)
    
    
    def __call__(self, s1, s2):
        """ Smith-Waterman algorithm """
        return self.SmithWaterman(s1, s2)
    
    def SmithWaterman(self, s1, s2):
        """ Smith-Waterman algorithm """
        pass


class NeedlemanWunsch(Metric):
    def __init__(self, name = "NeedlemanWunsch"):
        super().__init__(name)
    
    def __call__(self, s1, s2):
        """ Needleman-Wunsch algorithm """
        return self.NeedlemanWunsch(s1, s2)

    def NeedlemanWunsch(self, s1, s2):
        """ Needleman-Wunsch algorithm """
        pass


class Dice(Metric):
    def __init__(self, name = "Dice"):
        super().__init__(name)
    
    def __call_(self, s1, s2):
        """ Dice coefficient """
        return self.Dice(s1, s2)
    
    def Dice(self, s1, s2):
        """ Dice coefficient """
        pass


class LargestCommonString(Metric):
    def __init__(self, name = "LargestCommonString"):
        super().__init__(name)
    
    def __call__(self, s1, s2):
        """ Largest common string """
        return self.LargestCommonString(s1, s2)
    
    def LargestCommonString(self, s1, s2):
        """ Largest common string """
        pass


class Overlap(Metric):
    def __init__(self, name = "Overlap"):
        super().__init__(name)
    
    def __call__(self, s1, s2):
        """ Overlap """
        return self.Overlap(s1, s2)
    
    def Overlap(self, s1, s2):
        """ Overlap """
        pass


class EditDistanceByLPSE(Metric):
    def __init__(self, name = "EditDistanceByLPSE"):
        super().__init__(name)
    
    def __call__(self, s1, s2):
        """ Edit distance by LPSE """
        return self.EditDistanceByLPSE(s1, s2)
    
    def EditDistanceByLPSE(self, s1, s2):
        """ Edit distance by LPSE """
        pass


class EditDistanceCosineSimilarity(Metric):
    """ The combination of Edit-Distance and Cosine Similarity. """
    def __init__(self, name = "EditDistanceCosineSimilarity"):
        super().__init__(name)
    
    def __call__(self, s1, s2):
        """ Edit distance cosine similarity """
        return self.EditDistanceCosineSimilarity(s1, s2)
    
    def EditDistanceCosineSimilarity(self, s1, s2):
        """ Edit distance cosine similarity """
        pass

###############################################################################

class PwdTransformWorkflow(Metric):
    """ The workflow to measure a user's password transformation patterns. 
        refered from the paper ""
    """
    def __init__(self, name = "PwdTransformWorkflow"):
        super().__init__(name)
    
    def identical(self, s1, s2) -> int:
        if s1 == s2:
            return 1
        return 0

    def substring(self, s1, s2) -> int:
        if s1 in s2:
            return 1
        if s2 in s1:
            return -1
        return 0
    
    def cpatalization(self, s1, s2) -> int:
        if s1.capitalize() == s2:
            return 1
        if s2.capitalize() == s1:
            return -1
        return 0
    
    def leet(self, s1, s2) :
        """ check if some characters in s1 was leeted to s2 """
        pass
    
    def reversal(self, s1, s2) -> int:
        """ check if s1 and s2 are reversal """
        if s1[::-1] == s2:
            return 1
        return 0
    
    def sequenceKey(self, s1, s2):
        """ check if s1"""
        pass
    
    def LCS(self, s1, s2):
        """ Largetest Common String (LCS) """
        pass
    
    def CombineRules(self, s1, s2):
        """ Check if s1 transform to s2 with combination rules. """
        pass
    
    def workflow(self, s1, s2):
        

###############################################################################


