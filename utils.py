# -*- coding:utf-8 -*-
# Author: K.D. Xiu
# Create Date: 10/16/2023
# Description:
#   Some utils functions

import math
import numpy


def Keyboard_Characters():
    return 'aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ1!2@3#4$5%6^7&8*9(0)-_=+[{]}\\|;:\'\",<.>/?`~AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz!1@2#3$4%5^6&7*8(9)0_-+={[}]|\\:;\"\'<,>.?/~`  '

def Keyboard_Characters_without_shift():
    return "`1234567890-=qwertyuiop[]\\asdfghjkl;\'zxcvbnm,./ "

def Keyboard_Characters_with_shift():
    return '~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:\"ZXCVBNM<>?'


# 计算字符串向量
def calc_vec1D(psw1, psw2):
    """生成字符串特征向量，这里实际上仍是1D的，因为每个字符只会转化成一维的坐标

    Args:
        psw1 (str): password 1
        psw2 (str): passwrod 2

    Returns:
        list, list: vector1, vector2
    """

    key_chars = list(set(list(psw1) + list(psw2)))
    vec1 = [0] * len(key_chars)
    vec2 = [0] * len(key_chars)
    char_idx = {w: i for i, w in enumerate(key_chars)}
    # idx_char = {i: w for i, w in enumerate(key_chars)}
    for c in psw1:
        vec1[char_idx[c]] += 1
    for c in psw2:
        vec2[char_idx[c]] += 1
    return vec1, vec2
