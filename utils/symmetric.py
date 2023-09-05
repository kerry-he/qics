import math

def svec_length(side):
    return side * (side + 1) / 2

def svec_side(len):
    side = math.isqrt(1 + 8 * len) / 2
    assert side * (side + 1) == 2 * len
    return side