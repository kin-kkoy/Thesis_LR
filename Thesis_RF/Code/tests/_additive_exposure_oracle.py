"""Independent test oracle. No project or NumPy imports."""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import math
import struct


class OracleInconclusive(ArithmeticError):
    pass


def encoding(value, width):
    if width not in (32, 64):
        raise ValueError("Encoding width must be 32 or 64")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Encoding distance requires nonnegative finite outputs")
    value = 0.0 if value == 0 else value
    return struct.unpack(">I" if width == 32 else ">Q",
                         struct.pack(">f" if width == 32 else ">d", value))[0]


def distance(a, b, width):
    return abs(int(encoding(a, width)) - int(encoding(b, width)))


def decode(bits, width):
    fraction_bits, exponent_bits = (23, 8) if width == 32 else (52, 11)
    sign = -1 if bits >> (fraction_bits + exponent_bits) else 1
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
    mantissa = bits & ((1 << fraction_bits) - 1)
    bias = (1 << (exponent_bits - 1)) - 1
    if exponent == (1 << exponent_bits) - 1:
        raise ValueError("Nonfinite encoding")
    if exponent:
        mantissa += 1 << fraction_bits
    power = (exponent - bias if exponent else 1 - bias) - fraction_bits
    return sign * Fraction(mantissa) * (Fraction(2) ** power)


def round_bits(value, width):
    """Integer nearest-even implementation, independent of native float casts."""
    value = Fraction(value)
    fraction_bits, exponent_bits = (23, 8) if width == 32 else (52, 11)
    bias = (1 << (exponent_bits - 1)) - 1
    sign = int(value < 0) << (fraction_bits + exponent_bits)
    value = abs(value)
    if not value:
        return 0
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    exponent = max(exponent, 1 - bias)
    scaled = value / (Fraction(2) ** (exponent - fraction_bits))
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    integer += int(twice > scaled.denominator or
                   (twice == scaled.denominator and integer % 2 == 1))
    if integer == 1 << (fraction_bits + 1):
        integer >>= 1
        exponent += 1
    if exponent > bias:
        raise OverflowError("Oracle rounding overflow")
    if integer < 1 << fraction_bits:
        return sign | integer
    return sign | ((exponent + bias) << fraction_bits) | (integer - (1 << fraction_bits))


def susceptibility(s, building, proximity, material):
    coefficients = [decode(round_bits(Fraction(c), 32), 32)
                    for c in ("0.20", "0.10", "0.20", "0.35")]
    products = [round_bits(c * decode(struct.unpack(">I", struct.pack(">f", float(v)))[0], 32), 32)
                for c, v in zip(coefficients, (s, building, proximity, material))]
    total = round_bits(Fraction("0.12"), 32)
    additions = []
    for product in products:
        total = round_bits(decode(total, 32) + decode(product, 32), 32)
        additions.append(total)
    clipped = min(Fraction(1), max(Fraction(0), decode(total, 32)))
    return products, additions, round_bits(clipped, 32)


def _interval(beta, exposure, precision):
    """Outward enclosure using correctly rounded Decimal ln/exp and neighbors."""
    with localcontext() as context:
        context.prec = precision
        x = Decimal(beta.numerator) / Decimal(beta.denominator)
        e = Decimal(exposure.numerator) / Decimal(exposure.denominator)
        base = Decimal(1) - x
        if Fraction(base) != 1 - beta or Fraction(e) != exposure:
            raise OracleInconclusive("Insufficient exact decimal input precision")
        logarithm = base.ln()
        lower_log, upper_log = logarithm.next_minus(), logarithm.next_plus()
        context.rounding = ROUND_FLOOR
        lower_argument = lower_log * e
        context.rounding = ROUND_CEILING
        upper_argument = upper_log * e
        lower_exp = lower_argument.exp().next_minus()
        upper_exp = upper_argument.exp().next_plus()
        context.rounding = ROUND_FLOOR
        lower = max(Decimal(0), Decimal(1) - upper_exp)
        context.rounding = ROUND_CEILING
        upper = min(Decimal(1), Decimal(1) - lower_exp)
        return Fraction(lower), Fraction(upper)


def probability(beta_value, score_value):
    beta = decode(encoding(beta_value, 32), 32)
    score = decode(encoding(score_value, 32), 32)
    if not 0 <= beta <= 1 or not 0 <= score <= 1:
        raise ValueError("Oracle input range")
    exposure = min(Fraction(8), max(Fraction(0), score * 8))
    if not exposure or not beta:
        return round_bits(exposure, 64), 0, 0
    if beta == 1:
        return round_bits(exposure, 64), round_bits(1, 64), round_bits(1, 32)
    # Integer exposure admits an exact rational reference, including exact ties.
    exact = 1 - (1 - beta) ** exposure.numerator if exposure.denominator == 1 else None
    rounded = []
    for precision in (256, 512):
        lower, upper = _interval(beta, exposure, precision) if exact is None else (exact, exact)
        pair = []
        for width in (64, 32):
            low_bits, high_bits = round_bits(lower, width), round_bits(upper, width)
            if low_bits != high_bits:
                raise OracleInconclusive("Enclosure crosses a binary rounding boundary")
            pair.append(low_bits)
        rounded.append(tuple(pair))
    if rounded[0] != rounded[1]:
        raise OracleInconclusive("256/512 rounding disagreement")
    return round_bits(exposure, 64), *rounded[1]


def ideal_comparison(beta_value, score_value, offsets, wind):
    """Descriptive high-precision ideal reference, never a label authority."""
    with localcontext() as context:
        context.prec = 256
        root = Decimal("0.5").sqrt()
        directions = {0: (Decimal(1), Decimal(0)), 90: (Decimal(0), Decimal(-1)),
                      180: (Decimal(-1), Decimal(0)), 270: (Decimal(0), Decimal(1)),
                      315: (root, root)}
        speed, bearing = wind
        row, col = directions[int(bearing)]
        ideal = Decimal(0)
        for dr, dc in offsets:
            norm = Decimal(dr * dr + dc * dc).sqrt()
            alignment = (-Decimal(dr) * row - Decimal(dc) * col) / norm
            ideal += max(Decimal("0.05"),
                         Decimal(1) + Decimal("0.20") * (Decimal(speed) / 10) * alignment)
        beta = Decimal.from_float(float(beta_value))
        quantized = Decimal.from_float(float(score_value)) * 8
        def q(exposure):
            if not beta or not exposure:
                return Decimal(0)
            if beta == 1:
                return Decimal(1)
            return 1 - ((1 - beta).ln() * exposure).exp()
        return {"exposure_quantized_minus_ideal": str(quantized - ideal),
                "mathematical_probability_quantized_minus_ideal": str(q(quantized) - q(ideal)),
                "ideal_exposure": str(ideal)}
