def maybe_replace_leading_digit(val):
    val = val.replace('0th', 'first')
    val = val.replace('1th', 'second')
    val = val.replace('2th', 'third')
    val = val.replace('3th', 'fourth')
    val = val.replace('4th', 'fifth')
    return val.replace('5th', 'sixth')