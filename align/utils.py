def circulate(items, center=None):
    count = len(items)
    if count > 0:
        if center is None:
            center = count // 2
        center = min(max(center, 0), count - 1)
        yield center, items[center]
        for i in range(1, count):
            print('ANOTHER')
            if center + i < count:
                yield center + i, items[center + i]
            if center - i >= 0:
                yield center - i, items[center - i]


def greedy_minimum_search(a, b, compute, result_a=None, result_b=None):
    if a > b:
        a, b = b, a
        result_a, result_b = result_b, result_a
    if a == b:
        return result_a or result_b or compute(a)
    result_a = result_a or compute(a)
    result_b = result_b or compute(b)
    if b == a+1:
        return result_a if result_a[0] < result_b[0] else result_b
    c = (a+b) // 2
    if result_a[0] < result_b[0]:
        return greedy_minimum_search(a, c, compute, result_a=result_a)
    else:
        return greedy_minimum_search(c, b, compute, result_b=result_b)
