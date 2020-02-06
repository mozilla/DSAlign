
from multiprocessing.dummy import Pool as ThreadPool

def circulate(items, center=None):
    count = len(list(items))
    if count > 0:
        if center is None:
            center = count // 2
        center = min(max(center, 0), count - 1)
        yield center, items[center]
        for i in range(1, count):
            #print('ANOTHER')
            if center + i < count:
                yield center + i, items[center + i]
            if center - i >= 0:
                yield center - i, items[center - i]


def by_len(items):
    indexed = list(enumerate(items))
    return sorted(indexed, key=lambda e: len(e[1]), reverse=True)


def enweight(items, direction=0):
    """
    Enumerates all entries together with a positional weight value.
    The positional weight progresses quadratically.
    :param items: Items to enumerate
    :param direction: Order of assigning positional weights to N-grams:
        direction < 0: Weight of first N-gram is 1.0 and of last one 0.0
        direction > 0: Weight of first N-gram is 0.0 and of last one 1.0
        direction == 0: Weight of center N-gram(s) near or equal 0, weight of first and last N-gram 1.0
    :return: Produces (object, float) tuples representing the enumerated item
             along with its assigned positional weight value
    """
    items = list(items)
    direction = -1 if direction < 0 else (1 if direction > 0 else 0)
    n = len(items) - 1
    if n < 1:
        if n == 0:
            yield items[0], 1
        raise StopIteration
    for i, item in enumerate(items):
        c = (i + n * (direction - 1) / 2) / n
        yield item, c * c * (4 - abs(direction) * 3)


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


class LimitingPool:
    def __init__(self, processes=None, limit_factor=2, sleeping_for=0.1):
        self.processes = os.cpu_count() if processes is None else processes
        self.pool = ThreadPool(processes=processes)
        self.sleeping_for = sleeping_for
        self.max_ahead = self.processes * limit_factor
        self.processed = 0

    def __enter__(self):
        return self

    def limit(self, it):
        for obj in it:
            while self.processed >= self.max_ahead:
                time.sleep(self.sleeping_for)
            self.processed += 1
            yield obj

    def map(self, fun, it):
        for obj in self.pool.imap(fun, self.limit(it)):
            self.processed -= 1
            yield obj

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.close()
