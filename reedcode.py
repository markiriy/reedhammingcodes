import operator
import itertools
from functools import reduce


def _binom(n, k):
    """Биномиальный коэффициент (n-k)!/k!."""
    return reduce(operator.mul, range(n - k + 1, n + 1)) // reduce(operator.mul, range(1, k + 1))


def _construct_vector(m, i):
    """Построим вектор для x_i длины 2^m, который имеет вид:
Строка из 2^{m-i-1} 1s, за которой следуют 2 ^{m-i-1} 0s, повторяющаяся
2^m / (2*2 ^{m-i-1}) = 2 ^{m-1}/2 ^ {m-i-1} = 2 ^i раз.
ЗАМЕТЬ: у нас должно быть 0 <= i < m."""
    return ([1] * (2 ** (m - i - 1)) + [0] * (2 ** (m - i - 1))) * (2 ** i)


def _vector_mult(*vecs):
    """For any number of length-n vectors, pairwise multiply the entries, e.g. for
    x = (x_0, ..., x_{n-1}), y = (y_0, ..., y_{n-1}),
    xy = (x_0y_0, x_1y_1, ..., x_{n-1}y{n-1})."""
    assert (len(set(map(len, vecs))) == 1)
    return list(map(lambda a: reduce(operator.mul, a, 1), list(zip(*vecs))))


def _vector_add(*vecs):
    """Для любого числа n векторов длины попарно добавьте записи, например для
x = (x_0,..., x_{n-1}), у = (y_0,..., y_{n-1}),
xy = (x_0+y_0, x_1+y_1,..., x_{n-1}+у{n-1})."""
    assert (len(set(map(len, vecs))) == 1)
    return list(map(lambda a: reduce(operator.add, a, 0), list(zip(*vecs))))


def _vector_neg(x):
    """Возьмем отрицание вектора над Z_2, т.е. поменяем местами 1 и 0."""
    return list(map(lambda a: 1 - a, x))


def _vector_reduce(x, modulo):
    """Уменьшите каждый элемент вектора по модулю предоставленного значения."""
    return list(map(lambda a: a % modulo, x))


def _dot_product(x, y):
    """Вычислите точечное произведение двух векторов."""
    assert (len(x) == len(y))
    return sum(_vector_mult(x, y))


def _generate_all_rows(m, S):
    """Сгенерируйте все строки над одночленами в S, например, если S = {0,2}, мы хотим сгенерировать
список из четырех строк, а именно:
где (x_0) * где (x_2)
где (x_0)* !где (x_2)
!где (x_0) * где (x_2)
!где (x_0)* !где (x_2).
Мы делаем это, используя рекурсию на S."""

    if not S:
        return [[1] * (2 ** m)]

    i, Srest = S[0], S[1:]

    # Найдите все строки над Srest.
    Srest_rows = _generate_all_rows(m, Srest)

    # Теперь, как для представления x_i, так и для !x_i, верните строки, умноженные на эти.
    xi_row = _construct_vector(m, i)
    not_xi_row = _vector_neg(xi_row)
    return [_vector_mult(xi_row, row) for row in Srest_rows] + [_vector_mult(not_xi_row, row) for row in Srest_rows]


class ReedMuller:
    """Класс, представляющий код Рида-Мюллера RM(r,m), который кодирует слова длиной:
k = C(m,0) + C(m,1) + ... + C(m,r)
к словам длиной n = 2^m.
Обратите внимание, что C(m, 0) + ... + C(m, m) = 2 ^ m, поэтому k <= n во всех случаях, как и ожидалось.
Код RM(r,m) имеет вес 2^{m-r} и, таким образом, может исправить до 2^{m-r-1}-1 ошибок."""

    def __init__(self, r, m):
        """Создайте кодер/ декодер Рида-Мюллера для RM(r,m)."""
        self.r, self.m = (r, m)
        self._construct_matrix()
        self.k = len(self.M[0])
        self.n = 2 ** m

    def strength(self):
        """Возвращает надежность кода, то есть количество ошибок, которые мы можем исправить."""
        return 2 ** (self.m - self.r - 1) - 1

    def message_length(self):
        """Длина сообщения, подлежащего кодированию."""
        return self.k

    def block_length(self):
        """Длина закодированного сообщения."""
        return self.n

    def _construct_matrix(self):
        # Постройте все строки x_i.
        x_rows = [_construct_vector(self.m, i) for i in range(self.m)]

        # Для каждого s-установите S для всех 0 <= s <= r, создайте строку, которая является произведением векторов x_j для j в S.
        self.matrix_by_row = [reduce(_vector_mult, [x_rows[i] for i in S], [1] * (2 ** self.m))
                              for s in range(self.r + 1)
                              for S in itertools.combinations(range(self.m), s)]

        # Для декодирования для каждой строки матрицы нам нужен список всех векторов, состоящий из представлений
        # из всех одночленов, не находящихся в строке. Это строки, которые используются при голосовании, чтобы определить, есть ли 0 или 1
        self.voting_rows = [_generate_all_rows(self.m, [i for i in range(self.m) if i not in S])
                            for s in range(self.r + 1)
                            for S in itertools.combinations(range(self.m), s)]

        # Теперь единственное, что нам нужно, - это список индексов строк, соответствующих одночленам степени i.
        self.row_indices_by_degree = [0]
        for degree in range(1, self.r + 1):
            self.row_indices_by_degree.append(self.row_indices_by_degree[degree - 1] + _binom(self.m, degree))

        # # Теперь мы хотим выполнить транспонирование для кодовой матрицы, чтобы облегчить умножение векторов справа на матрицу.
        self.M = list(zip(*self.matrix_by_row))

    def encode(self, word):
        """Закодируйте вектор длины k в вектор длины n."""
        assert (len(word) == self.k)
        return [_dot_product(word, col) % 2 for col in self.M]

    def decode(self, eword):
        """Декодируйте вектор длины n обратно в его исходный вектор длины k, используя мажоритарную логику."""
# Мы хотим выполнить итерацию по каждой строке r матрицы и определить, появляется ли 0 или 1 в
# позиции r исходного слова w, используя мажоритарную логику.

        row = self.k - 1
        word = [-1] * self.k

        for degree in range(self.r, -1, -1):
            # Мы рассчитываем количество записей для получения степени. Нам нужен диапазон строк кодовой матрицы
            # соответствует степени r.
            upper_r = self.row_indices_by_degree[degree]
            lower_r = 0 if degree == 0 else self.row_indices_by_degree[degree - 1] + 1

            # Теперь выполните итерацию по этим строкам, чтобы определить значение word для позиций lower_right
            # через upper_r включительно.
            for pos in range(lower_r, upper_r + 1):
                # Мы голосуем за значение этой позиции на основе векторов в voting_rows.
                votes = [_dot_product(eword, vrow) % 2 for vrow in self.voting_rows[pos]]

                # Если будет ничья, мы ничего не сможем сделать.
                if votes.count(0) == votes.count(1):
                    return None

                # В противном случае мы присваиваем позицию победителю.
                word[pos] = 0 if votes.count(0) > votes.count(1) else 1

            # Теперь нам нужно изменить слово. Мы хотим вычислить произведение того, что мы только что
            # проголосовано по строкам матрицы.
            s = [_dot_product(word[lower_r:upper_r + 1], column[lower_r:upper_r + 1]) % 2 for column in self.M]
            eword = _vector_reduce(_vector_add(eword, s), 2)

        # Задекодили
        return word

    def __repr__(self):
        return '<Reed-Muller code RM(%s,%s), strength=%s>' % (self.r, self.m, self.strength())


def _generate_all_vectors(n):
    """Генератор для получения всех возможных векторов длины n в Z_2."""
    v = [0] * n
    while True:
        yield v

        # Сгенерируйте следующий вектор, добавив 1 в конец.
        # Затем продолжайте изменять на 2 и перемещайте все излишки обратно вверх по вектору.
        v[n - 1] = v[n - 1] + 1
        pos = n - 1
        while pos >= 0 and v[pos] == 2:
            v[pos] = 0
            pos = pos - 1
            if pos >= 0:
                v[pos] += 1

        # Завершается, если мы снова достигнем вектора all-0.
        if v == [0] * n:
            break


def _characteristic_vector(n, S):
    """Возвращает характеристический вектор подмножеств n-множества."""
    return [0 if i not in S else 1 for i in range(n)]


if __name__ == '__main__':
    # Проверьте правильность аргументов командной строки и, если они отсутствуют, выведите информационное сообщение.
    import sys

    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s r m\n' % (sys.argv[0],))
        sys.exit(1)
    r, m = map(int, sys.argv[1:])
    if (m <= r):
        sys.stderr.write('We require r > m.\n')
        sys.exit(2)

    # Создание
    rm = ReedMuller(r, m)
    strength = rm.strength()
    message_length = rm.message_length()
    block_length = rm.block_length()

    # Создайте список всех возможных ошибок с максимальной точностью.
    error_vectors = [_characteristic_vector(block_length, S)
                     for numerrors in range(strength + 1)
                     for S in itertools.combinations(range(block_length), numerrors)]

    # Закодируйте все возможные сообщения message_length.
    success = True
    for word in _generate_all_vectors(message_length):
        codeword = rm.encode(word)

        # Теперь произведите все исправляемые ошибки и убедитесь, что мы все еще расшифровываем нужное слово.
        for error in error_vectors:
            error_codeword = _vector_reduce(_vector_add(codeword, error), 2)
            error_word = rm.decode(error_codeword)
            if error_word != word:
                print('ERROR: encode(%s) => %s, decode(%s+%s=%s) => %s' % (word, codeword, codeword,
                                                                           error, error_codeword, error_word))
                success = False

    if success:
        print('RM(%s,%s): success.' % (r, m))