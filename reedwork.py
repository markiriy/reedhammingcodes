import numpy as np

# Определить генератор и матрицы проверки на четность
G = np.array([[1, 1, 1, 1],
              [1, 1, -1, -1]])
H = np.array([[1, 0, 1, 0],
              [0, 1, 1, 1]])

# Определяем функцию кодирования
def encode (mess):
    mess = np.array(mess)
    codeword = np.dot(mess, G)% 2
    return codeword.tolist()

# Определяем функцию декодирования
def decode(res):
    res = np.array(res)
    synd = np.dot(H, res) % 2
    if np.array_equal(synd, np.array([0, 0])):
        return res[:2].tolist()
    else:
        for i in range(4):
            if np.array_equal (synd, H[:, i]):
                err = np.zeros(4)
                err[i] = 1
                fixed = (res + err) % 2
                return fixed[:2].tolist()

# Протестируйте функции на примере сообщения
mess = [4, 3]
codeword = encode(mess)
print("Кодовое слово:", codeword)
res = [1, 0, 1, 1, 1, 0, 1, 1]
decoded = decode(res)
print("Расшифрованное сообщение:", decoded)