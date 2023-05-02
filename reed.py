from reedcode import *

# Параметры
r = 4  # r >= m
m = 3

# создание объекта класса
rm = ReedMuller(r, m)

# сообщение
msg = [1, 0, 1, 1, 1, 0, 1, 1] # длина сообщения 2**m битов

# кодирование сообщения
codeword = rm.encode(msg)

# изменение кодового слова
codeword[2] = 1

# раскодирование кодового слова
msg_decoded = rm.decode(codeword)

# ответы
print("Original message: ", msg)
print('Encoded message: ', codeword)
print("Decoded message: ", msg_decoded)