import sys
import reedcode

if __name__ == '__main__':
    # Проверьте параметры командной строки.
    if len(sys.argv) < 4:
        sys.stderr.write('Usage: %s r m codeword [codeword [...]]\n' % (sys.argv[0],))
        sys.exit(1)

    r, m = map(int, sys.argv[1:3])
    if (m <= r):
        sys.stderr.write('We require r < m.\n')
        sys.exit(2)

    # Создание
    rm = reedcode.ReedMuller(r, m)
    n = rm.block_length()

    # Теперь выполните итерацию по словам для кодирования, подтвердите их и закодируйте.
    for codeword in sys.argv[3:]:
        try:
            listword = list(map(int, codeword))
            if (not set(listword).issubset([0, 1])) or (len(listword) != n):
                sys.stderr.write('FAIL: word %s is not a 0-1 string of length %d\n' % (codeword, n))
            else:
                decodeword = rm.decode(listword)
                if not decodeword:
                    print('Could not unambiguously decode word %s' % (codeword))
                else:
                    print(''.join(map(str, decodeword)))
        except:
            print("Unexpected error:", sys.exc_info()[0])