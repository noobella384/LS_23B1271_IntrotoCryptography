import Crypto
from Crypto.Random import random
import Crypto.Random
from Crypto.Util.number import bytes_to_long, long_to_bytes
import random
from decimal import Decimal, getcontext
import sys

getcontext().prec = 1024

def modularExponentiation(a, b, n):
    result = 1
    a = a % n
    while b > 0:
        if b % 2 == 1:
            result = (result * a) % n
        b = b // 2
        a = (a * a) % n
    return result

class RSA:
    """Implements the RSA public key encryption / decryption."""

    def __init__(self, key_length):
        # define self.p, self.q, self.e, self.n, self.d here based on key_length
        self.__p = Crypto.Util.number.getPrime(key_length)
        self.__q = Crypto.Util.number.getPrime(key_length)
        self.n = self.__p * self.__q
        self.__tot = (self.__p - 1) * (self.__q - 1)
        self.e = random.randint(1, self.__tot)
        while Crypto.Util.number.GCD(self.e, self.__tot) != 1:
            self.e += 1
        self.d = pow(self.e, -1, self.__tot)

    def encrypt(self, binary_data):
        int_data = bytes_to_long(binary_data)
        int_data %= self.n
        return pow(int_data, self.e, self.n)

    def decrypt(self, encrypted_int_data):
        return long_to_bytes(pow(encrypted_int_data, self.d, self.n)).decode()

class RSAParityOracle(RSA):
    """Extends the RSA class by adding a method to verify the parity of data."""

    def is_parity_odd(self, encrypted_int_data):
        result = pow(encrypted_int_data, self.d, self.n)
        return result % 2 == 1

def parity_oracle_attack(ciphertext, rsa_parity_oracle):
    n = rsa_parity_oracle.n
    original_ciphertext = ciphertext%n
    ciphertext %= n
    if n % 2 == 0:
        d = pow(rsa_parity_oracle.e, -1, n // 2 - 1)
        return long_to_bytes(pow(ciphertext, d, n)).decode()
    left = 0
    right = n-1
    mid = (left + right) // 2
    power = modularExponentiation(2, rsa_parity_oracle.e, n)
    while left < right:
        mid = (left + right) // 2
        ciphertext *= power
        ciphertext %= n
        if rsa_parity_oracle.is_parity_odd(ciphertext):
            left = mid + 1
        else:
            right = mid
    for i in range(max(0, left - 1000), left + 1000):
        if pow(i, rsa_parity_oracle.e, n) == original_ciphertext:
            return long_to_bytes(i).decode()
    return None

def main():
    # input_bytes = input("Enter the message: ")
    assert len(sys.argv) == 2
    input_bytes = sys.argv[1]
    encoded_message = input_bytes.encode()

    # Generate a 1024-bit RSA pair    
    rsa_parity_oracle = RSAParityOracle(1024)
    # Encrypt the message
    ciphertext = rsa_parity_oracle.encrypt(encoded_message)
    print("Encrypted message is: ", ciphertext)

    print("Decrypted text is: ", rsa_parity_oracle.decrypt(ciphertext))

    # Check if the attack works
    plaintext = parity_oracle_attack(ciphertext, rsa_parity_oracle)
    print("Obtained plaintext: ", plaintext)
    if plaintext != input_bytes:
        print(rsa_parity_oracle.n)
        print(rsa_parity_oracle.e)
        print(rsa_parity_oracle.d)
        print(input_bytes)
        exit(1)

if __name__ == '__main__':
    main
