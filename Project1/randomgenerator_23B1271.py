import numpy as np

np.random.seed(0)
# Function to generate a random key matrix for Hill cipher
def generate_random_key(size):
    return np.random.randint(26, size=(size, size))  # Generating random key matrix modulo 26

# Function to encrypt using Hill cipher
def hill_cipher_encrypt(plaintext, key):
    n = len(key)  # Size of the key matrix

    # Pad the plaintext if its length is not a multiple of n
    if len(plaintext) % n != 0:
        plaintext += 'X' * (n - len(plaintext) % n)

    # Convert plaintext to numerical values (A=0, B=1, ..., Z=25)
    numerical_plaintext = [ord(char) - ord('A') for char in plaintext]

    # Reshape numerical plaintext into blocks of size n
    blocks = np.array(numerical_plaintext).reshape(-1, n)

    # Encrypt each block using the key
    ciphertext_blocks = []
    for block in blocks:
        encrypted_block = (np.dot(key, block) % 26).tolist()
        ciphertext_blocks.extend(encrypted_block)

    print(blocks)
    print(ciphertext_blocks)
    # Convert numerical ciphertext back to characters
    ciphertext = ''.join(chr(num + ord('A')) for num in ciphertext_blocks)

    return ciphertext

# Generate a random key matrix of size 3x3 for Hill cipher
key = generate_random_key(3)
print("Random Key Matrix:")
print(key)

# Generate a random plaintext of length 9
plaintext = "HELLOWORLD"
print("\nPlaintext:")
print(plaintext)

# Encrypt the plaintext using Hill cipher
ciphertext = hill_cipher_encrypt(plaintext, key)
print("\nCiphertext:")
print(ciphertext)
