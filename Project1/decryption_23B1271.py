# import numpy

# plaintext=input("Please Enter the Plaintext: ").strip().upper()
# ciphertext=input("Please Enter the Ciphertext: ").strip().upper()

# def gauss_jordan_elimination():
#     global message_matrix, ciphertext_matrix
#     left_matrix=numpy.concatenate((numpy.dstack((numpy.eye(3),numpy.ones((3,3)))), message_matrix), axis=1)
#     print(left_matrix)

# key_size=3
# message_len=len(plaintext)
# if message_len%key_size!=0:
#     plaintext=plaintext+"X"*(key_size-message_len%key_size)
# if len(ciphertext)%key_size!=0:
#     print("Ciphertext length is not a multiple of 3")
#     exit()

# message_matrix=numpy.array([ord(char)-65 for char in plaintext]).reshape(len(plaintext)//key_size, key_size).T
# message_matrix=numpy.dstack((message_matrix, numpy.ones((key_size, len(plaintext)//key_size))))
# ciphertext_matrix=numpy.array([ord(char)-65 for char in ciphertext]).reshape(len(ciphertext)//key_size, key_size).T
# ciphertext_matrix=numpy.dstack((ciphertext_matrix, numpy.ones((key_size, len(plaintext)//key_size))))
# # key_matrix=numpy.dot(ciphertext_matrix, numpy.linalg.pinv(message_matrix))%26

# # key="".join([chr(char+65) for char in key_matrix.flatten(order='C')])
# # print(key)

# print(message_matrix)
# print(ciphertext_matrix)

# gauss_jordan_elimination()

import numpy as np
from math import gcd

plaintext=input("Please Enter the Plaintext: ").strip().upper()
ciphertext=input("Please Enter the Ciphertext: ").strip().upper()
# plaintext="SUMSUMPLANSX"
# ciphertext="COACOAOZWJBH"

def gauss_jordan_elimination(A, B):
    n = len(A)
    augmented_matrix = np.concatenate((A, B), axis=1).astype(float)
    
    # Forward Elimination
    for i in range(n):
        if augmented_matrix[i][i] == 0.0:
            for j in range(i + 1, n):
                if augmented_matrix[j][i] != 0.0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    break

        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i][i]

        for j in range(i + 1, n):
            augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j][i]

    # Backward Elimination
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j][i]
            
    # Extracting the solution
    D = augmented_matrix[:, n:]
    return D

key_size = 3
message_len = len(plaintext)
if message_len % key_size != 0:
    plaintext = plaintext + "X" * (key_size - message_len % key_size)
if len(ciphertext) % key_size != 0:
    print("Ciphertext length is not a multiple of 3")
    exit()

message_matrix = np.array([ord(char) - 65 for char in plaintext]).reshape(len(plaintext) // key_size, key_size)
ciphertext_matrix = np.array([ord(char) - 65 for char in ciphertext]).reshape(len(ciphertext) // key_size, key_size)
column_number=len(ciphertext_matrix)
done=False
for i in range(column_number):
    for j in range(i+1, column_number):
        for k in range(j+1, column_number):
            a=np.column_stack((message_matrix[i], message_matrix[j], message_matrix[k])).astype(float)
            determinant=round(np.linalg.det(a))
            if gcd(determinant, 26) == 1:
                cofactor_matrix=np.matrix([[(a[(row+1)%3][(col+1)%3]*a[(row+2)%3][(col+2)%3]-a[(row+1)%3][(col+2)%3]*a[(row+2)%3][(col+1)%3]) for row in range(3)] for col in range(3)]).astype(int)%26
                key=np.column_stack((ciphertext_matrix[i], ciphertext_matrix[j], ciphertext_matrix[k])).astype(int)%26
                key=np.matmul(key, cofactor_matrix)
                key*=pow(determinant, -1, 26)
                key%=26
                done=True
                break
        if done:
            break
    if done:
        break
if done:
    print(key)
    key="".join([chr(int(round(char)) + 65) for char in np.array(key.flatten(order='C'))[0]])
    print(key)
    exit()

# for i in range(column_number):
#     for j in range(i+1, column_number):
#         linarly_independent=True
#         for k in range(26):
#             if np.sum(((message_matrix[i]*k)-message_matrix[j])%26)==0:
#                 linarly_independent=False
#                 break
#         if not linarly_independent:
#             continue
#         a=np.column_stack((message_matrix[i], message_matrix[j])).astype(float)
#         determinant=round(np.linalg.det(np.matmul(a.T, a)))
#         if gcd(determinant, 26) == 1:
#             ata=np.matmul(a.T, a)
#             cofactor_matrix=np.matrix([[pow(-1, row+col)*ata[(row+1)%2][(col+1)%2] for row in range(2)] for col in range(2)]).T.astype(int)%26
#             cofactor_matrix=np.matmul(cofactor_matrix, a.T)
#             cofactor_matrix*=pow(determinant, -1, 26)
#             cofactor_matrix%=26
#             key=np.column_stack((ciphertext_matrix[i], ciphertext_matrix[j])).astype(int)%26
#             key=np.matmul(key, cofactor_matrix)
#             print(a)
#             print(message_matrix.T)
#             print(np.matmul(key, message_matrix.T)%26)
#             print(np.column_stack((ciphertext_matrix[i], ciphertext_matrix[j])))
#             key%=26
#             done=True
#             break
#     if done:
#         break
# if done:
#     exit()

def solve_brute_force(row):
    output=[]
    result=ciphertext_matrix.T[row]
    for c in range(column_number):
        for r in range(3):
            if message_matrix[c][r]%2 == 0 or message_matrix[c][r]%13 == 0:
                continue
            if r==0:
                i_row=1
                j_row=2
            elif r==1:
                i_row=0
                j_row=2
            else:
                i_row=0
                j_row=1
            for i in range(26):
                for j in range(26):
                    k=((ciphertext_matrix[c][row]-i*message_matrix[c][i_row]-j*message_matrix[c][j_row])*pow(int(message_matrix[c][r]), -1, 26))%26
                    if r==0:
                        key_matrix=np.array([k, i, j])
                    elif r==1:
                        key_matrix=np.array([i, k, j])
                    else:
                        key_matrix=np.array([i, j, k])
                    if np.array_equal(np.matmul(key_matrix, message_matrix.T)%26, result):
                        output.append(key_matrix)
            return output
    for i in range(26):
        for j in range(26):
                for k in range(26):
                    key_matrix=np.array([i, j, k])
                    if np.array_equal(np.matmul(key_matrix, message_matrix.T)%26, result):
                        output.append(key_matrix)
    return output
    


key=[]
for row in range(3):
    key.append(solve_brute_force(row))
for choice1 in key[0]:
    for choice2 in key[1]:
        for choice3 in key[2]:
            key_matrix=np.column_stack((choice1, choice2, choice3))
            key="".join([chr(int(round(char)) + 65) for char in np.array(key_matrix.flatten(order='C'))[0]])            
            print(key_matrix)
            print(key)
