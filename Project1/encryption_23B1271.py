import numpy
from math import sqrt

key=input("Please Enter the Key: ").strip().upper()
message=input("Please Enter the Message: ").strip().upper()
key_size=int(sqrt(len(key)))
message_len=len(message)
if (key_size*key_size!=len(key)):
    print("Key is not square")
    exit()

if (len(message)%key_size!=0):
    message=message+"X"*(key_size-len(message)%key_size)

key_matrix=numpy.array([ord(char)-65 for char in key]).reshape(key_size,key_size)
message_matrix=numpy.array([ord(char)-65 for char in message]).reshape(len(message)//key_size, key_size).T
result_matrix=numpy.dot(key_matrix, message_matrix)%26
result="".join([chr(char+65) for char in result_matrix.flatten(order='F')])
print(result)

# Note: If result is to be reduced to the same size as the key, then the following line can be used:
# print(result[:message_len])
