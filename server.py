import socket
import numpy as np
import time
import random
import copy
from joblib import Parallel, delayed

k = 33 
round_num = 6 #log(k-1)+1

# example boolean share
bo1 = ['010010100100001100101001101010100', '011111010010011101110100111001110', '001100010110110110001100110100000', '111100001010101010001010101001111', '110001110010010101110110100100110', '010000001011110110111011111000101', '001111011100101110100000100110011', '111000010001110100011010110000000', '100000010000110000011100110100001', '010101111101110100110110110100001', '111001011100101110011100110011011', '111111001101010100001100001010000', '000111101110010100001011110010111', '111011101100101010100100010111111', '111001110001100011010101001101111', '001111000001010110101001100010101', '001010110111111001110101010011010', '110000011111010100011111100100011', '000001100111111010001010111111111', '111000010100010001010001011100101', '000010001100000101111000011100010', '111111010111111010111100111100000', '000101111110111110110010011100011', '010010011100100101110010100101111', '010000100101001010111111001110000', '000100100010111001101110011111100', '001001010000010100100111101100100', '010111010111000011010110011111001', '111011111111011100001001101100010', '000011110110111001000111111011110', '110110101010001111111101110110101', '100111010110011011100101100100011', '100101111011010101011010010101011', '100011110010011011000111110110011', '101010101100000001000101110000011', '010001001011000111011110000010010', '101011010001001101001100011010110', '101000011010111000111010111100111', '101011001011001001101111000001111', '111011111011101001110000101010111', '101110101001101101000001101110110', '100100000100000001101011100010101', '010110111110000111111111111101110', '000100100101001101011111010101010', '110010010001000100101000111010000', '011101100011101001011010111101101', '010011111101010101101010110011011', '100010111101011111100101110111001', '101110011111110000110100100001001', '110101011011001010011010110010100', '110000111001011100111010010110101', '111111011010110001001000000110011', '011111011111100100001011000000100', '111111110111000110010110110111010', '110110000111111010101110001010010', '110010100001101000111010000001101', '111111110011010100111111101011101', '001011010101101111011010011111011', '101110001011010010001011010101010', '000101011011010111101101100000100', '110011101001101011000011110001100', '101100100010111000000110111101111', '101100011111010101101010111100010', '000011010010101011011011110001110', '011001001010101010111000001111000', '011010001001100111101101000101010', '110010010110011001010111001010100', '011100000010111110000000001010111', '101101011011001110011101010001101', '010010111110001111001101010001110', '011111110101101001101100110101011', '011001100100011101101101100110101', '000111111100001000000010111110110', '011011110111100001110000001001111', '111100010101110100010000111101001', '111111111110010000101100110101011', '111110001000101110011101011111110', '011101010111110110000001001010001', '011010101001001011000100100010100', '111110000101011101101111110001111', '111000100010101010011101011011100', '011111101011010010000111110111000', '011001101110101001101101001001011', '110011000010101010011110010010001', '110110111000000000100001001101111', '011000010100010110000000110111111', '001100111001111010110001010101010', '010111101010100001010010011111101', '011001001100101001110101000011010', '011001100000011110000111000011110', '000101010101011101110101100000011', '110100001001110101000110101111000', '110000010101101110100101001111111', '011010110101001111110011010001010', '111010101110111001011111001100101', '010010110000001001110111111110100', '000001010000100001001100001101000', '001001010101111000001100101010001', '010011101000110000110100011111100', '100010000100001000110100111011000', '001100101010000100000111000111111', '010001110001001111011111000000011', '100100111110111001101100001011011', '011110111100010111110110000100101', '111011100010100110000010100110100', '000001011011001011101100100010011', '110110000111101010011110011111110', '111001111111110100100010000110101', '011010000000110011111111000010010', '001100000110000100101100100110110', '110110100001000000011010001110010', '001011001101100101101000001011101', '100111100111111101001100011100010', '101101110101000000000011110101101', '011001110010110110101110000011001', '100110001101011011001000101100001', '111101010001101001011000110011000', '010100110011111011110001101001010', '000110111100011000000111111110110', '110001010000100100111101000011101', '000011111110110001001000111011100', '111100110001111110001100101011011', '100110000111111010011001111010001', '101111010010001000011000000110001', '011110110011010111110010101101000']

# socket preparation
print('[Socket Prepared]')
server = socket.socket()
s_name = socket.gethostname()
server.bind((s_name, 8080))
server.listen(3)


def mul_share(b, bit): # OT based multiplication in FL

    client, address = server.accept()
    choice = client.recv(16384).decode()
    choice = bin(int(choice, 16))[3:]
    result = np.zeros(bit, dtype=np.int8)

    for i in range(bit):
        result[i] = int(choice[4*i+b[i][1]]) ^ int(choice[4 * i+2+b[i][0]]) ^ (b[i][0] & b[i][1])

    client.close()
    return result

def sort1(bob): # naive comparison algorithm

    B = np.zeros((k+1, 2), dtype=np.int8)

    for i in range(k):
        B[i] = [int(bob[0][i]), int(bob[1][i]) ^ 1]

    B[k] = [int(bob[0][0]) ^ int(bob[1][0]) ^ 1,
             int(bob[0][1]) ^ int(bob[1][1]) ^ 1]

    result1 = mul_share(B, k+1)

    bob_c = 1 ^ result1[0]

    B = np.array([[int(bob[0][0]) ^ int(bob[1][0]) ^ 1, result1[1]],
                  [result1[-1], result1[2]],
                  [result1[-1], int(bob[0][2]) ^ int(bob[1][2]) ^ 1]])

    result2 = mul_share(B, 3)

    bob_c = bob_c ^ result2[0] ^ result2[1]

    for i in range(3, k-1):

        B = np.array([[result2[-1], result1[i]],
                      [result2[-1], int(bob[0][i]) ^ int(bob[1][i]) ^ 1]])

        result2 = mul_share(B, 2)

        bob_c = bob_c ^ result2[0]

    B = np.array([[result2[-1], result1[k-1]]])

    result3 = mul_share(B, 1)

    bob_c = bob_c ^ result3[0]

    return bob_c

def pa_sort1(bob): # naive comparison algorithm of parallel computing
    
    m = len(bob)

    B = np.zeros(((k+1)*m,2), dtype = np.int8)
    
    for j in range(m):
        for i in range(k):
            B[(k+1)*j+i] = [int(bob[j][0][i]), int(bob[j][1][i])^1]

        B[(k+1)*j+k] = [int(bob[j][0][0])^int(bob[j][1][0])^1, int(bob[j][0][1])^int(bob[j][1][1])^1]
    
    result1 = mul_share(B, len(B))

    bob_c = [1^result1[(k+1)*j] for j in range(m)]

    B = np.zeros((3*m,2), dtype = np.int8)    
    
    for j in range(m):       
        B[3*j:3*j+3] = np.array([[int(bob[j][0][0])^int(bob[j][1][0])^1, result1[(k+1)*j+1]],
                                 [result1[(k+1)*(j+1)-1], result1[(k+1)*j+2]],
                                 [result1[(k+1)*(j+1)-1], int(bob[j][0][2])^int(bob[j][1][2])^1]])

    result2 = mul_share(B, len(B))
    
    for j in range(m):
        bob_c[j] = bob_c[j]^result2[3*j]^result2[3*j+1]
    
    for i in range(3, k-1):
        B = np.zeros((2*m,2), dtype = np.int8)
        
        for j in range(m):
            if i == 3:
                
                B[2*j:2*j+2] = np.array([[result2[3*(j+1)-1], result1[(k+1)*j+i]],
                                         [result2[3*(j+1)-1], int(bob[j][0][i])^int(bob[j][1][i])^1]])
                
            else:

                B[2*j:2*j+2] = np.array([[result2[2*(j+1)-1], result1[(k+1)*j+i]],
                                         [result2[2*(j+1)-1], int(bob[j][0][i])^int(bob[j][1][i])^1]])
        
        result2 = mul_share(B, len(B))
        
        for j in range(m):
            bob_c[j] = bob_c[j]^result2[2*j]
        
    B = np.zeros((m,2), dtype = np.int8)
    
    for j in range(m): 
        B[j] = np.array([result2[2*(j+1)-1], result1[(k+1)*j+k-1]]) 
    
    result3 = mul_share(B, len(B))
    
    for j in range(m):
        bob_c[j] = bob_c[j]^result3[j]

    return bob_c

def sort2(bob): # Our algorithm for comparison
    
    results = []

    B = np.zeros(((3*k-1)//2,2), dtype = np.int8)
    
    for i in range(k):
        B[(k-1)//2+i] = [int(bob[0][i]), int(bob[1][i])^1]
        
    for i in range((k-1)//2):
        B[i] = [int(bob[0][2*i])^int(bob[1][2*i])^1, int(bob[0][2*i+1])^int(bob[1][2*i+1])^1]
        
    results.append(mul_share(B, len(B)))
    
    bob_c = 1^results[0][(k-1)//2]
    
    save = [int(bob[0][0])^int(bob[1][0])^1, results[0][0]]

    B = np.zeros(((k-1)//2, 2), dtype = np.int8)
    
    for i in range((k-1)//4):
        B[2*i] = [results[0][2*i], int(bob[0][4*i+2])^int(bob[1][4*i+2])^1]
        B[2*i+1] = [results[0][2*i], results[0][2*i+1]]
        
    results.append(mul_share(B, len(B)))
    save.append(results[-1][0])
    save.append(results[-1][1])
    
    for j in range(1, round_num-2):

        B = np.zeros(((k-1)//2, 2), dtype = np.int8)
        
        n = 2**(j+1)
        
        index = [2**x for x in range(j+1)]
        
        for i in range((k-1)//(2*n)):
            B[n*i] = [results[-1][n*i+1], int(bob[0][n*2*i+n])^int(bob[1][n*2*i+n])^1]
            
            for x in range(len(index)):
                for y in range(int(index[x])):
                    B[n*i+2**x+y] = [results[-1][n*i+1], results[x][n*i+n//2+y]]
                    
        results.append(mul_share(B, len(B)))
        
        for i in range(n):
            save.append(results[-1][i])

    B = np.zeros((k-1, 2), dtype = np.int8)
    
    for i in range((k-1)//2):
        B[i] = [save[i], results[0][i+1+(k-1)//2]]
        B[i+(k-1)//2] = [results[-1][i], results[0][i+k]]        
        
    result = mul_share(B, len(B))
    
    for i in range(k-1):
        bob_c = bob_c^result[i]
        
    return bob_c

def pa_sort2(bob): # Our algorithm for comparison for parallel computing
    
    m = len(bob)
    
    results = []

    B = np.zeros(((3*k-1)//2*m,2), dtype = np.int8)
    
    for j in range(m):
        for i in range(k):
            B[(3*k-1)//2*j+(k-1)//2+i] = [int(bob[j][0][i]), int(bob[j][1][i])^1]
        
        for i in range((k-1)//2):
            B[(3*k-1)//2*j+i] = [int(bob[j][0][2*i])^int(bob[j][1][2*i])^1, int(bob[j][0][2*i+1])^int(bob[j][1][2*i+1])^1]

    results.append(mul_share(B, len(B)))
    
    bob_c = [1^results[0][(3*k-1)//2*j+(k-1)//2] for j in range(m)]
    
    save = []
    
    for j in range(m):
        save.append([int(bob[j][0][0])^int(bob[j][1][0])^1, results[0][(3*k-1)//2*j]])

    B = np.zeros(((k-1)//2*m, 2), dtype = np.int8)
    
    for j in range(m):
        for i in range((k-1)//4):
            B[(k-1)//2*j+2*i] = [results[0][(3*k-1)//2*j+2*i], int(bob[j][0][4*i+2])^int(bob[j][1][4*i+2])^1]
            B[(k-1)//2*j+2*i+1] = [results[0][(3*k-1)//2*j+2*i], results[0][(3*k-1)//2*j+2*i+1]]
        
    results.append(mul_share(B, len(B)))
    
    for j in range(m):
        save[j].append(results[-1][(k-1)//2*j])
        save[j].append(results[-1][(k-1)//2*j+1])
    
    for j in range(1, round_num-2):

        B = np.zeros(((k-1)//2*m, 2), dtype = np.int8)
        
        n = 2**(j+1)
        
        index = [2**x for x in range(j+1)]
        
        for z in range(m):
            for i in range((k-1)//(2*n)):

                B[(k-1)//2*z+n*i] = [results[-1][(k-1)//2*z+n*i+1], int(bob[z][0][n*2*i+n])^int(bob[z][1][n*2*i+n])^1]

                for x in range(len(index)):
                    for y in range(int(index[x])):
                        if x == 0:
                            B[(k-1)//2*z+n*i+2**x+y] = [results[-1][(k-1)//2*z+n*i+1], results[x][(3*k-1)//2*z+n*i+n//2+y]]
                        else:
                            B[(k-1)//2*z+n*i+2**x+y] = [results[-1][(k-1)//2*z+n*i+1], results[x][(k-1)//2*z+n*i+n//2+y]]

        results.append(mul_share(B, len(B)))
        
        for z in range(m):
            for i in range(n):
                save[z].append(results[-1][(k-1)//2*z+i])
            
    B = np.zeros(((k-1)*m, 2), dtype = np.int8)
    
    for j in range(m):
        for i in range(k-1):
            B[(k-1)*j+i] = [save[j][i], results[0][(3*k-1)//2*j+i+1+(k-1)//2]]   
        
    result = mul_share(B, len(B))
    
    for j in range(m):
        for i in range(k-1):
            bob_c[j] = bob_c[j]^result[(k-1)*j+i]
        
    return bob_c

def xor(a, b): # xor
    y = int(a, 2)^int(b, 2)
    return bin(y)[2:].zfill(len(a))

def swap(bob_c, bob): # swapping algorithm when the comparison result(c) is given
    A = np.zeros((2*k, 2), dtype=np.int8)
    for i in range(k):
        A[i] = [bob_c, int(bob[0][i])]
        A[i+k] = [bob_c ^ 1, int(bob[1][i])]
    result = mul_share(A, 2*k)
    string = str(result)[1:-1].replace(' ', '').replace('\n', '')
    a1 = string[:k]
    b1 = string[k:]
    a1 = xor(a1, b1)
    b1 = xor(xor(bob[0], bob[1]), a1)

    bob_new = [a1, b1]
    return bob_new

def pa_swap(bob_c, bob): # parallel swapping algorithm
    m = len(bob_c)
    A = np.zeros((2*k*m, 2), dtype = np.int8)
    for j in range(m):
        for i in range(k):
            A[2*k*j+i] = [bob_c[j], int(bob[j][0][i])]
            A[2*k*j+i+k] = [bob_c[j]^1, int(bob[j][1][i])]
    result = mul_share(A, 2*k*m)
    string = ''
    for i in range(len(result)):
        string += str(result[i])
   
    bob_new = []
    
    for j in range(m):
        a1 = string[2*k*j:2*k*j+k]
        b1 = string[2*k*j+k:2*k*(j+1)]
        a1 = xor(a1, b1)
        b1 = xor(xor(bob[j][0], bob[j][1]), a1)
        bob_new.append([a1, b1])
        
    return bob_new

def med5_v1(y):
    
    # Exact median selection algorithm of array length 5
    # Used naive comparison algorithm
    
    c1 = pa_sort1([y[0:2], y[2:4]])
    r1 = pa_swap(c1, [y[0:2], y[2:4]])
    y[0:2] = r1[0]
    y[2:4] = r1[1]

    c3 = sort1([y[0], y[2]])
    r2 = pa_swap([c3, c3], [[y[0], y[2]], y[1::2]])
    y[0], y[2] = r2[0][0], r2[0][1]
    y[1::2] = r2[1]

    c4 = sort1([y[1], y[4]])
    r2 = swap(c4, [y[1], y[4]])
    y[1], y[4] = r2[0], r2[1]

    c5 = sort1(y[1:3])
    y[1:3] = swap(c5, y[1:3])
    y[4:2:-1] = swap(c5, y[4:2:-1])

    c6 = sort1(y[2::2])
    y[2::2] = swap(c6, y[2::2])
    
    return y[2]

def med5_v2(y):
    
    # Exact median selection algorithm of array length 5
    # Used our algorithm for comparison
    
    c1 = pa_sort2([y[0:2], y[2:4]])
    r1 = pa_swap(c1, [y[0:2], y[2:4]])
    y[0:2] = r1[0]
    y[2:4] = r1[1]

    c3 = sort2([y[0], y[2]])
    r2 = pa_swap([c3, c3], [[y[0], y[2]], y[1::2]])
    y[0], y[2] = r2[0][0], r2[0][1]
    y[1::2] = r2[1]

    c4 = sort2([y[1], y[4]])
    r2 = swap(c4, [y[1], y[4]])
    y[1], y[4] = r2[0], r2[1]

    c5 = sort2(y[1:3])
    r3 = pa_swap([c5, c5], [y[1:3], y[4:2:-1]])
    y[1:3] = r3[0]
    y[4:2:-1] = r3[1]

    c6 = sort2(y[2::2])
    y[2::2] = swap(c6, y[2::2])
    
    return y[2]

def Medlist25_v2(x):
    
    # Collects medians for each chunk of size 5
    
    m = len(x)
    y = [x[5*i:5*i+5] for i in range(m//5)]
    
    # round 1
    x = []
    for i in range(m//5):
        x.append(y[i][0:2])
        x.append(y[i][2:4])
    c1 = pa_sort2(x)
    r1 = pa_swap(c1, x)
    for i in range(m//5):
        y[i][0:2] = r1[2*i]
        y[i][2:4] = r1[2*i+1]
    
    # round 2
    x = [[y[i][0], y[i][2]] for i in range(m//5)]
    c2 = pa_sort2(x)
    for i in range(m//5):
        x.append(y[i][1::2])
    r2 = pa_swap(c2+c2, x)
    for i in range(m//5):
        y[i][0], y[i][2] = r2[i][0], r2[i][1]
        y[i][1::2] = r2[i+m//5]
        
    # round 3    
    x = [[y[i][1], y[i][4]] for i in range(m//5)]
    c3 = pa_sort2(x)
    r3 = pa_swap(c3, x)
    for i in range(m//5):
        y[i][1], y[i][4] = r3[i][0], r3[i][1]   
        
    # round 4    
    x = [y[i][1:3] for i in range(m//5)]
    c4 = pa_sort2(x)
    for i in range(m//5):
        x.append(y[i][4:2:-1])
    r4 = pa_swap(c4+c4, x)
    for i in range(m//5):
        y[i][1:3] = r4[i]
        y[i][4:2:-1] = r4[m//5+i]
    
    # round 5
    x = [y[i][2::2] for i in range(m//5)]
    c5 = pa_sort2(x)
    r5 = pa_swap(c5, x)
    for i in range(m//5):
        y[i][2::2] = r5[i] 
    
    return [y[i][2] for i in range(m//5)]

def fasterMedOfMed5_v2(x):
    
    # Approximate median selectino algorithm
    # array length = 25, chunck size = 5
    
    medians = Medlist25_v2(x)
    return med5_v2(medians)

def Med3_v2(y):
    
    # Exact median selection algorithm of array length 3
    # Used our algorithm
    
    c1 = sort2(y[0:2])
    y[0:2] = swap(c1, y[0:2])
    
    c2 = sort2(y[1:3])
    y[1:3] = swap(c2, y[1:3])
    
    c3 = sort2(y[0:2])
    y[0:2] = swap(c3, y[0:2])
    
    return y[1]

def pa_Med3(x):
    
    # Collects exact medians for each chunk of size 3
    
    m = len(x)
    y = [x[3*i: 3*i+3] for i in range(m//3)]
    
    x = [y[i][0:2] for i in range(m//3)]
    c1 = pa_sort2(x)
    r1 = pa_swap(c1, x)
    for i in range(m//3):
        y[i][0:2] = r1[i]
        
    x = [y[i][1:3] for i in range(m//3)]
    c2 = pa_sort2(x)
    r2 = pa_swap(c2, x)
    for i in range(m//3):
        y[i][1:3] = r2[i]
        
    x = [y[i][0:2] for i in range(m//3)]
    c3 = pa_sort2(x)
    r3 = pa_swap(c3, x)
    for i in range(m//3):
        y[i][0:2] = r3[i]
    
    return [y[i][2] for i in range(m//3)]

def Med4_v2(y):
    
    # Exact median selection algorithm of array length 4
    # Used our algorithm
    
    c1 = pa_sort2([y[0:2], y[2:4]])
    r1 = pa_swap(c1, [y[0:2], y[2:4]])
    y[0:2] = r1[0]
    y[2:4] = r1[1]
    
    c2 = sort2(y[0::2])
    r2 = pa_swap([c2, c2], [y[0::2], y[1::2]])
    y[0::2] = r2[0]
    y[1::2] = r2[1]
    
    c3 = sort2(y[1::2])
    r3 = swap(c3, y[1::2])
    y[1::2] = r3
    
    return y[1]

def exact_median32(y):
    
    # Exact median selection of array length 32
    # Used bitonic sorting network
    
    # round 1
    x1 = [[y[2*i],y[2*i+1]] for i in range(16)]
    c1 = pa_sort2(x1)
    r1 = pa_swap(c1, x1)
    for i in range(16):
        [y[2*i],y[2*i+1]] = r1[i]

    # round 2
    x1 = [[y[4*i],y[4*i+3]] for i in range(8)]
    x2 = [[y[4*i+1],y[4*i+2]] for i in range(8)]
    c2 = pa_sort2(x1+x2)
    r2 = pa_swap(c2, x1+x2)
    for i in range(8):
        [y[4*i],y[4*i+3]] = r2[i]
        [y[4*i+1],y[4*i+2]] = r2[i+8]

    # round 3
    x1 = [[y[2*i],y[2*i+1]] for i in range(16)]
    c3 = pa_sort2(x1)
    r3 = pa_swap(c3, x1)
    for i in range(16):
        [y[2*i],y[2*i+1]] = r3[i]
 
    # round 4
    x1 = [[y[8*i],y[8*i+7]] for i in range(4)]
    x2 = [[y[8*i+1],y[8*i+6]] for i in range(4)]
    x3 = [[y[8*i+2],y[8*i+5]] for i in range(4)]
    x4 = [[y[8*i+3],y[8*i+4]] for i in range(4)]
    c4 = pa_sort2(x1+x2+x3+x4)
    r4 = pa_swap(c4, x1+x2+x3+x4)
    for i in range(4):
        [y[8*i],y[8*i+7]] = r4[i]
        [y[8*i+1],y[8*i+6]] = r4[i+4]
        [y[8*i+2],y[8*i+5]] = r4[i+8]
        [y[8*i+3],y[8*i+4]] = r4[i+12]
        
    # round 5
    x1 = [[y[4*i],y[4*i+2]] for i in range(8)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(8)]
    c5 = pa_sort2(x1+x2)
    r5 = pa_swap(c5, x1+x2)
    for i in range(8):
        [y[4*i],y[4*i+2]] = r5[i]
        [y[4*i+1],y[4*i+3]] = r5[i+8]
    
    # round 6
    x1 = [[y[2*i],y[2*i+1]] for i in range(16)]
    c6 = pa_sort2(x1)
    r6 = pa_swap(c6, x1)
    for i in range(16):
        [y[2*i],y[2*i+1]] = r6[i]
        
    # round 7
    x = []
    for j in range(8):
        for i in range(2):
            x.append([y[16*i+j],y[16*i+15-j]])
    c7 = pa_sort2(x)
    r7 = pa_swap(c7,x)
    for j in range(8):
        for i in range(2):
            [y[16*i+j],y[16*i+15-j]] = r7[2*j+i]
    
    # round 8
    x = []
    for j in range(4):
        for i in range(4):
            x.append([y[8*i+j],y[8*i+j+4]])
    c8 = pa_sort2(x)
    r8 = pa_swap(c8, x)
    for j in range(4):
        for i in range(4):
            [y[8*i+j],y[8*i+j+4]] = r8[4*j+i]

    # round 9
    x1 = [[y[4*i],y[4*i+2]] for i in range(8)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(8)]
    c9 = pa_sort2(x1+x2)
    r9 = pa_swap(c9, x1+x2)
    for i in range(8):
        [y[4*i],y[4*i+2]] = r9[i]
        [y[4*i+1],y[4*i+3]] = r9[i+8]

    # round 10
    x1 = [[y[2*i],y[2*i+1]] for i in range(16)]
    c10 = pa_sort2(x1)
    r10 = pa_swap(c10, x1)
    for i in range(16):
        [y[2*i],y[2*i+1]] = r10[i]

    # round 11
    x = []
    for j in range(16):
        x.append([y[j],y[31-j]])
    c11 = pa_sort2(x)
    r11 = pa_swap(c11,x)
    for j in range(16):
        [y[j],y[31-j]] = r11[j]
    
    # round 12
    x = []
    for j in range(8):
        for i in range(2):
            x.append([y[16*i+j],y[16*i+j+8]])
    c12 = pa_sort2(x)
    r12 = pa_swap(c12, x)
    for j in range(8):
        for i in range(2):
            [y[16*i+j],y[16*i+j+8]] = r12[2*j+i]

    # round 13
    x = []
    for j in range(4):
        for i in range(4):
            x.append([y[8*i+j],y[8*i+j+4]])
    c13 = pa_sort2(x)
    r13 = pa_swap(c13, x)
    for j in range(4):
        for i in range(4):
            [y[8*i+j],y[8*i+j+4]] = r13[4*j+i]
            
    # round 14
    x1 = [[y[4*i],y[4*i+2]] for i in range(8)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(8)]
    c14 = pa_sort2(x1+x2)
    r14 = pa_swap(c14, x1+x2)
    for i in range(8):
        [y[4*i],y[4*i+2]] = r14[i]
        [y[4*i+1],y[4*i+3]] = r14[i+8]

    # round 15
    x1 = [[y[2*i],y[2*i+1]] for i in range(16)]
    c15 = pa_sort2(x1)
    r15 = pa_swap(c15, x1)
    for i in range(16):
        [y[2*i],y[2*i+1]] = r15[i]
    
    return y[15]

def exact_median64(y):
    
    # Exact median selection of array length 64
    # Used bitonic sorting network
    
    # round 1
    x1 = [[y[2*i],y[2*i+1]] for i in range(32)]
    c1 = pa_sort2(x1)
    r1 = pa_swap(c1, x1)
    for i in range(32):
        [y[2*i],y[2*i+1]] = r1[i]

    # round 2
    x1 = [[y[4*i],y[4*i+3]] for i in range(16)]
    x2 = [[y[4*i+1],y[4*i+2]] for i in range(16)]
    c2 = pa_sort2(x1+x2)
    r2 = pa_swap(c2, x1+x2)
    for i in range(16):
        [y[4*i],y[4*i+3]] = r2[i]
        [y[4*i+1],y[4*i+2]] = r2[i+16]

    # round 3
    x1 = [[y[2*i],y[2*i+1]] for i in range(32)]
    c3 = pa_sort2(x1)
    r3 = pa_swap(c3, x1)
    for i in range(32):
        [y[2*i],y[2*i+1]] = r3[i]
 
    # round 4
    x1 = [[y[8*i],y[8*i+7]] for i in range(8)]
    x2 = [[y[8*i+1],y[8*i+6]] for i in range(8)]
    x3 = [[y[8*i+2],y[8*i+5]] for i in range(8)]
    x4 = [[y[8*i+3],y[8*i+4]] for i in range(8)]
    c4 = pa_sort2(x1+x2+x3+x4)
    r4 = pa_swap(c4, x1+x2+x3+x4)
    for i in range(8):
        [y[8*i],y[8*i+7]] = r4[i]
        [y[8*i+1],y[8*i+6]] = r4[i+8]
        [y[8*i+2],y[8*i+5]] = r4[i+16]
        [y[8*i+3],y[8*i+4]] = r4[i+24]
        
    # round 5
    x1 = [[y[4*i],y[4*i+2]] for i in range(16)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(16)]
    c5 = pa_sort2(x1+x2)
    r5 = pa_swap(c5, x1+x2)
    for i in range(16):
        [y[4*i],y[4*i+2]] = r5[i]
        [y[4*i+1],y[4*i+3]] = r5[i+16]
    
    # round 6
    x1 = [[y[2*i],y[2*i+1]] for i in range(32)]
    c6 = pa_sort2(x1)
    r6 = pa_swap(c6, x1)
    for i in range(32):
        [y[2*i],y[2*i+1]] = r6[i]
        
    # round 7
    x = []
    for j in range(8):
        for i in range(4):
            x.append([y[16*i+j],y[16*i+15-j]])
    c7 = pa_sort2(x)
    r7 = pa_swap(c7,x)
    for j in range(8):
        for i in range(4):
            [y[16*i+j],y[16*i+15-j]] = r7[4*j+i]
    
    # round 8
    x = []
    for j in range(4):
        for i in range(8):
            x.append([y[8*i+j],y[8*i+j+4]])
    c8 = pa_sort2(x)
    r8 = pa_swap(c8, x)
    for j in range(4):
        for i in range(8):
            [y[8*i+j],y[8*i+j+4]] = r8[8*j+i]

    # round 9
    x1 = [[y[4*i],y[4*i+2]] for i in range(16)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(16)]
    c9 = pa_sort2(x1+x2)
    r9 = pa_swap(c9, x1+x2)
    for i in range(16):
        [y[4*i],y[4*i+2]] = r9[i]
        [y[4*i+1],y[4*i+3]] = r9[i+16]

    # round 10
    x1 = [[y[2*i],y[2*i+1]] for i in range(32)]
    c10 = pa_sort2(x1)
    r10 = pa_swap(c10, x1)
    for i in range(32):
        [y[2*i],y[2*i+1]] = r10[i]

    # round 11
    x = []
    for j in range(16):
        for i in range(2):
            x.append([y[32*i+j],y[32*i+31-j]])
    c11 = pa_sort2(x)
    r11 = pa_swap(c11,x)
    for j in range(16):
        for i in range(2):
            [y[32*i+j],y[32*i+31-j]] = r11[2*j+i]
    
    # round 12
    x = []
    for j in range(8):
        for i in range(4):
            x.append([y[16*i+j],y[16*i+j+8]])
    c12 = pa_sort2(x)
    r12 = pa_swap(c12, x)
    for j in range(8):
        for i in range(4):
            [y[16*i+j],y[16*i+j+8]] = r12[4*j+i]

    # round 13
    x = []
    for j in range(4):
        for i in range(8):
            x.append([y[8*i+j],y[8*i+j+4]])
    c13 = pa_sort2(x)
    r13 = pa_swap(c13, x)
    for j in range(4):
        for i in range(8):
            [y[8*i+j],y[8*i+j+4]] = r13[8*j+i]
            
    # round 14
    x1 = [[y[4*i],y[4*i+2]] for i in range(16)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(16)]
    c14 = pa_sort2(x1+x2)
    r14 = pa_swap(c14, x1+x2)
    for i in range(16):
        [y[4*i],y[4*i+2]] = r14[i]
        [y[4*i+1],y[4*i+3]] = r14[i+16]

    # round 15
    x1 = [[y[2*i],y[2*i+1]] for i in range(32)]
    c15 = pa_sort2(x1)
    r15 = pa_swap(c15, x1)
    for i in range(32):
        [y[2*i],y[2*i+1]] = r15[i]
        
    # round 16
    x = []
    for j in range(32):
        x.append([y[j],y[63-j]])
    c16 = pa_sort2(x)
    r16 = pa_swap(c16,x)
    for j in range(32):
        [y[j],y[63-j]] = r16[j]

    # round 17
    x = []
    for j in range(16):
        for i in range(2):
            x.append([y[32*i+j],y[32*i+j+16]])
    c16 = pa_sort2(x)
    r16 = pa_swap(c16, x)
    for j in range(16):
        for i in range(2):
            [y[32*i+j],y[32*i+j+16]] = r12[2*j+i]
        
    # round 18
    x = []
    for j in range(8):
        for i in range(4):
            x.append([y[16*i+j],y[16*i+j+8]])
    c12 = pa_sort2(x)
    r12 = pa_swap(c12, x)
    for j in range(8):
        for i in range(4):
            [y[16*i+j],y[16*i+j+8]] = r12[4*j+i]
    
    # round 19
    x = []
    for j in range(4):
        for i in range(8):
            x.append([y[8*i+j],y[8*i+j+4]])
    c13 = pa_sort2(x)
    r13 = pa_swap(c13, x)
    for j in range(4):
        for i in range(8):
            [y[8*i+j],y[8*i+j+4]] = r13[8*j+i]
            
    # round 20
    x1 = [[y[4*i],y[4*i+2]] for i in range(16)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(16)]
    c14 = pa_sort2(x1+x2)
    r14 = pa_swap(c14, x1+x2)
    for i in range(16):
        [y[4*i],y[4*i+2]] = r14[i]
        [y[4*i+1],y[4*i+3]] = r14[i+16]

    # round 21
    x1 = [[y[2*i],y[2*i+1]] for i in range(32)]
    c15 = pa_sort2(x1)
    r15 = pa_swap(c15, x1)
    for i in range(32):
        [y[2*i],y[2*i+1]] = r15[i]
    
    return y[31]

def approxmed32(y):
    
    # Approximate median selection algorithm of array length 32, chunk size 5
    
    y1 = y[:30]
    r1 = Medlist25_v2(y1)
    r1.append(y[30])
    r2 = med5_v2(r1[:5])
    return Med3_v2([r2, r1[5], r1[6]])

def approxmed64(y):
    
    # Approximate median selection algorithm of array length 64, chunk size 5
    
    r1 = Medlist25_v2(y)
    y1 = Med4_v2(y[60:64])
    r1.append(y1)
    
    r2 = Medlist25_v2(r1)
    y2 = Med3_v2(r1[10:13])
    r2.append(y2)
    
    return Med3_v2(r2)

def approxmed128(y):
    
    # Approximate median selection algorithm of array length 128, chunk size 5    
    
    r1 = Medlist25_v2(y)
    y1 = Med3_v2(y[125:128])
    r1.append(y1)
    
    r2 = Medlist25_v2(r1)
    
    r3 = med5_v2(r2)
    
    return r3

def exact_median128(y):
    
    # Exact median selection of array length 64
    # Used bitonic sorting network
    
    # round 1
    x1 = [[y[2*i],y[2*i+1]] for i in range(64)]
    c1 = pa_sort2(x1)
    r1 = pa_swap(c1, x1)
    for i in range(64):
        [y[2*i],y[2*i+1]] = r1[i]

    # round 2
    x1 = [[y[4*i],y[4*i+3]] for i in range(32)]
    x2 = [[y[4*i+1],y[4*i+2]] for i in range(32)]
    c2 = pa_sort2(x1+x2)
    r2 = pa_swap(c2, x1+x2)
    for i in range(32):
        [y[4*i],y[4*i+3]] = r2[i]
        [y[4*i+1],y[4*i+2]] = r2[i+32]

    # round 3
    x1 = [[y[2*i],y[2*i+1]] for i in range(64)]
    c3 = pa_sort2(x1)
    r3 = pa_swap(c3, x1)
    for i in range(64):
        [y[2*i],y[2*i+1]] = r3[i]
 
    # round 4
    x1 = [[y[8*i],y[8*i+7]] for i in range(16)]
    x2 = [[y[8*i+1],y[8*i+6]] for i in range(16)]
    x3 = [[y[8*i+2],y[8*i+5]] for i in range(16)]
    x4 = [[y[8*i+3],y[8*i+4]] for i in range(16)]
    c4 = pa_sort2(x1+x2+x3+x4)
    r4 = pa_swap(c4, x1+x2+x3+x4)
    for i in range(16):
        [y[8*i],y[8*i+7]] = r4[i]
        [y[8*i+1],y[8*i+6]] = r4[i+16]
        [y[8*i+2],y[8*i+5]] = r4[i+32]
        [y[8*i+3],y[8*i+4]] = r4[i+48]
        
    # round 5
    x1 = [[y[4*i],y[4*i+2]] for i in range(32)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(32)]
    c5 = pa_sort2(x1+x2)
    r5 = pa_swap(c5, x1+x2)
    for i in range(32):
        [y[4*i],y[4*i+2]] = r5[i]
        [y[4*i+1],y[4*i+3]] = r5[i+32]
    
    # round 6
    x1 = [[y[2*i],y[2*i+1]] for i in range(64)]
    c6 = pa_sort2(x1)
    r6 = pa_swap(c6, x1)
    for i in range(64):
        [y[2*i],y[2*i+1]] = r6[i]
        
    # round 7
    x = []
    for j in range(8):
        for i in range(8):
            x.append([y[16*i+j],y[16*i+15-j]])
    c7 = pa_sort2(x)
    r7 = pa_swap(c7,x)
    for j in range(8):
        for i in range(8):
            [y[16*i+j],y[16*i+15-j]] = r7[8*j+i]
    
    # round 8
    x = []
    for j in range(4):
        for i in range(16):
            x.append([y[8*i+j],y[8*i+j+4]])
    c8 = pa_sort2(x)
    r8 = pa_swap(c8, x)
    for j in range(4):
        for i in range(16):
            [y[8*i+j],y[8*i+j+4]] = r8[16*j+i]

    # round 9
    x1 = [[y[4*i],y[4*i+2]] for i in range(32)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(32)]
    c9 = pa_sort2(x1+x2)
    r9 = pa_swap(c9, x1+x2)
    for i in range(32):
        [y[4*i],y[4*i+2]] = r9[i]
        [y[4*i+1],y[4*i+3]] = r9[i+32]

    # round 10
    x1 = [[y[2*i],y[2*i+1]] for i in range(64)]
    c10 = pa_sort2(x1)
    r10 = pa_swap(c10, x1)
    for i in range(64):
        [y[2*i],y[2*i+1]] = r10[i]

    # round 11
    x = []
    for j in range(16):
        for i in range(4):
            x.append([y[32*i+j],y[32*i+31-j]])
    c11 = pa_sort2(x)
    r11 = pa_swap(c11,x)
    for j in range(16):
        for i in range(4):
            [y[32*i+j],y[32*i+31-j]] = r11[4*j+i]
    
    # round 12
    x = []
    for j in range(8):
        for i in range(8):
            x.append([y[16*i+j],y[16*i+j+8]])
    c12 = pa_sort2(x)
    r12 = pa_swap(c12, x)
    for j in range(8):
        for i in range(8):
            [y[16*i+j],y[16*i+j+8]] = r12[8*j+i]

    # round 13
    x = []
    for j in range(4):
        for i in range(16):
            x.append([y[8*i+j],y[8*i+j+4]])
    c13 = pa_sort2(x)
    r13 = pa_swap(c13, x)
    for j in range(4):
        for i in range(16):
            [y[8*i+j],y[8*i+j+4]] = r13[16*j+i]
            
    # round 14
    x1 = [[y[4*i],y[4*i+2]] for i in range(32)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(32)]
    c14 = pa_sort2(x1+x2)
    r14 = pa_swap(c14, x1+x2)
    for i in range(32):
        [y[4*i],y[4*i+2]] = r14[i]
        [y[4*i+1],y[4*i+3]] = r14[i+32]

    # round 15
    x1 = [[y[2*i],y[2*i+1]] for i in range(64)]
    c15 = pa_sort2(x1)
    r15 = pa_swap(c15, x1)
    for i in range(64):
        [y[2*i],y[2*i+1]] = r15[i]
        
    # round 16
    x = []
    for j in range(32):
        for i in range(2):
            x.append([y[64*i+j],y[64*i+63-j]])
    c16 = pa_sort2(x)
    r16 = pa_swap(c16,x)
    for j in range(32):
        for i in range(2):
            [y[64*i+j],y[64*i+63-j]] = r16[2*j+i]

    # round 17
    x = []
    for j in range(16):
        for i in range(4):
            x.append([y[32*i+j],y[32*i+j+16]])
    c16 = pa_sort2(x)
    r16 = pa_swap(c16, x)
    for j in range(16):
        for i in range(4):
            [y[32*i+j],y[32*i+j+16]] = r12[4*j+i]
        
    # round 18
    x = []
    for j in range(8):
        for i in range(8):
            x.append([y[16*i+j],y[16*i+j+8]])
    c12 = pa_sort2(x)
    r12 = pa_swap(c12, x)
    for j in range(8):
        for i in range(8):
            [y[16*i+j],y[16*i+j+8]] = r12[8*j+i]
    
    # round 19
    x = []
    for j in range(4):
        for i in range(16):
            x.append([y[8*i+j],y[8*i+j+4]])
    c13 = pa_sort2(x)
    r13 = pa_swap(c13, x)
    for j in range(4):
        for i in range(16):
            [y[8*i+j],y[8*i+j+4]] = r13[16*j+i]
            
    # round 20
    x1 = [[y[4*i],y[4*i+2]] for i in range(32)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(32)]
    c14 = pa_sort2(x1+x2)
    r14 = pa_swap(c14, x1+x2)
    for i in range(32):
        [y[4*i],y[4*i+2]] = r14[i]
        [y[4*i+1],y[4*i+3]] = r14[i+32]

    # round 21
    x1 = [[y[2*i],y[2*i+1]] for i in range(64)]
    c15 = pa_sort2(x1)
    r15 = pa_swap(c15, x1)
    for i in range(64):
        [y[2*i],y[2*i+1]] = r15[i]
        
    # round 22
    x = []
    for j in range(64):
        x.append([y[j], y[127-j]])
    c22 = pa_sort2(x)
    r22 = pa_swap(c22, x)
    for j in range(64):
        [y[j], y[127-j]] = r22[j]

    # round 23
    x = []
    for j in range(32):
        for i in range(2):
            x.append([y[64*i+j],y[64*i+j+32]])
    c16 = pa_sort2(x)
    r16 = pa_swap(c16, x)
    for j in range(32):
        for i in range(2):
            [y[64*i+j],y[64*i+j+32]] = r16[2*j+i]        
            
    # round 24
    x = []
    for j in range(16):
        for i in range(4):
            x.append([y[32*i+j],y[32*i+j+16]])
    c16 = pa_sort2(x)
    r16 = pa_swap(c16, x)
    for j in range(16):
        for i in range(4):
            [y[32*i+j],y[32*i+j+16]] = r12[4*j+i]
        
    # round 25
    x = []
    for j in range(8):
        for i in range(8):
            x.append([y[16*i+j],y[16*i+j+8]])
    c12 = pa_sort2(x)
    r12 = pa_swap(c12, x)
    for j in range(8):
        for i in range(8):
            [y[16*i+j],y[16*i+j+8]] = r12[8*j+i]
    
    # round 26
    x = []
    for j in range(4):
        for i in range(16):
            x.append([y[8*i+j],y[8*i+j+4]])
    c13 = pa_sort2(x)
    r13 = pa_swap(c13, x)
    for j in range(4):
        for i in range(16):
            [y[8*i+j],y[8*i+j+4]] = r13[16*j+i]
            
    # round 27
    x1 = [[y[4*i],y[4*i+2]] for i in range(32)]
    x2 = [[y[4*i+1],y[4*i+3]] for i in range(32)]
    c14 = pa_sort2(x1+x2)
    r14 = pa_swap(c14, x1+x2)
    for i in range(32):
        [y[4*i],y[4*i+2]] = r14[i]
        [y[4*i+1],y[4*i+3]] = r14[i+32]

    # round 28
    x1 = [[y[2*i],y[2*i+1]] for i in range(64)]
    c15 = pa_sort2(x1)
    r15 = pa_swap(c15, x1)
    for i in range(64):
        [y[2*i],y[2*i+1]] = r15[i]
    
    return y[63]

def approxmed125(y):
    
    # Approximate median selection of array length 125, chunk size 5
    
    y1 = Medlist25_v2(y)
    return fasterMedOfMed5_v2(y1)

# Experiment
for i in range(5):
    print(approxmed125(bo1))
    
