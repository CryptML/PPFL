import socket
import numpy as np
import time
import random
import copy
from joblib import Parallel, delayed

k = 33
round_num = 6
al1 = ['010010100100001100101001101011010', '011111010010011101110100111100010', '001100010110110110001100110000010', '111100001010101010001010100101110', '110001110010010101110110101111111', '010000001011110110111011111011000', '001111011100101110100000100110011', '111000010001110100011010110010001', '100000010000110000011100111000111', '010101111101110100110110110100100', '111001011100101110011100111110101', '111111001101010100001100001001010', '000111101110010100001011111010010', '111011101100101010100100010101000', '111001110001100011010101000000000', '001111000001010110101001101010100', '001010110111111001110101010010010', '110000011111010100011111100011010', '000001100111111010001010111000010', '111000010100010001010001010011001', '000010001100000101111000010110111', '111111010111111010111100110111111', '000101111110111110110010010000011', '010010011100100101110010101111001', '010000100101001010111111000111011', '000100100010111001101110010011111', '001001010000010100100111101011100', '010111010111000011010110011010100', '111011111111011100001001100010100', '000011110110111001000111110110101', '110110101010001111111101111011001', '100111010110011011100101100011000', '100101111011010101011010010100000', '100011110010011011000111111110101', '101010101100000001000101110100111', '010001001011000111011110000110010', '101011010001001101001100011100001', '101000011010111000111010111011000', '101011001011001001101111000111010', '111011111011101001110000100011110', '101110101001101101000001101111001', '100100000100000001101011101101100', '010110111110000111111111110111110', '000100100101001101011111011010010', '110010010001000100101000110001100', '011101100011101001011010111000010', '010011111101010101101010110101101', '100010111101011111100101111000010', '101110011111110000110100101011011', '110101011011001010011010110000100', '110000111001011100111010011100100', '111111011010110001001000000101010', '011111011111100100001011001100000', '111111110111000110010110111110110', '110110000111111010101110000000110', '110010100001101000111010000101100', '111111110011010100111111101111011', '001011010101101111011010010010011', '101110001011010010001011011100100', '000101011011010111101101101100110', '110011101001101011000011111000001', '101100100010111000000110111101101', '101100011111010101101010111111101', '000011010010101011011011110011100', '011001001010101010111000000111111', '011010001001100111101101000001101', '110010010110011001010111001001000', '011100000010111110000000000010100', '101101011011001110011101011010110', '010010111110001111001101011100011', '011111110101101001101100110000011', '011001100100011101101101101111010', '000111111100001000000010111101000', '011011110111100001110000001001001', '111100010101110100010000111111010', '111111111110010000101100111000001', '111110001000101110011101011001010', '011101010111110110000001001001001', '011010101001001011000100100010000', '111110000101011101101111110011010', '111000100010101010011101010011100', '011111101011010010000111110000100', '011001101110101001101101000010011', '110011000010101010011110011100011', '110110111000000000100001001001100', '011000010100010110000000110001101', '001100111001111010110001010000011', '010111101010100001010010011101011', '011001001100101001110101000011001', '011001100000011110000111000000101', '000101010101011101110101100110011', '110100001001110101000110100001100', '110000010101101110100101000010110', '011010110101001111110011010101111', '111010101110111001011111001101000', '010010110000001001110111111001110', '000001010000100001001100001101111', '001001010101111000001100101000101', '010011101000110000110100011010111', '100010000100001000110100110011010', '001100101010000100000111001111011', '010001110001001111011111000001001', '100100111110111001101100000100001', '011110111100010111110110001101101', '111011100010100110000010100000101', '000001011011001011101100101100110', '110110000111101010011110010110100', '111001111111110100100010001101111', '011010000000110011111111001001100', '001100000110000100101100101010001', '110110100001000000011010000000101', '001011001101100101101000001100011', '100111100111111101001100010010001', '101101110101000000000011110100100', '011001110010110110101110001001110', '100110001101011011001000101001011', '111101010001101001011000111101001', '010100110011111011110001101100100', '000110111100011000000111110000110', '110001010000100100111101001000000', '000011111110110001001000110111001', '111100110001111110001100100001000', '100110000111111010011001111100010', '101111010010001000011000000110000', '011110110011010111110010101100100']

def mul_share(a, bit):
    m = np.random.randint(2, size = (bit, 2))
    result = np.zeros(bit, dtype = np.int8)
    choice = np.array([], dtype = np.int8)
    for i in range(bit):
        result[i] = (a[i][0]&a[i][1])^m[i][0]^m[i][1]
        choice = np.append(choice, [m[i][0], m[i][0]^a[i][0], m[i][1], m[i][1]^a[i][1]])
    
    choice1 = '1'
    # choice = str(choice)[1:-1].replace('\n','')
    for i in range(len(choice)):
        choice1 += str(choice[i])
    choice = hex(int(choice1, 2))[2:]
    client = socket.socket()
    c_name = socket.gethostname()
    
    client.connect((c_name, 8080))
    client.send(bytes(choice,'utf-8'))

    return result

def sort1(alice):
    
    A = np.zeros((k+1,2), dtype = np.int8)
    
    for i in range(k):
        A[i] = [int(alice[0][i]), int(alice[1][i])]
    
    A[k] = [int(alice[0][0])^int(alice[1][0]), int(alice[0][1])^int(alice[1][1])]
    
    result1 = mul_share(A, k+1)

    alice_c = result1[0]
    
    A = np.array([[int(alice[0][0])^int(alice[1][0]), result1[1]],
                  [result1[-1], result1[2]],
                  [result1[-1], int(alice[0][2])^int(alice[1][2])]])
    
    result2 = mul_share(A, 3)
    
    alice_c = alice_c^result2[0]^result2[1]

    for i in range(3, k-1):
        A = np.array([[result2[-1], result1[i]],
                      [result2[-1], int(alice[0][i])^int(alice[1][i])]])
        
        result2 = mul_share(A, 2)
        
        alice_c = alice_c^result2[0]
        
    A = np.array([[result2[-1], result1[k-1]]])
    
    result3 = mul_share(A, 1)
    
    alice_c = alice_c^result3[0]
    
    return alice_c

def pa_sort1(alice):
    
    m = len(alice)
    
    A = np.zeros(((k+1)*m,2), dtype = np.int8)
    
    for j in range(m):
        for i in range(k):
            A[(k+1)*j+i] = [int(alice[j][0][i]), int(alice[j][1][i])]
    
        A[(k+1)*j+k] = [int(alice[j][0][0])^int(alice[j][1][0]), int(alice[j][0][1])^int(alice[j][1][1])]
    
    result1 = mul_share(A, len(A))

    alice_c = [result1[(k+1)*j] for j in range(m)]
    
    A = np.zeros((3*m,2), dtype = np.int8) 
    
    for j in range(m):
        A[3*j:3*j+3] = np.array([[int(alice[j][0][0])^int(alice[j][1][0]), result1[(k+1)*j+1]],
                                 [result1[(k+1)*(j+1)-1], result1[(k+1)*j+2]],
                                 [result1[(k+1)*(j+1)-1], int(alice[j][0][2])^int(alice[j][1][2])]])

    result2 = mul_share(A, len(A))
    
    for j in range(m):
        alice_c[j] = alice_c[j]^result2[3*j]^result2[3*j+1]
    
    for i in range(3, k-1):
        
        A = np.zeros((2*m,2), dtype = np.int8)
        
        for j in range(m):
            if i == 3:
                A[2*j:2*j+2] = np.array([[result2[3*(j+1)-1], result1[(k+1)*j+i]],
                                         [result2[3*(j+1)-1], int(alice[j][0][i])^int(alice[j][1][i])]])
                
            else:
                A[2*j:2*j+2] = np.array([[result2[2*(j+1)-1], result1[(k+1)*j+i]],
                                         [result2[2*(j+1)-1], int(alice[j][0][i])^int(alice[j][1][i])]])
        
        result2 = mul_share(A, len(A))
        
        for j in range(m):
            alice_c[j] = alice_c[j]^result2[2*j]
        
    A = np.zeros((m,2), dtype = np.int8)
    
    for j in range(m):
        A[j] = np.array([result2[2*(j+1)-1], result1[(k+1)*j+k-1]]) 
    
    result3 = mul_share(A, len(A))
    
    for j in range(m):
        alice_c[j] = alice_c[j]^result3[j]
    
    return alice_c

def sort2(alice):
    
    results = []
    
    A = np.zeros(((3*k-1)//2,2), dtype = np.int8)
    
    for i in range(k):
        A[(k-1)//2+i] = [int(alice[0][i]), int(alice[1][i])]
        
    for i in range((k-1)//2):
        A[i] = [int(alice[0][2*i])^int(alice[1][2*i]), int(alice[0][2*i+1])^int(alice[1][2*i+1])]
        
    results.append(mul_share(A, len(A)))
    
    alice_c = results[0][(k-1)//2]
    
    save = [int(alice[0][0])^int(alice[1][0]), results[0][0]]
    
    A = np.zeros(((k-1)//2, 2), dtype = np.int8)
    
    for i in range((k-1)//4):
        A[2*i] = [results[0][2*i], int(alice[0][4*i+2])^int(alice[1][4*i+2])]
        A[2*i+1] = [results[0][2*i], results[0][2*i+1]]
        
    results.append(mul_share(A, len(A)))
    save.append(results[-1][0])
    save.append(results[-1][1])
    
    for j in range(1, round_num-2):
        
        A = np.zeros(((k-1)//2, 2), dtype = np.int8)

        n = 2**(j+1)
        
        index = [2**x for x in range(j+1)]
        
        for i in range((k-1)//(2*n)):
            
            A[n*i] = [results[-1][n*i+1], int(alice[0][n*2*i+n])^int(alice[1][n*2*i+n])]
            
            for x in range(len(index)):
                for y in range(int(index[x])):
                    A[n*i+2**x+y] = [results[-1][n*i+1], results[x][n*i+n//2+y]]
                    
        results.append(mul_share(A, len(A)))
        
        for i in range(n):
            save.append(results[-1][i])
    
    A = np.zeros((k-1, 2), dtype = np.int8)
    
    for i in range((k-1)//2):
        A[i] = [save[i], results[0][i+1+(k-1)//2]]
        A[i+(k-1)//2] = [results[-1][i], results[0][i+k]]      
        
    result = mul_share(A, len(A))
    
    for i in range(k-1):
        alice_c = alice_c^result[i]
        
    return alice_c

def pa_sort2(alice):
    
    m = len(alice)
    
    results = []
    
    A = np.zeros(((3*k-1)//2*m,2), dtype = np.int8)
    
    for j in range(m):
        for i in range(k):
            A[(3*k-1)//2*j+(k-1)//2+i] = [int(alice[j][0][i]), int(alice[j][1][i])]
        
        for i in range((k-1)//2):
            A[(3*k-1)//2*j+i] = [int(alice[j][0][2*i])^int(alice[j][1][2*i]), int(alice[j][0][2*i+1])^int(alice[j][1][2*i+1])]

    results.append(mul_share(A, len(A)))
    
    alice_c = [results[0][(3*k-1)//2*j+(k-1)//2] for j in range(m)]
    
    save = []
    
    for j in range(m):
        save.append([int(alice[j][0][0])^int(alice[j][1][0]), results[0][(3*k-1)//2*j]])
    
    A = np.zeros(((k-1)//2*m, 2), dtype = np.int8)
    
    for j in range(m):
        for i in range((k-1)//4):
            A[(k-1)//2*j+2*i] = [results[0][(3*k-1)//2*j+2*i], int(alice[j][0][4*i+2])^int(alice[j][1][4*i+2])]
            A[(k-1)//2*j+2*i+1] = [results[0][(3*k-1)//2*j+2*i], results[0][(3*k-1)//2*j+2*i+1]]
        
    results.append(mul_share(A, len(A)))
    
    for j in range(m):
        save[j].append(results[-1][(k-1)//2*j])
        save[j].append(results[-1][(k-1)//2*j+1])
    
    for j in range(1, round_num-2):
        
        A = np.zeros(((k-1)//2*m, 2), dtype = np.int8)
        
        n = 2**(j+1)
        
        index = [2**x for x in range(j+1)]
        
        for z in range(m):
            for i in range((k-1)//(2*n)):

                A[(k-1)//2*z+n*i] = [results[-1][(k-1)//2*z+n*i+1], int(alice[z][0][n*2*i+n])^int(alice[z][1][n*2*i+n])]

                for x in range(len(index)):
                    for y in range(int(index[x])):
                        if x == 0:
                            A[(k-1)//2*z+n*i+2**x+y] = [results[-1][(k-1)//2*z+n*i+1], results[x][(3*k-1)//2*z+n*i+n//2+y]]
                        else:
                            A[(k-1)//2*z+n*i+2**x+y] = [results[-1][(k-1)//2*z+n*i+1], results[x][(k-1)//2*z+n*i+n//2+y]]

        results.append(mul_share(A, len(A)))
        
        for z in range(m):
            for i in range(n):
                save[z].append(results[-1][(k-1)//2*z+i])
            
    A = np.zeros(((k-1)*m, 2), dtype = np.int8)
    
    for j in range(m):
        for i in range(k-1):
            A[(k-1)*j+i] = [save[j][i], results[0][(3*k-1)//2*j+i+1+(k-1)//2]]
        
    result = mul_share(A, len(A))
    
    for j in range(m):
        for i in range(k-1):
            alice_c[j] = alice_c[j]^result[(k-1)*j+i]
        
    return alice_c

def xor(a, b):
    y = int(a, 2)^int(b, 2)
    return bin(y)[2:].zfill(len(a))

def swap(alice_c, alice):
    A = np.zeros((2*k, 2), dtype = np.int8)
    for i in range(k):
        A[i] = [alice_c, int(alice[0][i])]
        A[i+k] = [alice_c, int(alice[1][i])]
    result = mul_share(A, 2*k)
    string = str(result)[1:-1].replace(' ','').replace('\n','')
    a1 = string[:k]
    b1 = string[k:] 
    a1 = xor(a1, b1)
    b1 = xor(xor(alice[0], alice[1]), a1)
    
    alice_new = [a1, b1]
    return alice_new

def pa_swap(bob_c, bob):
    m = len(bob_c)
    A = np.zeros((2*k*m, 2), dtype = np.int8)
    for j in range(m):
        for i in range(k):
            A[2*k*j+i] = [bob_c[j], int(bob[j][0][i])]
            A[2*k*j+i+k] = [bob_c[j], int(bob[j][1][i])]
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

def MedOfMed5_v2(x):
    procedure = [x[0:5], x[5:10], x[10:15], x[15:20], x[20:25]]
    medians = []
    for i in range(5):
        medians.append(med5_v2(procedure[i]))
    return med5_v2(medians)

def Medlist25_v2(x):
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
    medians = Medlist25_v2(x)
    return med5_v2(medians)

def Med3_v2(y):
    c1 = sort2(y[0:2])
    y[0:2] = swap(c1, y[0:2])
    
    c2 = sort2(y[1:3])
    y[1:3] = swap(c2, y[1:3])
    
    c3 = sort2(y[0:2])
    y[0:2] = swap(c3, y[0:2])
    
    return y[1]

def pa_Med3(x):
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
    y1 = y[:30]
    r1 = Medlist25_v2(y1)
    r1.append(y[30])
    r2 = med5_v2(r1[:5])
    return Med3_v2([r2, r1[5], r1[6]])

def approxmed64(y):
    r1 = Medlist25_v2(y)
    y1 = Med4_v2(y[60:64])
    r1.append(y1)
    
    r2 = Medlist25_v2(r1)
    y2 = Med3_v2(r1[10:13])
    r2.append(y2)
    
    return Med3_v2(r2)

def approxmed128(y):
    r1 = Medlist25_v2(y)
    y1 = Med3_v2(y[125:128])
    r1.append(y1)
    
    r2 = Medlist25_v2(r1)
    
    r3 = med5_v2(r2)
    
    return r3

def exact_median128(y):
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
    y1 = Medlist25_v2(y)
    return fasterMedOfMed5_v2(y1)

start = time.time()  
for i in range(5):
    print(approxmed125(al1))
print(time.time()-start)








   