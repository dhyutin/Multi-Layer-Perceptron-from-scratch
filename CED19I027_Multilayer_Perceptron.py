# Multilayer Perceptron
import numpy as np
import matplotlib.pyplot as plt
import math

# Input
X = [1, 0.7, 1.2]

# Weight Vector 
W = [[[0.5, 1.5, 0.8], [0.8, 0.2, -1.6]], 
    [[0.9, -1.7, 1.6], [1.2, 2.1, -0.2]]]

# Target classes
t = [1, 0]

# Sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def matrix_multipy(X, Y):

    result = [0, 0]
    for i in range(2):
        for j in range(3):
            result[i] += X[i][j]*Y[i]

    return result
looper = 500
iter_num = []
loss = []

for i in range(looper):

    # Step 1: Feed Forward Pass - Layer 0 to 1
    X_1 = matrix_multipy(W[0], X)
    X_1[0] = sigmoid(X_1[0])
    X_1[1] = sigmoid(X_1[1])

    # Step 2: Layer 1 to 2
    onee = [1]
    X_2 = matrix_multipy(W[1], onee+X_1)
    X_2[0] = sigmoid(X_2[0])
    X_2[1] = sigmoid(X_2[1])


    # Step 3: error 'E'
    d12 = (X_2[0] - t[0])*(X_2[0])*(1 - X_2[0])
    d22 = (X_2[1] - t[1])*(X_2[1])*(1 - X_2[0])

    dedw012 = d12*X[0]
    dedw022 = d22*X[0]

    dedw112 = d12*X[1]
    dedw122 = d22*X[1]

    dedw212 = d12*X[2]
    dedw222 = d22*X[2]


    # Step 4: Back progragation from K-1 to k-2 layer
    d11 = X_1[0]*(1 - X_1[0])*((dedw012*W[1][0][1]) + (dedw022*W[1][1][1]))
    d21 = X_1[1]*(1 - X_1[1])*((dedw012*W[1][0][1]) + (dedw022*W[1][1][1]))

    dedw011 = d11*1
    dedw021 = d21*1

    dedw111 = d11*X[0]
    dedw121 = d21*X[0]

    dedw211 = d11*X[1]
    dedw221 = d21*X[1]


    #update weights
    def update_wt(wold, dedw):
        w_new = wold - ((0.5)*dedw)
        return w_new


    W[0][0][0] = update_wt(W[0][0][0], dedw011)
    W[1][0][0] = update_wt(W[1][0][0], dedw012)
    W[0][0][1] = update_wt(W[0][0][1], dedw111)
    W[1][0][1] = update_wt(W[1][0][1], dedw112)
    W[0][0][2] = update_wt(W[0][0][2], dedw211)
    W[1][0][2] = update_wt(W[1][0][2], dedw212)

    W[0][1][0] = update_wt(W[0][1][0], dedw021)
    W[1][1][0] = update_wt(W[1][1][0], dedw022)
    W[0][1][1] = update_wt(W[0][1][1], dedw121)
    W[1][1][1] = update_wt(W[1][1][1], dedw122)
    W[0][1][2] = update_wt(W[0][1][2], dedw221)
    W[1][1][2] = update_wt(W[1][1][2], dedw222)

    print("Iteration No - " , (i+1))
    print(" Updated Weight Vector:")
    print(W[0])
    print(W[1])

    print("Updated Output")
    print(X_2[0], X_2[1])
    #print(X_2[1])
    print("\n")

    iter_num.append(i+1)
    losss = math.sqrt(((t[0] - X_2[0])**2) + ((t[1] - X_2[1])**2))
    loss.append(losss)


print("After all iterations:")
print("O1: ", X_2[0])
print("O2: ", X_2[1])
plt.plot(iter_num, loss)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.show()


