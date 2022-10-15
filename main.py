import matplotlib.pyplot as plt
import numpy as np
import timeit
import numpy.ma as ma
from numba import guvectorize, float64, float32, void, njit, cuda
from numba.typed import List
import time
import math
import random
import os, sys
import csv
from imageio import imread
from PIL import Image

MAX_EPOCH = 20
m = 60000  # SAMPLES TRAIN
val = 10001
n = 784  # INPUTS
H = 800
outs = 10  # OUTPUT LAYER
LAYERS = 1  # HIDDEN LAYERS

x = np.ascontiguousarray(np.ones((n + 1) * m, dtype=np.float32).reshape(n + 1, m))  # INPUT MATRIX
v_x = np.ascontiguousarray(np.ones((n + 1) * val, dtype=np.float32).reshape(n + 1, val))
e = np.ascontiguousarray(np.zeros((outs, m), dtype=np.float32))  # SOLUTIONS
v_e = np.ascontiguousarray(np.zeros((outs, val), dtype=np.float32))
W = List()
Y = List()
OUT = np.ascontiguousarray(np.zeros(outs, dtype=np.float32))
W_IN = np.ascontiguousarray(np.zeros((n + 1) * H, dtype=np.float32).reshape(n + 1, H))
W_OUT = np.ascontiguousarray(np.zeros((H + 1) * outs, dtype=np.float32).reshape(H + 1, outs))
D_OUT = np.ascontiguousarray(np.zeros(outs * outs, dtype=np.float32).reshape(outs, outs))
D = List()
Eps = np.float32(10 ** -4)  # LEARNING RATE

x_plot = np.arange(MAX_EPOCH)
y_plot = np.zeros(MAX_EPOCH, dtype=np.float32)
vx_plot = np.arange(MAX_EPOCH)
vy_plot = np.zeros(MAX_EPOCH, dtype=np.float32)

"""path = "C:\\Users\\Giovanni\\Desktop\\ComputerVisionNN\\real_img_resized\\"
dirs = os.listdir( path )
ind = 0
for img in dirs:
    image = Image.open(path+img)
    data = np.ascontiguousarray(image)
    #print(data)
    pixel = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x[pixel, ind] = np.float32(1 - np.float32(((int(data[i, j][0]) + int(data[i, j][1]) + int(data[i, j][2])) / 765)))
            pixel += 1
    ind += 1
"""
train_set = open("C:\\Users\\Giovanni\\Documenti\\ComputerVisionNN\\mnist\\mnist_train.csv")
csv_reader = csv.reader(train_set)
header = next(csv_reader)

i = 0
for row in csv_reader:
    x[:, i][:n] = np.array(row[1:], dtype=np.float32) / 255
    e[int(row[0]), i] = 1
    i += 1
    if (i == m): break
train_set.close()

validation_set = open("C:\\Users\\Giovanni\\Documenti\\ComputerVisionNN\\mnist\\mnist_test.csv")
csv_reader = csv.reader(validation_set)
header = next(csv_reader)

i = 0
for row in csv_reader:
    v_x[:, i][:n] = np.array(row[1:], dtype=np.float32) / 255
    v_e[int(row[0]), i] = 1
    i += 1
validation_set.close()

# --- inizializzare x ed e ---

if LAYERS == 1:
    W.append(np.empty(0))

for i in range(0, LAYERS - 1):
    W.append(np.ascontiguousarray(np.zeros((H + 1) * H, dtype=np.float32).reshape(H + 1, H)))
    D.append(np.ascontiguousarray(np.zeros(H * H, dtype=np.float32).reshape(H, H)))  # MATRIX OF DERIVATIVES
    Y.append(np.ascontiguousarray(np.zeros(H + 1, dtype=np.float32)))
    Y[i][H] = np.float32(1)

# ------ recupero pesi ----
"""#with open('C:\\Users\\Giovanni\\Desktop\\ComputerVisionNN\\weights\\weights_W.csv', 'r') as weights_W:
    csvreader = csv.reader(weights_W)
    i = 0
    for row in csvreader:
        W[i] = np.ascontiguousarray(np.array(row, dtype=np.float32)).reshape(H + 1, H)
        i += 1

W_IN = np.ascontiguousarray(
    np.loadtxt('C:\\Users\\Giovanni\\Desktop\\ComputerVisionNN\\weights\\weights_W_in.csv', delimiter=","),
    dtype=np.float32).reshape(n + 1, H)
W_OUT = np.ascontiguousarray(
    np.loadtxt('C:\\Users\\Giovanni\\Desktop\\ComputerVisionNN\\weights\\weights_W_out.csv', delimiter=","),
    dtype=np.float32).reshape(H + 1, outs)"""

# ----------------------"""

D.append(np.ascontiguousarray(np.zeros(H * H, dtype=np.float32).reshape(H, H)))  # MATRIX OF DERIVATIVES
Y.append(np.ascontiguousarray(np.zeros(H + 1, dtype=np.float32)))
Y[LAYERS - 1][H] = np.float32(1)


@njit
def randomize(W, W_IN, W_OUT):
    if LAYERS > 1:
        for i in range(LAYERS - 1):
            W[i] = randomize_w(W[i])

    W_IN = randomize_w(W_IN)
    W_OUT = randomize_w(W_OUT)


@njit
def randomize_w(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = random.uniform(-4 * math.sqrt(2 / (W.shape[0] + W.shape[1])),
                                     4 * math.sqrt(2 / (W.shape[0] + W.shape[1])))
    return W


#@njit
def sigma(x):
    return np.divide(1, np.add(1, np.exp(-2 * x)))


@njit
def relu(x):
    return np.maximum(0, x)


@njit
def reluDerivative(D, y):
    for i in range(D.shape[0]):
        D[i] = 1 if y[i] >= 0 else 0
    return D


@guvectorize([void(float32[:, :], float32[:, :], float32[:, :])], '(m,l),(l,n)->(m,n)', target='cuda')
def matmul_gu3(A, B, out):
    """Perform square matrix multiplication of out = A * B
    """
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        out[i, j] = tmp


matmul_gu3.max_blocksize = 32

@guvectorize([void(float32[:, :], float32[:], float32[:])], '(m,l),(l)->(m)', target='cuda')
def matvetmul_gu3(A, B, out):
    """Perform square matrix-vector multiplication of out = A * B
    """
    i, j = cuda.grid(2)
    if i < A.shape[0]:
        tmp = 0.
        for k in range(B.shape[0]):
            tmp += A[i, k] * B[k]
        out[i] = tmp


matvetmul_gu3.max_blocksize = 32

@njit
def neuron(x, w):
    return relu(x @ w)


@njit
def forward(x, Y, OUT, W, W_IN, W_OUT, D, D_OUT):
    Y[0][0:H] = neuron(x, W_IN)
    #D[0] = np.diag((Y[0] * (1 - Y[0]))[:(H)])
    D[0] = np.diag(reluDerivative(np.diag(D[0]), Y[0][:H]))
    if LAYERS > 1:
        for i in range(1, LAYERS):
            Y[i][0:H] = neuron(Y[i - 1], W[i - 1])
            #D[i] = np.diag((Y[i] * (1 - Y[i]))[:H])
            D[i] = np.diag(reluDerivative(np.diag(D[i]), Y[i][:H]))
    OUT = neuron(Y[LAYERS - 1], W_OUT)
    #D_OUT = np.diag(OUT * (1 - OUT))
    D_OUT = np.diag(reluDerivative(np.diag(D_OUT), OUT))
    return (Y, OUT, D, D_OUT)


@njit
def backward(x, Y, OUT, W, W_IN, W_OUT, D, D_OUT, e):
    # BACKPROPAGATED ERROR TO THE OUTPUT LAYER
    WAUX = W_OUT
    delta = OUT - e
    delta_out= D_OUT @ delta
    #delta_out = matvetmul_gu3(D_OUT, delta)
    W_OUT = W_OUT - Eps * np.outer(Y[LAYERS - 1], delta_out)
    delta_pre = delta_out

    if LAYERS > 1:
        for i in range(LAYERS - 2, -1, -1):  # UPDATE EVERY LAYER
            delta_n = (D[i + 1] @ np.ascontiguousarray(WAUX[:H, :H])) @ delta_pre  # BACKPROPAGATED ERROR TO THE I-TH LAYER
            #delta_n = matvetmul_gu3(matmul_gu3(D[i + 1], WAUX[:H, :H]), delta_pre)
            WAUX = W[i]
            W[i] = W[i] - Eps * np.outer(Y[i], delta_n)  # UPDATE I-TH NEURON WEIGHTS
            delta_pre = delta_n

    delta_0 = (D[0] @ np.ascontiguousarray(WAUX[:H, :H])) @ delta_pre
    #delta_0 = matvetmul_gu3(matmul_gu3(D[0], WAUX[:H, :H]), delta_pre)
    W_IN = W_IN - Eps * np.outer(x, delta_0)

    return W, W_IN, W_OUT


@njit
def E(x, Y, OUT, W, W_IN, W_OUT, D, D_OUT, e):
    E = np.float32(0)
    for i in range(x.shape[1]):
        (Y, OUT, D, D_OUT) = forward(np.ascontiguousarray(x[:, i]), Y, OUT, W, W_IN, W_OUT, D, D_OUT)
        index_max_out = np.where(OUT == np.amax(OUT))
        index_max_e = np.where(e[:, i] == np.amax(e[:, i]))
        if index_max_out[0][0] != index_max_e[0][0]:
            E = E + 1
    return E / x.shape[1]


def stop():
    global Error
    # Error = E()
    # if Error < 11: return True
    # if deltaE < 10 ** -4: return True
    return False


# start

randomize(W, W_IN, W_OUT)

print('started')



#@njit
def START_NN(x, Y, OUT, W, W_IN, W_OUT, D, D_OUT, e, y_plot, vy_plot):
    tot_time_f = 0
    tot_time_b = 0
    for i in range(MAX_EPOCH):
        for ind in range(m):
            start_time_f = time.perf_counter_ns()
            (Y, OUT, D, D_OUT) = forward(np.ascontiguousarray(x[:, ind]), Y, OUT, W, W_IN, W_OUT, D, D_OUT)
            stop_time_f = time.perf_counter_ns()
            tot_time_f += stop_time_f-start_time_f
            print("\n--- Forward: " + str(stop_time_f-start_time_f) + " ---")
            print("\n--- AGV Forward: " + str(tot_time_f/((ind+1)*1e6)) + " ---")
            start_time_b = time.perf_counter_ns()
            (W, W_IN, W_OUT) = backward(x[:, ind], Y, OUT, W, W_IN, W_OUT, D, D_OUT,
                                        e[:, ind])  # esegue la back propagation, aggiorna tutti i pesi
            stop_time_b = time.perf_counter_ns()
            tot_time_b += (stop_time_b - start_time_b)
            print("\n--- Backward: " + str(stop_time_b-start_time_b) + "  ---")
            print("\n--- AGV Backward: " + str(tot_time_b/((ind+1)*1e6)) + " ---")
        print("Epoch: " + str(i))
        Acc = np.float32(1 - E(x, Y, OUT, W, W_IN, W_OUT, D, D_OUT, e))
        Acc_val = np.float32(1 - E(v_x, Y, OUT, W, W_IN, W_OUT, D, D_OUT, v_e))
        print("Accuracy train: ")
        print(Acc)
        print("Accuracy test: ")
        print(Acc_val)
        y_plot[i] = Acc
        vy_plot[i] = Acc_val
    return OUT, W, W_IN, W_OUT, y_plot, vy_plot


start_time = time.time()

(OUT, W, W_IN, W_OUT, y_plot, vy_plot) = START_NN(x, Y, OUT, W, W_IN, W_OUT, D, D_OUT, e, y_plot, vy_plot)

print('Progression: 100%')
print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(x_plot, y_plot, 'b', label='Train set')
plt.plot(vx_plot, vy_plot, 'g', label='Test set')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
leg = plt.legend(loc='upper left')
plt.show()
# print(OUT)

"""if LAYERS > 1:
    saveW = List()
    for w in W:
        saveW.append(str(",".join(str(x) for x in w.reshape((H + 1) * H))))
    np.savetxt('C:\\Users\\Giovanni\\Desktop\\ComputerVisionNN\\weights\\weights_W.csv', saveW, delimiter=" ", fmt='% s')

np.savetxt('C:\\Users\\Giovanni\\Desktop\\ComputerVisionNN\\weights\\weights_W_out.csv', W_OUT, delimiter=",")
np.savetxt('C:\\Users\\Giovanni\\Desktop\\ComputerVisionNN\\weights\\weights_W_in.csv', W_IN, delimiter=",")

for i in range(10):
    (Y, OUT, D, D_OUT) = forward(np.ascontiguousarray(x[:, i]), Y, OUT, W, W_IN, W_OUT, D, D_OUT)
    print(np.where(OUT == np.amax(OUT))[0])
"""