import numpy as np

print("\n========= E1 =========")

A = np.array(
    [[0.20, -0.50, 1.10, 0.40], 
    [0.80, 0.30, -0.70, 0.60],
    [-0.40, 0.90, 0.10, -0.20]]
)

print(A.ndim)
print(A.shape)
print(A.size)

a1 = A[1]
a2 = A[:, -2:]

print("=========")
print(a1, a1.shape)
print(a2, a2.shape)

B = A.copy()
B[B<0] = 0

print("=========")
print("A:\n", A)
print("B:\n", B)


print("\n========= E2 =========")

a = np.array([0.80, -0.60, 0.70])
b = np.array([-0.20, 0.90, 0.40])

mul = a * b
dot = a @ b

print(mul, "shape:", mul.shape)
print(dot, "shape:", dot.shape)


length_a = np.sqrt(np.sum(a ** 2))
length_b = np.sqrt(np.sum(b ** 2))

print("a 的长:", length_a)
print("b 的长:", length_b)

print("norm(a):", np.linalg.norm(a))
print("norm(b):", np.linalg.norm(b))


print("\n========= E3 =========")

def angle_deg(a, b):
    length_a = np.linalg.norm(a)
    length_b = np.linalg.norm(b)

    if length_a == 0 or length_b == 0:
        raise ValueError("Zero vector")

    cos = (a @ b) / (length_a * length_b)
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))


a = np.array([1.00, -1.00])
b = np.array([1.00, 0.00])

print("angle:", angle_deg(a, b))


print("\n========= E4 =========")

p = np.array([-0.50, 1.20, 2.00])
q = np.array([0.80, -0.40, 1.10])

diff = p - q
d1 = np.sqrt(np.sum(diff ** 2))
d2 = np.sum(np.abs(diff))

print(diff)
print("Euclidean distance:", d1)
print("Manhattan distance:", d2)


print("\n========= E5 =========")

def unit_vector(a):
    length = np.linalg.norm(a)

    if length == 0:
        raise ValueError("Zero vector")

    return a / length


a = np.array([-1.20, 0.40, 0.80])
b = unit_vector(a)

print(b)
print(np.allclose(np.linalg.norm(b), 1))
print("a:", a)


print("\n========= E6 =========")

A = np.array(
    [[0.20, -0.50, 1.10],
     [0.80, 0.40, -0.30]]
)

a1 = A.sum(axis=0)
a2 = A.mean(axis=1)

print(a1, "shape:", a1.shape)
print(a2, "shape:", a2.shape)
print("max:", A.max())
print("axis=0: 每列求和")
print("axis=1: 每行求平均")


print("\n========= E7 =========")

X = np.array(
    [[0.80, -0.60, 0.70],
     [-0.40, 0.50, 0.20]]
)
scale = np.array([1.50, 0.50, -1.00])
bias = np.array([-0.10, 0.20, 0.30])

Y = X * scale + bias

print(Y, "shape:", Y.shape)
print("scale 和 bias 沿最后一轴对齐，对 X 的每一行执行相同运算")


print("\n========= E8 =========")

a = np.array(
    [[[0.20, -0.40], [0.70, 0.10]],
     [[0.30, 0.80], [-0.20, 0.50]],
     [[-0.10, 0.60], [0.40, -0.30]]]
)
b = a + 0.10

batch = np.stack((a, b), axis=0)
blue = batch[:, 2, :, :]
new_batch = batch.transpose(0, 2, 3, 1)

print(batch.shape)
print(blue)
print(new_batch.shape)


print("\n========= M1 =========")

def angles_to(query, X):
    length_q = np.linalg.norm(query)
    length_X = np.linalg.norm(X, axis=1)

    if length_q == 0:
        raise ValueError("Zero query vector")

    cos = np.full(length_X.shape, np.nan)
    np.divide(X @ query, length_X * length_q,
              out=cos, where=length_X != 0)

    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))


query = np.array([0.60, -0.80, 0.00])
X = np.array(
    [[0.30, -0.40, 0.00],
     [-0.60, 0.80, 0.00],
     [0.80, 0.60, 0.00],
     [0.00, 0.00, 0.00]]
)

print(angles_to(query, X))


print("\n========= M2 =========")

def top_k_cosine(query, gallery, k):
    length_q = np.linalg.norm(query)
    lengths = np.linalg.norm(gallery, axis=1)

    if length_q == 0:
        raise ValueError("Zero query vector")

    valid = np.where(lengths != 0)[0]

    if (not isinstance(k, (int, np.integer))
            or isinstance(k, (bool, np.bool_))
            or k <= 0 or k > len(valid)):
        raise ValueError("Invalid k")

    scores = (gallery[valid] @ query) / (lengths[valid] * length_q)
    order = np.argsort(-scores, kind="stable")[:k]

    return valid[order], scores[order]


query = np.array([0.60, -0.80, 0.00])
gallery = np.array(
    [[0.60, -0.80, 0.00],
     [0.30, -0.40, 0.00],
     [0.80, 0.60, 0.00],
     [-0.60, 0.80, 0.00],
     [0.00, 0.00, 0.00]]
)
ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])

indices, scores = top_k_cosine(query, gallery, 3)

print(indices)
print(scores)
print(ids[indices])


print("\n========= M3 =========")

A = np.array(
    [[0.20, -0.40],
     [0.80, 0.30],
     [-0.50, 0.70]]
)
B = np.array(
    [[0.10, 0.20],
     [-0.60, -0.20]]
)

diff = A[:, None, :] - B[None, :, :]
D = np.sqrt(np.sum(diff ** 2, axis=-1))
nearest = np.argmin(D, axis=1)

print("diff shape:", diff.shape)
print(D, "shape:", D.shape)
print(nearest)


print("\n========= M4 =========")

X = np.array(
    [[0.80, -0.60, 0.70],
     [-0.40, -0.50, 0.90]]
)
W1 = np.array(
    [[-0.10, 0.20, -0.80],
     [0.30, -0.50, 0.10]]
)
b1 = np.array([-0.15, 0.05])
W2 = np.array(
    [[0.20, -0.60],
     [-0.80, 0.52]]
)
b2 = np.array([0.10, -0.20])

Z = X @ W1.T + b1
H = np.maximum(Z, 0.0)
Y = H @ W2.T + b2

print("Z:", Z, "shape:", Z.shape)
print("H:", H, "shape:", H.shape)
print("Y:", Y, "shape:", Y.shape)


