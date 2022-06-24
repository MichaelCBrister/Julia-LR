using Flux, Images, MLDatasets, Plots
using Flux: crossentropy, onecold, onehotbatch, train!

using LinearAlgebra, Random, Statistics


# Loading data
X_train_raw, y_train_raw = MLDatasets.MNIST(split = :train)[:]
X_test_raw, y_test_raw = MLDatasets.MNIST(split = :test)[:]


# View training input
X_train_raw
index = 1
img = X_train_raw[:, :, index]

colorview(Gray, img')


# View training label
y_train_raw
y_train_raw[index]


# View testing input
X_test_raw
img = X_test_raw[:, :, index]

colorview(Gray, img')

# View testing label
y_test_raw
y_test_raw[index]


# Flatten input data
X_train = Flux.flatten(X_train_raw)

X_test = Flux.flatten(X_test_raw)


# One-Hot Encode
y_train = onehotbatch(y_train_raw, 0:9)

y_test = onehotbatch(y_test_raw, 0:9)


# Model Architecture
model = Chain(
    Dense(28 * 28, 32, relu),
    Dense(32, 10),
    softmax
)


# Loss Function
loss(x, y) = crossentropy(model(x), y)


# Track Parameters
using Flux
ps = parameters(model)