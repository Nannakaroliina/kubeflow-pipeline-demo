import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

def main():
    print("[INFO] Download dataset")
    eddl.download_mnist()
    print("----------------------------") 

    print("[INFO] Model init")
    epochs = 1
    batch_size = 100
    num_classes = 10

    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.LeakyReLu(eddl.Dense(layer, 64))
    layer = eddl.LeakyReLu(eddl.Dense(layer, 128))
    layer = eddl.LeakyReLu(eddl.Dense(layer, 256))    
    layer = eddl.LeakyReLu(eddl.Dense(layer, 512))
    layer = eddl.LeakyReLu(eddl.Dense(layer, 1024))
    out = eddl.Softmax(eddl.Dense(layer, num_classes))
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.rmsprop(0.01),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU()
    )
    print("----------------------------") 

    x_train = Tensor.load("mnist_trX.bin")
    y_train = Tensor.load("mnist_trY.bin")
    x_test = Tensor.load("mnist_tsX.bin")
    y_test = Tensor.load("mnist_tsY.bin")

    Tensor.save(x_test, "mnist_tsX_saved.bin")
    Tensor.save(y_test, "mnist_tsY_saved.bin")

    x_train.div_(255.0)

    # train
    print("[INFO] Training ...")
    eddl.fit(net, [x_train], [y_train], batch_size, epochs)
    print("----------------------------") 

    # save model weights
    print("[INFO] Save model weight as bin")
    eddl.save(net, 'net.bin')
    print("----------------------------") 

if __name__ == "__main__":
    main()
