import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import argparse
import numpy as np

def main(model, x_test, y_test):
    print("[INFO] Model init")
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

    print("[INFO] READ TEST DATA") 
    test_data = Tensor.fromarray(np.load(x_test).astype(np.float32))
    test_label = Tensor.fromarray(np.load(y_test).astype(np.float32))
    test_data.div_(255.0)
    print("----------------------------")  

    # load model weights eddl.load(model_weights)
    print("[INFO] LOAD MODEL") 
    eddl.load(net, model)
    print("----------------------------")  

    # predict to x[test]
    print("[INFO] PREDICT") 
    y_pred = eddl.predict(net, [test_data])
    print("----------------------------")  

    print("[INFO] Y_PRED") 
    print(y_pred[0].getdata())
    print("----------------------------")  

    print("[INFO] GROUND TRUTH / LABELS") 
    print(test_label.getdata())
    print("----------------------------")  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--x_test')
    parser.add_argument('--y_test')
    args = parser.parse_args()
    main(args.model, args.x_test, args.y_test)
