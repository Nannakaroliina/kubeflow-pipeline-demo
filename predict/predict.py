import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import argparse
import tarfile

def extract(tar_url, extract_path='.'):
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])
    return tar.getnames()[0]


def main(model, x, y):
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

    file_x = extract(x)
    x_test = Tensor.load(file_x)
    file_y = extract(y)
    y_test = Tensor.load(file_y)
    x_test.div_(255.0)

    # load model weights eddl.load(model_weights)
    print("[INFO] LOAD MODEL") 
    file_model = extract(model)
    eddl.load(net, file_model)
    print("----------------------------")  

    # predict to x[test]
    print("[INFO] PREDICT") 
    y_pred = eddl.predict(net, [x_test])
    print("----------------------------")  

    print("[INFO] Y_PRED") 
    print(y_pred[0].getdata())
    print("----------------------------")  

    print("[INFO] GROUND TRUTH / LABELS") 
    print(y_test.getdata())
    print("----------------------------")  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--x')
    parser.add_argument('--y')
    args = parser.parse_args()
    main(args.model, args.x, args.y)
