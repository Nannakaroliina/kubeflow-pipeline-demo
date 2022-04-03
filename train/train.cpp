#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

using namespace eddl;
     
int main(int argc, char* argv[]){

    // download CIFAR data
    cout << "[INFO] Download dataset"<< endl;
    download_cifar10();
    cout << "----------------------------"<< endl;

    cout << "[INFO] Model init"<< endl;
    // Settings
    int epochs = 1;
    int batch_size = 100;
    int num_classes = 10;
    float lr = 0.01;

    // network
    layer in=Input({3,32,32}); // input RGB image with size 32x32 pixels
    layer l=in;

    // Data augmentation
    // l = RandomShift(l, {-0.2f, +0.2f}, {-0.2f, +0.2f});
    // l = RandomRotation(l, {-30.0f, +30.0f});
    // l = RandomScale(l, {0.85f, 2.0f});
    // l = RandomFlip(l, 1);
    // l = RandomCrop(l, {28, 28});
    // l = RandomCropScale(l, {0.f, 1.0f});
    // l = RandomCutout(l, {0.0f, 0.3f}, {0.0f, 0.3f});
    // l=Select(l, {"1", "1:31", "1:31"});

    l=MaxPool2D(ReLu(Conv2D(l,32,{3,3},{1,1})),{2,2});
    l=MaxPool2D(ReLu(Conv2D(l,64,{3,3},{1,1})),{2,2});
    l=MaxPool2D(ReLu(Conv2D(l,128,{3,3},{1,1})),{2,2});
    l=MaxPool2D(ReLu(Conv2D(l,256,{3,3},{1,1})),{2,2});

    l=Reshape(l,{-1});

    l=Activation(Dense(l,128),"relu");

    layer out=Activation(Dense(l, num_classes), "softmax");

    // net define input and output layers list
    model net=Model({in},{out});


    // Build model
    build(net,
          sgd(lr, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
        //   CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          CS_CPU()
    );
    cout << "----------------------------"<< endl;

    // plot the model
    cout << "[INFO] Model plot"<< endl;
    plot(net,"model.pdf");
    cout << "----------------------------"<< endl;

    // get some info from the network
    cout << "[INFO] Model summary"<< endl;
    summary(net);
    cout << "----------------------------"<< endl;

    // Load and preprocess training data
    Tensor* x_train = Tensor::load("cifar_trX.bin");
    Tensor* y_train = Tensor::load("cifar_trY.bin");
    x_train->div_(255.0f);

    // train
    cout << "[INFO] Training ..."<< endl;
    fit(net,{x_train},{y_train},batch_size,epochs);
    cout << "----------------------------"<< endl;

    // save model weights
    cout << "[INFO] Save model weight as bin"<< endl;
    save(net, "simple_CNN.bin");
    cout << "----------------------------"<< endl;
    return 0;
}