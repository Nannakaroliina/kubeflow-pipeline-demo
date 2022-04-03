#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

using namespace eddl;
using namespace std;
     
int main(int argc, char* argv[]){

    cout << "[INFO] Initialize"<< endl;
    // Settings
    int batch_size = 100;
    int num_classes = 10;
    float lr = 0.01;

    // network
    layer in=Input({3,32,32}); // input RGB image with size 32x32 pixels
    layer l=in;

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

    cout << "[INFO] Load model weights"<< endl;
    load(net, "simple_CNN.bin");
    cout << "----------------------------"<< endl;

    // Load and preprocess training data
    Tensor* x_test = Tensor::load("cifar_tsX.bin");
    Tensor* y_test = Tensor::load("cifar_tsY.bin");
    x_test->div_(255.0f);

    cout << "[INFO] Predict"<< endl;
    vector<Tensor*> y_pred = predict(net, {x_test});
    cout << "----------------------------"<< endl;

    cout << "[INFO] SHOW LABELS"<< endl;
    y_test->print();
    cout << "----------------------------"<< endl;

    cout << "[INFO] SHOW Y_PRED" << endl;
    // freopen( "log_p.txt", "w", stdout);
    for (int i = 0; i < y_pred.size(); i++) {
        y_pred[i]->print();
    }
    cout << "----------------------------" << endl;
    // evaluate(net, {x_test}, {y_test}, batch_size);

    return 0;
}