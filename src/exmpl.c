#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ml.h"

#define NUM_SAMPLES 4

int main() {

    // Define neural network parameters
    int num_layers = 3;
    int layers[] = {2, 3, 1};
    double learning_rate = 0.1;
    int epochs = 5000000;

    // Allocate memory for weights and biases
    NetworkParams params = allocate_params(num_layers, layers);
    
    // Set parameters of neural network
    setParams(num_layers, layers, learning_rate, epochs);

    // Define training data
    double X[NUM_SAMPLES][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double y_true[NUM_SAMPLES][1] = {
        {0},
        {1},
        {1},
        {0}
    };

    // Initialize weights and biases
    initialize_weights_biases(params.W, params.b);

    // Train the neural network
    printf("Training the neural network...\n");
    training(X, y_true, params.W, params.b);
    printf("Training complete.\n");

    // Test the neural network
    printf("\nTesting the neural network:\n");
    test(X, y_true, params.W, params.b, true);

    // Free allocated memory
    free_params(params, num_layers);

    return 0;
}
