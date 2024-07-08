#ifndef ML_H
#define ML_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

// Structure to hold weights and biases
typedef struct {
    double** W;  // Weight matrices between layers
    double** b;  // Bias vectors for each layer
} NetworkParams;

// Neural network parameters
int _NUM_LAYERS;          // Number of layers in the neural network
int* _LAYERS;             // Array defining the number of nodes in each layer: {Input layer, Hidden layers, Output layer}
double _LEARNING_RATE;    // Learning rate used in gradient descent for updating weights
int _EPOCHS;              // Number of epochs (iterations) for training the neural network

// Function prototypes
double sigmoid(double x);                                        // Sigmoid activation function
double sigmoid_derivative(double x);                             // Derivative of sigmoid activation function
NetworkParams allocate_params(int num_layers, int* layers);       // Allocate memory for weights and biases
void free_params(NetworkParams params, int num_layers);           // Free memory allocated for weights and biases
void setParams(int num_layers, int layers[], double learning_rate, int epochs);  // Set neural network parameters
void initialize_weights_biases(double** W, double** b);           // Initialize weights and biases with random values
void forward(double* X, double** W, double** b, double** a);      // Forward propagation
void backward(double* X, double** W, double** b, double** a, double* y_true);  // Backward propagation
void training(double X[][2], double y_true[][1], double** W, double** b);  // Training the neural network
void test(double X[][2], double y_true[][1], double** W, double** b, bool save);  // Testing the neural network

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid activation function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Allocate memory for weights and biases
NetworkParams allocate_params(int num_layers, int* layers) {
    NetworkParams params;

    // Allocate memory for weights
    params.W = (double**)malloc((num_layers - 1) * sizeof(double*));
    if (params.W == NULL) {
        fprintf(stderr, "Failed to allocate memory for weights\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for biases
    params.b = (double**)malloc((num_layers - 1) * sizeof(double*));
    if (params.b == NULL) {
        fprintf(stderr, "Failed to allocate memory for biases\n");
        exit(EXIT_FAILURE);
    }

    // Initialize weights and biases
    for (int i = 0; i < num_layers - 1; i++) {
        params.W[i] = (double*)malloc(layers[i] * layers[i + 1] * sizeof(double));
        if (params.W[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for W[%d]\n", i);
            exit(EXIT_FAILURE);
        }

        params.b[i] = (double*)malloc(layers[i + 1] * sizeof(double));
        if (params.b[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for b[%d]\n", i);
            exit(EXIT_FAILURE);
        }
    }

    return params;
}

// Free memory allocated for weights and biases
void free_params(NetworkParams params, int num_layers) {
    for (int i = 0; i < num_layers - 1; i++) {
        free(params.W[i]);
        free(params.b[i]);
    }
    free(params.W);
    free(params.b);
}

// Set neural network parameters
void setParams(int num_layers, int layers[], double learning_rate, int epochs) {
    srand(time(NULL));
    _NUM_LAYERS = num_layers;
    _LAYERS = (int*)malloc(num_layers * sizeof(int));
    if (_LAYERS == NULL) {
        fprintf(stderr, "Failed to allocate memory for _LAYERS\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_layers; i++) {
        _LAYERS[i] = layers[i];
    }
    _LEARNING_RATE = learning_rate;
    _EPOCHS = epochs;
}

// Initialize weights and biases with random values
void initialize_weights_biases(double** W, double** b) {
    for (int i = 0; i < _NUM_LAYERS - 1; i++) {
        for (int j = 0; j < _LAYERS[i] * _LAYERS[i + 1]; j++) {
            W[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        for (int j = 0; j < _LAYERS[i + 1]; j++) {
            b[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
}

// Forward propagation
void forward(double* X, double** W, double** b, double** a) {
    for (int i = 0; i < _LAYERS[0]; i++) {
        a[0][i] = X[i];
    }

    for (int l = 1; l < _NUM_LAYERS; l++) {
        for (int j = 0; j < _LAYERS[l]; j++) {
            double sum = 0;
            for (int i = 0; i < _LAYERS[l - 1]; i++) {
                sum += a[l - 1][i] * W[l - 1][i * _LAYERS[l] + j];
            }
            a[l][j] = sigmoid(sum + b[l - 1][j]);
        }
    }
}

// Backward propagation
void backward(double* X, double** W, double** b, double** a, double* y_true) {
    double** dW = (double**)malloc((_NUM_LAYERS - 1) * sizeof(double*));
    if (dW == NULL) {
        fprintf(stderr, "Failed to allocate memory for dW\n");
        exit(EXIT_FAILURE);
    }
    double** db = (double**)malloc((_NUM_LAYERS - 1) * sizeof(double*));
    if (db == NULL) {
        fprintf(stderr, "Failed to allocate memory for db\n");
        exit(EXIT_FAILURE);
    }
    double** delta = (double**)malloc(_NUM_LAYERS * sizeof(double*));
    if (delta == NULL) {
        fprintf(stderr, "Failed to allocate memory for delta\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < _NUM_LAYERS - 1; i++) {
        dW[i] = (double*)calloc(_LAYERS[i] * _LAYERS[i + 1], sizeof(double));
        if (dW[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for dW[%d]\n", i);
            exit(EXIT_FAILURE);
        }
        db[i] = (double*)calloc(_LAYERS[i + 1], sizeof(double));
        if (db[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for db[%d]\n", i);
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < _NUM_LAYERS; i++) {
        delta[i] = (double*)malloc(_LAYERS[i] * sizeof(double));
        if (delta[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for delta[%d]\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Output layer error
    for (int i = 0; i < _LAYERS[_NUM_LAYERS - 1]; i++) {
        delta[_NUM_LAYERS - 1][i] = (y_true[i] - a[_NUM_LAYERS - 1][i]) * sigmoid_derivative(a[_NUM_LAYERS - 1][i]);
    }

    // Backpropagate the error
    for (int l = _NUM_LAYERS - 2; l > 0; l--) {
        for (int i = 0; i < _LAYERS[l]; i++) {
            double sum = 0;
            for (int j = 0; j < _LAYERS[l + 1]; j++) {
                sum += delta[l + 1][j] * W[l][i * _LAYERS[l + 1] + j];
            }
            delta[l][i] = sum * sigmoid_derivative(a[l][i]);
        }
    }

    // Compute gradients
    for (int l = 0; l < _NUM_LAYERS - 1; l++) {
        for (int i = 0; i < _LAYERS[l]; i++) {
            for (int j = 0; j < _LAYERS[l + 1]; j++) {
                dW[l][i * _LAYERS[l + 1] + j] += delta[l + 1][j] * a[l][i];
            }
        }
        for (int j = 0; j < _LAYERS[l + 1]; j++) {
            db[l][j] += delta[l + 1][j];
        }
    }

    // Apply gradient descent
    for (int l = 0; l < _NUM_LAYERS - 1; l++) {
        for (int i = 0; i < _LAYERS[l] * _LAYERS[l + 1]; i++) {
            W[l][i] += _LEARNING_RATE * dW[l][i];
        }
        for (int j = 0; j < _LAYERS[l + 1]; j++) {
            b[l][j] += _LEARNING_RATE * db[l][j];
        }
    }

    // Free allocated memory
    for (int i = 0; i < _NUM_LAYERS - 1; i++) {
        free(dW[i]);
        free(db[i]);
    }
    for (int i = 0; i < _NUM_LAYERS; i++) {
        free(delta[i]);
    }
    free(dW);
    free(db);
    free(delta);
}

// Training the neural network
void training(double X[][2], double y_true[][1], double** W, double** b) {
    double** a = (double**)malloc(_NUM_LAYERS * sizeof(double*));
    if (a == NULL) {
        fprintf(stderr, "Failed to allocate memory for a\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < _NUM_LAYERS; i++) {
        a[i] = (double*)malloc(_LAYERS[i] * sizeof(double));
        if (a[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for a[%d]\n", i);
            exit(EXIT_FAILURE);
        }
    }

    for (int epoch = 0; epoch < _EPOCHS; epoch++) {
        for (int i = 0; i < 4; i++) {
            // Forward propagation
            forward(X[i], W, b, a);
            // Backward propagation
            backward(X[i], W, b, a, y_true[i]);
        }
    }

    // Free allocated memory
    for (int i = 0; i < _NUM_LAYERS; i++) {
        free(a[i]);
    }
    free(a);
}

// Testing the neural network
void test(double X[][2], double y_true[][1], double** W, double** b, bool save) {
    double** a = (double**)malloc(_NUM_LAYERS * sizeof(double*));
    if (a == NULL) {
        fprintf(stderr, "Failed to allocate memory for a\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < _NUM_LAYERS; i++) {
        a[i] = (double*)malloc(_LAYERS[i] * sizeof(double));
        if (a[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for a[%d]\n", i);
            exit(EXIT_FAILURE);
        }
    }

    FILE *file = NULL;
    if (save) {
        file = fopen("data.txt", "a");
        if (file == NULL) {
            fprintf(stderr, "Failed to open file for writing\n");
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < 4; i++) {
        // Forward propagation
        forward(X[i], W, b, a);

        // Print results
        printf("Input: %f %f, Predicted: %f, True: %f\n", X[i][0], X[i][1], a[_NUM_LAYERS - 1][0], y_true[i][0]);
        
        // Save results to file if needed
        if (save && file != NULL) {
            fprintf(file, "Input: %f %f, Predicted: %f, True: %f\n", X[i][0], X[i][1], a[_NUM_LAYERS - 1][0], y_true[i][0]);
        }
    }

    if (file != NULL) {
        fclose(file);
    }

    // Free allocated memory
    for (int i = 0; i < _NUM_LAYERS; i++) {
        free(a[i]);
    }
    free(a);
}

#endif // ML_H
