#ifndef NEURAL_NET
#define NEURAL_NET

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_POWER 4294967296

#define LOG {printf("IN FILE %s || IN LINE %d\n", __FILE__, __LINE__); fflush(stdout);}
#define UNTIL(X) {while(fgetc(fp) != X);}

typedef struct neunet{
    int l, *sub_l;
    float **neu, **bias;
    float ***wei;
}NeuNet;

typedef struct backprop{
    float ***wei_grad;
    float **bias_grad, **neu_grad;
    float cost;
}BackProp;


/* Numerical functions */
float cost_func(float *exp, int *teo, int l);
float sigmoid(float f);

/* Loading, saving, creating and destroying functions */
NeuNet load_neural(char *filename, int new);
void save_neural(char *filename, NeuNet neural);
void destroy_neural(NeuNet neural);
BackProp start_backprop(NeuNet neutral);
void destroy_backprop(BackProp back, NeuNet neural);

/* Using the neural network */
void use_neural(NeuNet neural, unsigned int num, int back);

#endif
