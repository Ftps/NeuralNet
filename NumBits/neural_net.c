#include "neural_net.h"

/* Numerical functions */

float cost_func(float *exp, int *teo, int l)
{
    float h = 0;

    for(int i = 0; i < l; ++i){
        h += pow(exp[i] - teo[i], 2);
    }

    return h;
}

float sigmoid(float f)
{
    return 1/(1+exp(-f));
}


/* Loading, saving, creating and destroying functions */

NeuNet load_neural(char *filename, int new)
{
    FILE *fp = fopen(filename, "r");
    NeuNet neural;
    int def;

    UNTIL(':')
    fscanf(fp, " %d", &(neural.l));

    if(neural.l < 3){
        printf("\nERROR: Not enough layers in the network.\n\n");
        exit(-1);
    }

    neural.sub_l = (int*)malloc(sizeof(int)*neural.l);

    for(int i = 0; i < neural.l; ++i){
        UNTIL(':')
        fscanf(fp, " %d", &(neural.sub_l[i]));
    }

    neural.neu = (float**)malloc(sizeof(float*)*neural.l);
    neural.bias = (float**)malloc(sizeof(float*)*neural.l);
    neural.wei = (float***)malloc(sizeof(float**)*neural.l);

    for(int i = 0; i < neural.l; ++i){
        neural.neu[i] = (float*)malloc(sizeof(float)*neural.sub_l[i]);
        if(i){
            neural.bias[i] = (float*)malloc(sizeof(float)*neural.sub_l[i]);
            neural.wei[i] = (float**)malloc(sizeof(float*)*neural.sub_l[i]);
            for(int j = 0; j < neural.sub_l[i]; ++j){
                neural.wei[i][j] = (float*)malloc(sizeof(float)*neural.sub_l[i-1]);
            }
        }
    }

    UNTIL(':')
    fscanf(fp, " %d", &def);

    if(def && !new){
        for(int i = 1; i < neural.l; ++i){
            UNTIL(':')
            fgetc(fp);
            for(int j = 0; j < neural.sub_l[i]; ++j){
                fscanf(fp, " %f", &(neural.bias[i][j]));
                for(int k = 0; k < neural.sub_l[i-1]; ++k){
                    fscanf(fp, " %f", &(neural.wei[i][j][k]));
                }
                fgetc(fp);
            }
        }
    }
    else{
        for(int i = 1; i < neural.l; ++i){
            for(int j = 0; j < neural.sub_l[i]; ++j){
                neural.bias[i][j] = ((rand() % 8001) - 4000)/1000.0;
                for(int k = 0; k < neural.sub_l[i-1]; ++k){
                    neural.wei[i][j][k] = ((rand() % 8001) - 4000)/1000.0;
                }
            }
        }
    }

    fclose(fp);
    return neural;
}

void save_neural(char *filename, NeuNet neural)
{
    FILE *fp = fopen(filename, "w");

    fprintf(fp, "Layers: %d\n", neural.l);

    for(int i = 0; i < neural.l; ++i){
        fprintf(fp, "L%d: %d\n", i+1, neural.sub_l[i]);
    }

    fprintf(fp, "\nDefined: 1");

    for(int i = 1; i < neural.l; ++i){
        fprintf(fp, "\n\nL%d-%d:", i, i+1);
        for(int j = 0; j < neural.sub_l[i]; ++j){
            fputc('\n', fp);
            fprintf(fp, " %f", neural.bias[i][j]);
            for(int k = 0; k < neural.sub_l[i-1]; ++k){
                fprintf(fp, " %f", neural.wei[i][j][k]);
            }
        }
    }

    fclose(fp);
}

void destroy_neural(NeuNet neural)
{
    for(int i = 0; i < neural.l; ++i){
        free(neural.neu[i]);
        if(i){
            free(neural.bias[i]);
            for(int j = 0; j < neural.sub_l[i]; ++j) free(neural.wei[i][j]);
            free(neural.wei[i]);
        }
    }

    free(neural.neu);
    free(neural.bias);
    free(neural.wei);
    free(neural.sub_l);
}

BackProp start_backprop(NeuNet neural)
{
    BackProp back;

    back.neu_grad = (float**)malloc(sizeof(float*)*neural.l);
    back.bias_grad = (float**)malloc(sizeof(float*)*neural.l);
    back.wei_grad = (float***)malloc(sizeof(float**)*neural.l);

    for(int i = 0; i < neural.l; ++i){
        back.neu_grad[i] = (float*)malloc(sizeof(float)*neural.sub_l[i]);
        if(i){
            back.bias_grad[i] = (float*)malloc(sizeof(float)*neural.sub_l[i]);
            back.wei_grad[i] = (float**)malloc(sizeof(float*)*neural.sub_l[i]);
            for(int j = 0; j < neural.sub_l[i]; ++j){
                back.wei_grad[i][j] = (float*)malloc(sizeof(float)*neural.sub_l[i-1]);
            }
        }
    }

    back.cost = 0;
    return back;
}

void destroy_backprop(BackProp back, NeuNet neural)
{
    for(int i = 0; i < neural.l; ++i){
        free(back.neu_grad[i]);
        if(i){
            free(back.bias_grad[i]);
            for(int j = 0; j < neural.sub_l[i]; ++j) free(back.wei_grad[i][j]);
            free(back.wei_grad[i]);
        }
    }

    free(back.neu_grad);
    free(back.bias_grad);
    free(back.wei_grad);
}


/* Using the neural network */

void use_neural(NeuNet neural, unsigned int num, int back)
{
    float aux;
    int *teo;

    printf("%d\n", num);

    for(int i = 0; i < neural.sub_l[0]; ++i){
        neural.neu[0][i] = (num & 1<<i) ? 1: 0;
    }

    for(int i = 1; i < neural.l; ++i){
        for(int j = 0; j < neural.sub_l[i]; ++j){
            aux = neural.bias[i][j];
            for(int k = 0; k < neural.sub_l[i-1]; ++k){
                aux += neural.wei[i][j][k]*neural.neu[i-1][k];
            }
            neural.neu[i][j] = sigmoid(aux);
        }
    }

    if(!back){
        teo = (int*)malloc(sizeof(int)*neural.sub_l[neural.l-1]);
        aux = 0;
        for(int i = 0; i < neural.sub_l[0]; ++i){
            aux += neural.neu[0][i];
        }
        printf("Number: %d or %x.\n", num, num);
        printf("Number of bits: %d\n", (int)aux);
        aux = 0;
        for(int i = 0; i < neural.sub_l[neural.l-1]; ++i){
            if(neural.neu[neural.l-1][i] < 0.5){
                aux += 1<<i;
                teo[i] = 1;
            }
            else teo[i] = 0;
        }
        printf("Number by NeuNet: %d, Cost: %f\n\n", (int)aux, cost_func(neural.neu[neural.l-1], teo, neural.sub_l[neural.l-1]));
    }
}
