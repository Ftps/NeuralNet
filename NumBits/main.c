#include "neural_net.h"

int main(int argc, char* argv[])
{
    time_t t;

    srand((unsigned)time(&t));

    if(argc < 2){printf("\nERROR: No filename given.\n\n"); exit(-10);}
    NeuNet neural = load_neural(argv[1], 0);

    for(int i = 0; i < 10; ++i){

        use_neural(neural, rand() % MAX_POWER, 0);
    }

    destroy_neural(neural);

    return 0;
}
