#include "neural_net.h"

int main(int argc, char* argv[])
{
    if(argc < 2){printf("\nERROR: No filename given.\n\n"); exit(-10);}
    NeuNet neural = load_neural(argv[1], 0);
    save_neural(argv[1], neural);
    destroy_neural(neural);

    return 0;
}
