#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int main(int argc, char *argv[]){
    if (argc < 2){
        fprintf(2, "usage: sleep ticks\n");
        exit(1);
    }
    
    printf("(nothing happens for a little while)");
    sleep(atoi(argv[1]));
    printf("\n");
    exit(0);
}