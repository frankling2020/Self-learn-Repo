#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

#define FIRST_NUMBER 2
#define LAST_NUMBER 35

// from https://swtch.com/~rsc/thread/

int prime_filter(int in, int prime){
    int num;
    int p[2];
    pipe(p);

    if (!fork()){
        while (read(in, &num, sizeof(int))){
            if (num % prime != 0){
                write(p[1], &num, sizeof(int));
            }
        }
        close(in);
        close(p[1]);
        exit(0);
    }
    
    // wait(0);
    close(in);
    close(p[1]);
    return p[0];
}

int generator(int first, int last){
    int p[2];
    pipe(p);

    if (!fork()){
        for (int num = first; num < last; ++num){
            write(p[1], &num, sizeof(int));
        }
        close(p[1]);
        exit(0);
    }

    wait(0);
    close(p[1]);
    return p[0];
}


int main(int argc, char* argv[]){
    int prime;
    
    int in = generator(FIRST_NUMBER, LAST_NUMBER);
    
    while (read(in, &prime, sizeof(int))){
        printf("prime %d\n", prime);
        in = prime_filter(in, prime);
    }

    close(in);
    exit(0);
}