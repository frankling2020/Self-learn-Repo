#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/param.h"


int main(int argc, char *argv[]){
    if (argc < 2){
        fprintf(2, "Usage: xargs [OPTION]... COMMAND [INITIAL-ARGS]...\n");
        exit(1);
    }
    
    char chr;
    char buf[1024];
    char *p = buf;
    char *v[MAXARG];
    int i = 0, offset = 0;


    for (i = 0; i < argc-1; ++i) v[i] = argv[i+1];

    while (read(0, &chr, 1) > 0){
        if (chr == ' ' || chr == '\t'){
            v[i++] = p;
            buf[offset++] = 0;
            p = buf + offset;
        }
        else if (chr != '\n'){
            buf[offset++] = chr;
        }
        else {
            v[i++] = p;
            p = buf + offset;
            if (!fork()){
                exit(exec(v[0], v));
            }
            wait(0);
            i = argc - 1;
        }
    }
    exit(0);
}