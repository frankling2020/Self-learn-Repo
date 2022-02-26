/*
Write a program that uses UNIX system calls to ''ping-pong'' a byte 
between two processes over a pair of pipes, one for each direction. 
The parent should send a byte to the child; the child should print "<pid>: 
received ping", where <pid> is its process ID, write the byte on the pipe
to the parent, and exit; the parent should read the byte from the child, 
print "<pid>: received pong", and exit. Your solution should be in the file 
user/pingpong.c.
*/

#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int main(int argc, char *argv[]){
    int p[2];
    char recv[128];
    pipe(p);

    if (fork() == 0){
        read(p[0], recv, 4);
        close(p[0]);
        

        printf("%d: received %s\n", getpid(), recv);

        write(p[1], "pong", 4);
        close(p[1]);
    } 
    else {
        write(p[1], "ping", 4);
        close(p[1]);

        wait(0);
        
        read(p[0], recv, 4);
        close(p[0]);
        
        printf("%d: received %s\n", getpid(), recv);
    }
    exit(0);
}

