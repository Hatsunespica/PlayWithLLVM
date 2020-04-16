#include <stdio.h>

void countDown(){
    int x=0;
    while(x<10){
        ++x;
    }
}

int addFunc(int x,int y){
    return x+y;
}

int main(){
    printf("5+2%d\n",addFunc(5,2));
    countDown();
    return 0;
}
