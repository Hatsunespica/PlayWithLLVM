#include <stdio.h>

int count(int x){
  while(x)--x;
  return x;
}

int add(int a,int b){
  return a+b;
}

int main(){
  printf("Hello\n");
  printf("2+2=%d\n",add(2,2));
  return 0;
}
