#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int arc, char**argv)
{ 
  int MEMSIZE = 1024*1024*1024;
  char *c = malloc(sizeof(char) * MEMSIZE);

  if(c)
  {
    printf("allocated memory %d \n", MEMSIZE);
    memset(c, 1, sizeof(char) * MEMSIZE);  
    free(c);
  }
  else
  {
    printf("Out of memory\n");
  }
  return 0;
}
