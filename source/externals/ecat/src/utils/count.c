#include <stdio.h>

main( argc, argv)
  int argc;
  char **argv;
{
        int start, end, inc;

        if (argc < 3) {
          printf( "usage: %s start end [inc]\n", argv[0]);
		  exit(1);
		}
        sscanf( argv[1], "%d", &start);
        sscanf( argv[2], "%d", &end);
        inc = 1;
        if (argc > 3)
          sscanf( argv[3], "%d", &inc);
        while ( start <= end)
        {
          printf("%d ", start);
          start += inc;
        }
        printf("\n");
        exit(0);
}
