/* @(#)matlist.c	1.2 7/10/92 */
/*
 * Updated by Sibomana@topo.ucl.ac.be for ECAT 7.0 support
 */

#include "matrix.h"

main( argc, argv)
  int argc;
  char **argv;
{
	MatrixFile *mptr;
	MatDirNode *node;
	struct Matval mat;
	int i;
	char cbufr[256];

	if (argc < 2) crash("usage : matlist matrix_file\n");
	mptr = matrix_open( argv[1], MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	if (!mptr) {
		matrix_perror(argv[1]);
		exit(1);
	}
	printf("file '%s' is of type %d with %d matrices\n", argv[1],
	  mptr->mhptr->file_type, mptr->dirlist->nmats);
	node = mptr->dirlist->first;
	while (node)
	{
	  mat_numdoc( node->matnum, &mat);
	  sprintf( cbufr, "%d,%d,%d,%d,%d",
	    mat.frame, mat.plane, mat.gate, mat.data, mat.bed);
	  printf("%8.8x %-12s %10d %10d\n",
	    node->matnum, cbufr,
	    node->strtblk, node->endblk);
	  node = node->next;
	}
	matrix_close(mptr);
}
