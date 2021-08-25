/* @(#)matpkg.c	1.1 3/16/92 */

#include "sincos.h"
#include "matpkg.h"

matmpy(a,b,c)
  Matrix a,b,c;
{	int i,j,k;
	float t;

	for (i=0; i<b->ncols; i++)
	  for (j=0; j<c->nrows; j++)
	  { t = 0.0;
	    for (k=0; k<b->nrows; k++)
		t += b->data[i+k*b->ncols]*c->data[k+j*c->ncols];
	    a->data[i+j*a->ncols]=t;
	  }
}

mat_print(a)
  Matrix a;
{
	int i,j;

	for (j=0;j<a->nrows;j++)
	{ for (i=0;i<a->ncols;i++)
	    printf("%13.6g ",a->data[i+j*a->ncols]);
	  printf("\n");
	}
}

mat_unity(a)
  Matrix a;
{
	int i,j;

	for (j=0; j<a->nrows; j++)
	  for (i=0; i<a->ncols; i++)
	    a->data[i+j*a->ncols]=(i==j)?1.0:0.0;
}

Matrix mat_alloc(ncols,nrows)
  int ncols,nrows;
{
	Matrix t;

	t=(Matrix)malloc(sizeof(struct matrix));
	t->ncols=ncols;
	t->nrows=nrows;
	t->data = (float*)malloc(ncols*nrows*sizeof(float));
	mat_unity(t);
	return t;
}

mat_copy(a,b)
  Matrix a,b;
{
	int i;

	for (i=0; i<a->ncols*a->nrows; i++)
	  a->data[i] = b->data[i];
}

rotate(a,rx,ry,rz)
  Matrix a;
  float rx,ry,rz;
{
	Matrix t,b;
	double sint,cost;

	t=mat_alloc(4,4);
	b=mat_alloc(4,4);
	mat_unity(b);
	sincos(rx*M_PI/180.,&sint,&cost);
	b->data[5]=cost;
	b->data[6] = -sint;
	b->data[9]=sint;
	b->data[10]=cost;
	matmpy(t,a,b);
	mat_unity(b);
	sincos(ry*M_PI/180.,&sint,&cost);
	b->data[0]=cost;
	b->data[2]=sint;
	b->data[8] = -sint;
	b->data[10]=cost;
	matmpy(a,t,b);
	mat_unity(b);
	sincos(rz*M_PI/180.,&sint,&cost);
	b->data[0]=cost;
	b->data[1] = -sint;
	b->data[4]=sint;
	b->data[5]=cost;
	matmpy(t,a,b);
	mat_copy(a,t);
	mat_free(t);
	mat_free(b);
}

translate(a,dx,dy,dz)
  Matrix a;
  float dx,dy,dz;
{
	Matrix b,t;

	b=mat_alloc(4,4);
	t=mat_alloc(4,4);
	mat_copy(b,a);
	mat_unity(t);
	t->data[3]=dx;
	t->data[7]=dy;
	t->data[11]=dz;
	matmpy(a,b,t);
	mat_free(b);
	mat_free(t);
}

scale(a,sx,sy,sz)
  Matrix a;
  float sx,sy,sz;
{
	Matrix b,t;

	b=mat_alloc(4,4);
	t=mat_alloc(4,4);
	mat_copy(b,a);
	mat_unity(t);
	t->data[0]=sx;
	t->data[5]=sy;
	t->data[10]=sz;
	matmpy(a,b,t);
	mat_free(b);
	mat_free(t);
}

mat_free(a)
  Matrix a;
{
	free(a->data);
	free(a);
}

#ifdef TEST
main()
{
	Matrix a, x;

	a = mat_alloc(4,4);
	mat_unity(a);

	translate(a,-.5,-.5,-.5);
	rotate(a,5.,-10.,15.);
	scale(a,.25,.25,.25);
	printf(" Transformer = [\n");
	mat_print(a);
	printf("]\n");
}
#endif
