/*  19-sep-2002: 
 *     Merge with bug fixes and support for CYGWIN provided by Kris Thielemans@csc.mrc.ac.uk
 *  12-sep-2003:
 *     Move ntohs and ntohl definitions to machine_indep.h
 */
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "machine_indep.h"
#define OK 0
#define ERROR -1

#if !defined(OLD_C)
void SWAB(const void *from, void *to, int length)
#else
void SWAB( from, to, length)
char *from, *to;
int length;
#endif
{
	if (ntohs(1) == 1) swab((char *)from, to, length);
	else memcpy(to,from,length);
}

void SWAW ( from, to, length)
#if !defined(OLD_C) 
const short *from;
#else
short *from;
#endif
short *to;
int length;
{
	if (ntohs(1) == 1) swaw((short*)from,to,length);
	else memcpy(to,from,length*2);
} 

float vaxftohf( bufr, off)
unsigned short *bufr;
int off;
{
	unsigned int sign_exp, high, low, mantissa, ret;
	unsigned u = (bufr[off+1] << 16) + bufr[off];
	
	if (u == 0) return 0.0;	
	sign_exp = u & 0x0000ff80;
	sign_exp = (sign_exp - 0x0100) & 0xff80; 
	high = u & 0x0000007f;
	low  = u & 0xffff0000;
	mantissa = (high << 16) + (low >> 16);
	sign_exp = sign_exp << 16;
	ret = sign_exp + mantissa;
	return *(float*)(&ret);
}

#if defined(__alpha) || defined(_WIN32) /* LITTLE_ENDIAN : alpha, intel */
void
ftovaxf(f, bufr)
float f;
unsigned short *bufr;
{
	unsigned *u, sign_exp, low, high, mantissa, ret;
	float f_tmp = f;

	u = (unsigned*)(&f_tmp);
	if (u == 0)  {
		bufr[0]	= bufr[1] = 0;
		return;
	}
	sign_exp = *u & 0xff800000;
	sign_exp = sign_exp >> 16;
	sign_exp = (sign_exp + 0x0110) & 0xff80; 
	low   = *u & 0x0000ffff;
	high  = *u & 0x007f0000;
	mantissa = (high >> 16) + (low << 16);
	ret = sign_exp + mantissa;
	bufr[0] = ret;
	bufr[1] =  ret >>16;
}
#else  /* BIG ENDIAN : sun hp sgi*/
void
ftovaxf(orig,number)
  unsigned short number[2];
  float orig;
{

  	/* convert from sun float to vax float */

  	union {
	 	  unsigned short t[2]; 
		  float t4;
	      } test;
	unsigned short int exp;

	number[0] = 0;
	number[1] = 0;

	test.t4 = orig;
	if (test.t4 == 0.0) return;

	number[1] = test.t[1];

	exp = ((test.t[0] & 0x7f00) + 0x0100) & 0x7f00;
	test.t[0] = (test.t[0] & 0x80ff) + exp;

	number[0] = test.t[0];

}
#endif /* LITTLE VS BIG ENDIAN*/

int file_data_to_host(dptr, nblks, dtype)
char *dptr;
int nblks, dtype;
{
	int i, j;
	char *tmp = NULL;


	matrix_errno = 0;
	matrix_errtxt[0] = '\0';
	if ((tmp = malloc(512)) == NULL) return ERROR;
	switch(dtype)
	{
	case ByteData:
		break;
	case VAX_Ix2:
		if (ntohs(1) == 1) 
			for (i=0, j=0; i<nblks; i++, j+=512) {
				swab( dptr+j, tmp, 512);
				memcpy(dptr+j, tmp, 512);
			}
		break;
	case VAX_Ix4:
		if (ntohs(1) == 1)
			for (i=0, j=0; i<nblks; i++, j+=512) {
				swab(dptr+j, tmp, 512);
				swaw((short*)tmp, (short*)(dptr+j), 256);
			}
		break;
	case VAX_Rx4:
	 	if (ntohs(1) == 1) 
			 for (i=0, j=0; i<nblks; i++, j+=512) {
				swab( dptr+j, tmp, 512);
/* remove next line (fix from Yohan.Nuyts@uz.kuleuven.ac.be, 28-oct-97)
				swaw((short*)tmp, (short*)(dptr+j), 256);
*/
			}
		for (i=0; i<nblks*128; i++)
			((float *)dptr)[i] = vaxftohf( (unsigned short *)dptr, i*2) ;
		break;
	case SunShort:
		if (ntohs(1) != 1)
			for (i=0, j=0; i<nblks; i++, j+=512) {
				swab(dptr+j, tmp, 512);
				memcpy(dptr+j, tmp, 512);
			}
		break;
	case SunLong:
	case IeeeFloat:
		if (ntohs(1) != 1) 
			for (i=0, j=0; i<nblks; i++, j+=512) {
				swab(dptr+j, tmp, 512);
				swaw((short*)tmp, (short*)(dptr+j), 256);
			}
		break;
	default:	/* something else...treat as Vax I*2 */
		if (ntohs(1) == 1)
			for (i=0, j=0; i<nblks; i++, j+=512) {
				swab(dptr+j, tmp, 512);
				memcpy(dptr+j, tmp, 512);
			}
		break;
	}
	free(tmp);
	return OK;
}

int 
read_raw_acs_data(fname, strtblk, nblks, dptr, dtype)
	char           *fname;
	int             strtblk, nblks, dtype;
	char           *dptr;
{
#ifndef _WIN32
	int             i, err;

	for (i = 0; i < nblks; i++) {
		err = rtsRblk(fname, strtblk + i, dptr + 512 * i);
		if (err) return -1;
	}
	return file_data_to_host(dptr, nblks, dtype);
#else
	return -1;
#endif
}


read_matrix_data( fptr, strtblk, nblks, dptr, dtype)
  FILE *fptr;
  int strtblk, nblks, dtype;
  char * dptr;
{
	int  err;

	err = mat_rblk( fptr, strtblk, dptr, nblks);
	if (err) return -1;
	return file_data_to_host(dptr,nblks,dtype);
}

write_matrix_data( fptr, strtblk, nblks, dptr, dtype)
FILE *fptr;
int strtblk, nblks, dtype;
char *dptr;
{
	int error_flag = 0;
	int i, j, k;
	char *bufr1 = NULL, *bufr2 = NULL;

	matrix_errno = 0;
	matrix_errtxt[0] = '\0';
	if ( (bufr1 = malloc(512)) == NULL) return ERROR;
	if ( (bufr2 = malloc(512)) == NULL) {
		free(bufr1);
		return ERROR;
	}
	switch( dtype)
	{
	case ByteData:
		if ( mat_wblk( fptr, strtblk, dptr, nblks) < 0) error_flag++;
		break;
	case VAX_Ix2: 
	default:	/* something else...treat as Vax I*2 */
		if (ntohs(1) == 1) {
			for (i=0, j=0; i<nblks && !error_flag; i++, j += 512) {
				swab( dptr+j, bufr1, 512);
				memcpy(bufr2, bufr1, 512);
				if ( mat_wblk( fptr, strtblk+i, bufr2, 1) < 0) error_flag++;
			}
		} else if ( mat_wblk( fptr, strtblk, dptr, nblks) < 0) error_flag++;
		break;
	case VAX_Ix4:
		if (ntohs(1) == 1)  {
			for (i=0, j=0; i<nblks && !error_flag; i++, j += 512) {
				swab( dptr, bufr1, 512);
				swaw( (short*)bufr1, (short*)bufr2, 256);
				if ( mat_wblk( fptr, strtblk+i, bufr2, 1) < 0) error_flag++;
			} 
		} else if ( mat_wblk( fptr, strtblk, dptr, nblks) < 0) error_flag++;
		break;
	case VAX_Rx4:
		k = 0;
		for (i=0; i<nblks; i++) {
			for (j=0; j<512; j += sizeof(float), k += sizeof(float)) 
				ftovaxf(*((float*)&dptr[k]),&bufr2[j]);
			if ( mat_wblk( fptr, strtblk+i, bufr2, 1) < 0) error_flag++;
		}
		break;
	case IeeeFloat:
	case SunLong:
		if (ntohs(1) != 1) {
			for (i=0, j=0; i<nblks && !error_flag; i++, j += 512) {
				swab( dptr+j, bufr1, 512);
				swaw( (short*)bufr1, (short*)bufr2, 256);
				if ( mat_wblk( fptr, strtblk+i, bufr2, 1) < 0) error_flag++;
			}
		} else if ( mat_wblk( fptr, strtblk, dptr, nblks) < 0) error_flag++;
		break;
	case SunShort:
		if (ntohs(1) != 1) {
			for (i=0, j=0; i<nblks && !error_flag; i++, j += 512) {
				swab( dptr+j, bufr1, 512);
				memcpy(bufr2, bufr1, 512);
				if ( mat_wblk( fptr, strtblk+i, bufr2, 1) < 0) error_flag++;
			}
		} else if ( mat_wblk( fptr, strtblk, dptr, nblks) < 0) error_flag++;
		break;
	}
	free(bufr1);
	free(bufr2);
	if (error_flag == 0) return OK;
	return ERROR;
}


/* buf...(...) - functions to handle copying header data to and from a buffer
   in the most type safe way possible; note that i is incremented
   by the size of the item copied so these functions must be
   called in the right order
*/


void bufWrite(s, buf, i, len)
char *s, *buf;
int *i, len;
{
   strncpy(&buf[*i], s, len);
    *i += len;
}

#ifdef __STDC__
void bufWrite_s(short val, char *buf, int *i)
#else
void bufWrite_s(val, buf, i)
short val;
char *buf;
int *i;
#endif
{
	union { short s; char b[2]; } tmp;
	tmp.s = val;
	if (ntohs(1) != 1) swab(tmp.b,&buf[*i],2);
    else memcpy(&buf[*i], tmp.b, sizeof(short));
    *i += sizeof(short);
}

void bufWrite_i(val, buf, i)
int val, *i;
char *buf;
{
	union { int i; char b[4]; } tmp;
	union { short s[2]; char b[4]; } tmp1;
	tmp.i = val;
	if (ntohs(1) != 1) {
		swab(tmp.b,tmp1.b,4);
		swaw(tmp1.s,(short*)&buf[*i],2);
	} else memcpy(&buf[*i], tmp.b, sizeof(int));
    *i += sizeof(int);
}

void bufWrite_u(val, buf, i)
unsigned int val;
int *i;
char *buf;
{
	union { int u; char b[4]; } tmp;
	union { short s[2]; char b[4]; } tmp1;
	tmp.u = val;
	if (ntohs(1) != 1) {
		swab(tmp.b,tmp1.b,4);
		swaw(tmp1.s,(short*)&buf[*i],2);
	} else memcpy(&buf[*i], tmp.b, sizeof(unsigned int));
    *i += sizeof(unsigned int);
}

#if !defined(OLD_C) 
void bufWrite_f(float val, char *buf, int *i)
#else
void bufWrite_f(val, buf, i)
float val;
char *buf;
int *i;
#endif
{
	union { float f; char b[4]; } tmp;
	union { short s[2]; char b[4]; } tmp1;
	tmp.f = val;
	if (ntohs(1) != 1) {
		swab(tmp.b,tmp1.b,4);
		swaw(tmp1.s,(short*)&buf[*i],2);
	} else memcpy(&buf[*i], tmp.b, sizeof(float));
    *i += sizeof(float);
}

void bufRead(s, buf, i, len)
char *s, *buf;
int *i, len;
{
    strncpy(s, &buf[*i], len);
    *i += len;
}

void bufRead_s(val, buf, i)
short* val;
char* buf;
int* i;
{
	union { short s; u_char b[2]; } tmp, tmp1;
	memcpy(tmp.b,&buf[*i],2);
	if (ntohs(1) != 1) {
		swab((char*)tmp.b,(char*)tmp1.b,2);
		*val = tmp1.s;
	} else *val = tmp.s;
    *i += sizeof(short);
}

void bufRead_i(val, buf, i)
int* val;
char* buf;
int* i;
{
	union {int i; u_char b[4]; } tmp, tmp1;
	memcpy(tmp1.b,&buf[*i],4);
	if (ntohs(1) != 1) {
		swab((char*)tmp1.b,(char*)tmp.b,4);
		swaw((short*)tmp.b,(short*)tmp1.b,2);
	}
	*val = tmp1.i;
    *i += sizeof(int);
}

void bufRead_u(val, buf, i)
unsigned int* val;
char* buf;
int* i;
{
	union {unsigned int u; u_char b[4]; } tmp, tmp1;
	memcpy(tmp1.b,&buf[*i],4);
	if (ntohs(1) != 1) {
		swab((char*)tmp1.b,(char*)tmp.b,4);
		swaw((short*)tmp.b,(short*)tmp1.b,2);
	}
	*val = tmp1.u;
    *i += sizeof(unsigned int);
}

void bufRead_f(val, buf, i)
float* val;
char* buf;
int* i;
{
	union {float f; u_char b[2]; } tmp, tmp1;
    memcpy(tmp1.b, &buf[*i], sizeof(float));
	if (ntohs(1) != 1) {
		swab((char*)tmp1.b,(char*)tmp.b,4);
		swaw((short*)tmp.b,(short*)tmp1.b,2);
	}
	*val = tmp1.f;
    *i += sizeof(float);
}
