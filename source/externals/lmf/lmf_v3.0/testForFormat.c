#include <stdlib.h>
#include <stdio.h>
#define TRERE 2
int main()
{

  FILE *fp;

  int sizeInt, sizeLong;
  unsigned long long int b = 5;
  sizeInt = sizeof(int);
  sizeLong = sizeof(long);
  fp = fopen("lmf_format.h", "w");

  /* LICENCE MESSAGE */
  fprintf(fp,
	  "/*-------------------------------------------------------\n");
  fprintf(fp, "\n");
  fprintf(fp, "           List Mode Format \n");
  fprintf(fp, "                        \n");
  fprintf(fp, "     --  lmf_format.h  --  \n");
  fprintf(fp, "\n");
  fprintf(fp, "     Luc.Simon@iphe.unil.ch\n");
  fprintf(fp, "\n");
  fprintf(fp, "     Crystal Clear Collaboration\n");
  fprintf(fp, "     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne\n");
  fprintf(fp, "\n");
  fprintf(fp, "     This software is distributed under the terms \n");
  fprintf(fp, "     of the GNU Lesser General \n");
  fprintf(fp, "     Public Licence (LGPL)\n");
  fprintf(fp, "     See LMF/LICENSE.txt for further details\n");
  fprintf(fp, "\n");
  fprintf(fp,
	  "-------------------------------------------------------*/\n");
  fprintf(fp, "\n\n\n");





  fprintf(fp, "#ifndef __FORMAT__FOR__LMF__\n");
  fprintf(fp, "#define __FORMAT__FOR__LMF__\n");


  fprintf(fp, "typedef char i8;\n");
  fprintf(fp, "typedef unsigned char u8;\n");
  fprintf(fp, "typedef short i16;\n");
  fprintf(fp, "typedef unsigned short u16;\n");

  if (sizeLong == 4) {
    fprintf(fp, "typedef long i32;\n");
    fprintf(fp, "typedef unsigned long u32;\n");

  } else if (sizeInt == 4) {
    fprintf(fp, "typedef int i32;\n");
    fprintf(fp, "typedef unsigned int u32;\n");
  }

  if (sizeLong == 8) {
    fprintf(fp, "typedef long i64;\n");
    fprintf(fp, "typedef unsigned long u64;\n");
  } else {
#if defined( _MSC_VER ) || ( defined( WIN32 ) && !defined( __MINGW32__ ) )
    fprintf(fp, "typedef unsigned __int64    u64;\n");
    fprintf(fp, "typedef signed __int64      i64;\n");
#else
    fprintf(fp, "typedef unsigned long long int u64;\n");
    fprintf(fp, "typedef long long int          i64;\n");
#endif

  }


  fprintf(fp, "#endif\n");

  fclose(fp);
}
