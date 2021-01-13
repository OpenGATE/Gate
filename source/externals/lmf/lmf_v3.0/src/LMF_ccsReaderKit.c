/*-------------------------------------------------------

           List Mode Format 
                        
     --  LMF_ccsReaderKit.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of LMF_ccsReaderKit.c:
     Open and close the LMF .ccs binary file

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include "lmf.h"



FILE *openCCSfile(const i8 * nameOfFile)
{
  FILE *pfile;

  pfile = fopen(nameOfFile, "rb");
  printf("\tOpen %s ok...\n", nameOfFile);

  return (pfile);
}


int closeCCSfile(FILE * pfile)
{
  if (pfile) {
    if (!fclose(pfile))
      return (0);
    else {
      printf
	  ("\n*** error : LMF_ccsReaderKit.c : closeCCSfile : impossible to close this file\n");
      return (1);
    }
  } else {
    printf
	("\n*** error : LMF_ccsReaderKit.c : closeCCSfile : impossible to close a NULL file\n");
    return (1);
  }
}
