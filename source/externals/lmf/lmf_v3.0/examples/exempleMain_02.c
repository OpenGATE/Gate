/*-------------------------------------------------------

           List Mode Format 
                        
     --  exempleMain_02.c  --                      

     Magalie.krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/
/*------------------------------------------------------------------------

			   
	 Description : 
	 Explain how to use the libLMF.a tools, for the binary part of LMF.
         This one displays XYZ position in an event record file. 

---------------------------------------------------------------------------*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../includes/lmf.h"
#include <time.h>

int main()
{

  static struct LMF_ccs_encodingHeader *pEncoH = NULL;

  double x1, y1, z1, x2, y2, z2;

  FILE *pfCCS = NULL;
  system("ls *ccs *cch");
  if (LMFcchReader(""))
    exit(EXIT_FAILURE);
  pfCCS = open_CCS_file2(ccsFileName);	/* // LMF binary file   */
  fseek(pfCCS, 0L, 0);
  pEncoH = readHead(pfCCS);	/*  // read head of ccs */


  while (findXYZinLMFfile(pfCCS, &x1, &y1, &z1, &x2, &y2, &z2, pEncoH)) {
    printf("\n\nOne coinci event read : \n");
    printf("first : %f\t", x1);
    printf("%f\t", y1);
    printf("%f\t and second : ", z1);
    printf("%f\t", x2);
    printf("%f\t", y2);
    printf("%f\n", z2);
  }

  if (pfCCS) {
    fclose(pfCCS);
    printf("File CCS closed\n");
  }

  destroy_findXYZinLMFfile(pEncoH);
  LMFcchReaderDestructor();
  destroyReadHead();
  printf("\nmain over\n");
  return (EXIT_SUCCESS);


}
