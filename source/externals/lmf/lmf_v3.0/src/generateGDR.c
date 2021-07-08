/*-------------------------------------------------------

           List Mode Format 
                        
     --  generateGDR.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of generateGDR.c:

   Example of filling of gate digi record

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"



static GATE_DIGI_RECORD *pGDR;



GATE_DIGI_RECORD(*generateGDR
		 (ENCODING_HEADER * pEncoH, GATE_DIGI_HEADER * pGDH))
{
  static int verboseLevel;
  i16 j;
  static i16 allocGDRdone = 0;

  if (allocGDRdone == 0) {	/* the allocation is just done once */
    allocGDRdone = 1;

    printf
	("verbose level for artificial LMF builder (gate digi record builder) :\n");
    verboseLevel = hardgeti16(0, 1);

    if ((pGDR =
	 (GATE_DIGI_RECORD *) malloc(sizeof(GATE_DIGI_RECORD))) == NULL)
      printf
	  ("\n***ERROR : in generateGDR.c : impossible to do : malloc()\n");



  }

  /* // generate an evoluting random time // just for developement */

  pGDR->runID = 10;



  pGDR->eventID[0] = 2;
  pGDR->eventID[1] = 2;

  pGDR->sourceID[0] = 3;
  pGDR->sourceID[1] = 4;

  pGDR->sourcePos[0].X = 1;
  pGDR->sourcePos[0].Y = 8;
  pGDR->sourcePos[0].Z = 25;
  pGDR->sourcePos[1].X = 10;
  pGDR->sourcePos[1].Y = 28;
  pGDR->sourcePos[1].Z = 35;

  for (j = 0; j < 42; j++) {

    pGDR->globalPos[j].X = 1;
    pGDR->globalPos[j].Y = 48;
    pGDR->globalPos[j].Z = 55;

  }

  pGDR->numberCompton[0] = 3;
  pGDR->numberCompton[1] = 4;

  pGDR->multipleID = 0;

  return (pGDR);

}


void generateGDRDestructor()
{
  if (pGDR) {

    free(pGDR);
  }

}
