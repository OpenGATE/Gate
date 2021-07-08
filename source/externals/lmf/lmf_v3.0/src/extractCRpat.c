/*-------------------------------------------------------

           List Mode Format 
                        
     --  extractCRpat.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of extractCRpat.c:
     This function fills a countRateHeader structure.
     It needs the count rate encoding pattern (u16)
     Ex : 
     pCRH = extractCRpat(pattern);


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

static int allocCRHdone = 0;
static COUNT_RATE_HEADER *pCRH;
COUNT_RATE_HEADER(*extractCRpat(u16 pattern))
{

  if (allocCRHdone == 0) {
    if ((pCRH =
	 (COUNT_RATE_HEADER *) malloc(sizeof(COUNT_RATE_HEADER))) == NULL)
      printf
	  ("\n***ERROR : in extractCRpat.c : impossible to do : malloc()\n");
    allocCRHdone = 1;
  }
/* // s   singles CR */
  if ((pattern & BIT12) == 0) {
    pCRH->singleRateBool = 0;
    pCRH->singleRatePart = 0;
  } else {
    pCRH->singleRateBool = 1;

    /*  // SS  */

    pCRH->singleRatePart = (u8) ((pattern & (BIT10 + BIT11)) >> 9);
  }
/* // c */
  if ((pattern & BIT9) == 0)
    pCRH->totalCoincidenceBool = 0;
  else
    pCRH->totalCoincidenceBool = 1;

  /* // F */
  if ((pattern & BIT8) == 0)
    pCRH->totalRandomBool = 0;
  else
    pCRH->totalRandomBool = 1;

  /* // r */
  if ((pattern & BIT7) == 0)
    pCRH->angularSpeedBool = 0;
  else
    pCRH->angularSpeedBool = 1;

  /* // b */
  if ((pattern & BIT6) == 0)
    pCRH->axialSpeedBool = 0;
  else
    pCRH->axialSpeedBool = 1;

/* // Check if the RRRRR Reserved bytes are 00000 */
  if ((pattern & 31) != 0) {
    printf("\nWARNING : Reserved bytes are not 00000\n");
    printf("* Message from extractCRpat.c for countRatepattern = %d\n",
	   pattern);
  }
  return (pCRH);
}

void destroyExtractCRpat()
{
  free(pCRH);

  allocCRHdone = 0;
}
