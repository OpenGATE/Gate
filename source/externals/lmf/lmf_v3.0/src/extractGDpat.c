/*-------------------------------------------------------

           List Mode Format 
                        
     --  extractGDpat.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of extractGDpat.c:
     This function fill a gate digi Header structure.
     It needs the gate digi encoding pattern (u16)
     Ex : 
     pGDH = extractGDpat(pattern);


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


static int allocGDHdone = 0;
static GATE_DIGI_HEADER *pGDH;

GATE_DIGI_HEADER(*extractGDpat(u16 pattern))
{

  if (allocGDHdone == 0) {
    if ((pGDH =
	 (GATE_DIGI_HEADER *) malloc(sizeof(GATE_DIGI_HEADER))) == NULL)
      printf
	  ("\n***ERROR : in extractGDpat.c : impossible to do : malloc()\n");
    allocGDHdone = 1;
  }


/*   // TTTT = 1100 ? */
  if ((pattern & (BIT16 + BIT15)) != (BIT16 + BIT15)) {
    printf("*** error : extractGDpat.c : wrong tag of gate digi header\n");
    exit(0);
  } else if ((pattern & BIT14) == BIT14) {
    printf("*** error : extractGDpat.c : wrong tag of gate digi header\n");
    exit(0);
  } else if ((pattern & BIT13) == BIT13) {
    printf("*** error : extractGDpat.c : wrong tag of gate digi header\n");
    exit(0);
  }





/*    // C */
  if ((pattern & BIT12) == 0)
    pGDH->comptonBool = 0;
  else
    pGDH->comptonBool = 1;



  /*  // S */
  if ((pattern & BIT11) == 0)
    pGDH->sourceIDBool = 0;
  else
    pGDH->sourceIDBool = 1;



  /* // p */
  if ((pattern & BIT10) == 0)
    pGDH->sourceXYZPosBool = 0;
  else
    pGDH->sourceXYZPosBool = 1;




  /* // e */
  if ((pattern & BIT9) == 0)
    pGDH->eventIDBool = 0;
  else
    pGDH->eventIDBool = 1;



  /* // r    */
  if ((pattern & BIT8) == 0)
    pGDH->runIDBool = 0;
  else
    pGDH->runIDBool = 1;

  /* // G */
  if ((pattern & BIT7) == 0)
    pGDH->globalXYZPosBool = 0;
  else
    pGDH->globalXYZPosBool = 1;

  /* // M */
  if ((pattern & BIT6) == 0)
    pGDH->multipleIDBool = 0;
  else
    pGDH->multipleIDBool = 1;
  /* // D */
  if ((pattern & BIT5) == 0)
    pGDH->comptonDetectorBool = 0;
  else
    pGDH->comptonDetectorBool = 1;





/* // Check if the R Reserved bits are 0 */
  if ((pattern & (BIT1 + BIT2 + BIT3 + BIT4)) != 0) {
    printf("\nWARNING : Reserved bytes are not 0 0000\n");
    printf("* Message from extractGDpat.c for gateDigipattern = %d\n",
	   pattern);
  }
  return (pGDH);
}

void destroyExtractGDpat()
{

  if (pGDH)
    free(pGDH);

  allocGDHdone = 0;
}
