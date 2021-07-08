/*-------------------------------------------------------

           List Mode Format 
                        
     --  extractEpat.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of extractEpat.c:

     This function fills an eventHeader structure.
     It needs the event encoding pattern (u16)
     Ex : 
     pEH = extractEpat(pattern);


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


static EVENT_HEADER *pEH;
static int allocEHdone = 0;
EVENT_HEADER(*extractEpat(u16 pattern))
{


  if (allocEHdone == 0) {
    if ((pEH = (EVENT_HEADER *) malloc(sizeof(EVENT_HEADER))) == NULL)
      printf
	  ("\n***ERROR : in extractEpat : impossible to do : malloc()\n");
    allocEHdone = 1;
  }
/* // c  coincidence */
  if ((pattern & BIT12) == 0)	/*AND 0000 1000 0000 0000 */
    pEH->coincidenceBool = 0;
  else
    pEH->coincidenceBool = 1;

  /* // d  detector id */
  if ((pattern & BIT11) == 0)	/*AND 0000 0100 0000 0000 */
    pEH->detectorIDBool = 0;
  else
    pEH->detectorIDBool = 1;

  /* // E energy */
  if ((pattern & BIT10) == 0) {	/*AND 0000 0010 0000 0000 */
    pEH->energyBool = 0;
    pEH->neighbourBool = 0;
    pEH->neighbourhoodOrder = 0;
    pEH->numberOfNeighbours = 0;
  } else {
    pEH->energyBool = 1;

    /*  // n neigh */
    if ((pattern & BIT9) == 0) {
      pEH->neighbourBool = 0;
      pEH->neighbourhoodOrder = 0;
      pEH->numberOfNeighbours = 0;
    } else {
      pEH->neighbourBool = 1;

      /*  // NN order of neighbourhood */
      if ((pattern & (BIT7 + BIT8)) == 0)	/*AND 0000 0000 1100 0000 */
	pEH->neighbourhoodOrder = 0;
      else if ((pattern & (BIT7 + BIT8)) == BIT7)
	pEH->neighbourhoodOrder = 1;
      else if ((pattern & (BIT7 + BIT8)) == BIT8)
	pEH->neighbourhoodOrder = 2;
      else if ((pattern & (BIT7 + BIT8)) == (BIT7 + BIT8))
	pEH->neighbourhoodOrder = 3;

      pEH->numberOfNeighbours = findvnn2((u16) pEH->neighbourhoodOrder);
    }
  }
  /* // g gantry ang pos */
  if ((pattern & BIT6) == 0)	/* AND 0000 0000 0010 0000 */
    pEH->gantryAngularPosBool = 0;
  else
    pEH->gantryAngularPosBool = 1;

  /* // b gantry axi pos */
  if ((pattern & BIT5) == 0)	/*AND 0000 0000 0001 0000 */
    pEH->gantryAxialPosBool = 0;
  else
    pEH->gantryAxialPosBool = 1;

/*  // s source */
  if ((pattern & BIT4) == 0)	/*AND 0000 0000 0000 1000 */
    pEH->sourcePosBool = 0;
  else
    pEH->sourcePosBool = 1;

  /* // G gate info */
  if ((pattern & BIT3) == 0)	/*AND 0000 0000 0000 0100 */
    pEH->gateDigiBool = 0;
  else
    pEH->gateDigiBool = 1;

  /* // Z fpga neigh  info */
  if ((pattern & BIT2) == 0)	/*AND 0000 0000 0000 0010 */
    pEH->fpgaNeighBool = 0;
  else
    pEH->fpgaNeighBool = 1;





/*  // R Check if Reserved bytes are 0 */
  if ((pattern & (BIT1)) != 0) {	/*AND 0000 0000 0000 0001 */
    printf("* \nWarning : R Reserved bytes are not 0\n");
    printf("* Message from extractEpat.c for Eventpattern = %d\n",
	   pattern);
  }
  return (pEH);
}


void destroyExtractEpat()
{
  free(pEH);

  allocEHdone = 0;
}
