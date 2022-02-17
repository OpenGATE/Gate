/*-------------------------------------------------------

           List Mode Format 
                        
     --  dumpEventHeader.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dumpEventHeader.c:
     This function called by dumpTheRecord()
     dispays on screen the containing of a
     event header structure.


-------------------------------------------------------*/
#include <stdio.h>
#include "lmf.h"

void dumpEventHeader(const EVENT_HEADER * pEH)
{

  printf("\n");
  printf("--> coincidenceBool = %d\n", pEH->coincidenceBool);	/* Coincidence if 1, singles if 0 */
  printf("--> detectorIDBool = %d\n", pEH->detectorIDBool);	/* detector ID recorded if 1 */
  printf("--> energyBool = %d\n", pEH->energyBool);	/* energy recorded if 1 */
  printf("--> neighbourBool = %d\n", pEH->neighbourBool);	/* energy of neighbours recorded if 1 */
  printf("--> neighbourhoodOrder = %d\n", pEH->neighbourhoodOrder);	/* 0, 1, 2 , or 3 (cf fig. 1) */
  printf("--> numberOfNeighbours = %d\n", pEH->numberOfNeighbours);	/*Number of neighbours */
  printf("--> gantryAxialPosBool = %d\n", pEH->gantryAxialPosBool);	/* gantry's axial position */
  printf("--> gantryAngularPosBool = %d\n", pEH->gantryAngularPosBool);	/* gantry's angular position */
  printf("--> sourcePosBool = %d\n", pEH->sourcePosBool);	/* source's position */
  printf("--> gateDigiBool = %d\n", pEH->gateDigiBool);	/* source's position */
  printf("--> fpgaNeighBool= %d\n", pEH->fpgaNeighBool);	/* fpga neigh info position */



}
