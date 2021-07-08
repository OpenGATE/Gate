/*-------------------------------------------------------

           List Mode Format 
                        
     --  dumpGateDigiRecord.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dumpGateDigiRecord.c:
     This function called by dumpTheRecord()
     dispays on screen the containing of a
     gate digi record structure.



-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"




void dumpGateDigiRecord(const ENCODING_HEADER * pEncoH,
			const EVENT_HEADER * pEH,
			const GATE_DIGI_HEADER * pGDH,
			const GATE_DIGI_RECORD * pGDR)
{
  static int j;
  printf("\n");
  if (pGDH->runIDBool) {
    printf("runID = %d\n", (int)pGDR->runID);
  }
  if (pGDH->eventIDBool) {
    printf("eventID = %d", (int)pGDR->eventID[0]);
    if (pEH->coincidenceBool)
      printf("\t 2nd : eventID = %d", (int)pGDR->eventID[1]);
    printf("\n");
  }

  if ((pGDH->multipleIDBool) && (pEH->coincidenceBool))
    printf("multipleID = %d\n", (int)pGDR->multipleID);


  if (pGDH->sourceIDBool) {
    printf("sourceID = %hu", pGDR->sourceID[0]);
    if (pEH->coincidenceBool)
      printf("\t 2nd : sourceID = %hu", pGDR->sourceID[1]);
    printf("\n");
  }
  if (pGDH->sourceXYZPosBool) {
    printf("source pos = %d %d %d", pGDR->sourcePos[0].X,
	   pGDR->sourcePos[0].Y, pGDR->sourcePos[0].Z);
    if (pEH->coincidenceBool)
      printf("\t 2nd :  %d %d %d", pGDR->sourcePos[1].X,
	     pGDR->sourcePos[1].Y, pGDR->sourcePos[1].Z);
    printf("\n");
  }
  if (pGDH->globalXYZPosBool) {
    printf("global pos = %d %d %d\n", pGDR->globalPos[0].X,
	   pGDR->globalPos[0].Y, pGDR->globalPos[0].Z);

    if (pEH->neighbourBool == TRUE) {
      for (j = 1; j < pEH->numberOfNeighbours; j++) {
	printf("global pos of neigh %d = %d %d %d\n", j,
	       pGDR->globalPos[j].X,
	       pGDR->globalPos[j].Y, pGDR->globalPos[j].Z);
      }
    }

    if (pEH->coincidenceBool) {
      printf("\n2nd global pos. : %d %d %d\n",
	     pGDR->globalPos[pEH->numberOfNeighbours + 1].X,
	     pGDR->globalPos[pEH->numberOfNeighbours + 1].Y,
	     pGDR->globalPos[pEH->numberOfNeighbours + 1].Z);

      if (pEH->neighbourBool == TRUE) {
	for (j = pEH->numberOfNeighbours + 2;
	     j <= (2 * pEH->numberOfNeighbours) + 1; j++) {
	  printf("global pos of neigh %d = %d %d %d\n", j,
		 pGDR->globalPos[j].X, pGDR->globalPos[j].Y,
		 pGDR->globalPos[j].Z);
	}
      }



    }
    printf("\n");
  }

  if (pGDH->comptonBool) {
    printf("number of compton in phantom: %d  \n", pGDR->numberCompton[0]);
    if (pEH->coincidenceBool)
      printf("number of compton in phantom (2nd) %d : \n",
	     pGDR->numberCompton[1]);


    printf("\n");


  }
  if (pGDH->comptonDetectorBool) {
    printf("number of compton in detector: %d  \n",
	   pGDR->numberDetectorCompton[0]);
    if (pEH->coincidenceBool)
      printf("number of compton in detector (2nd) %d : \n",
	     pGDR->numberDetectorCompton[1]);


    printf("\n");


  }

}
