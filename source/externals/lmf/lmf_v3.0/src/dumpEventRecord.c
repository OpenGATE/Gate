/*-------------------------------------------------------

           List Mode Format 
                        
     --  dumpEventRecord.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dumpEventRecord.c:
     This function called by dumpTheRecord()
     dispays on screen the containing of a
     event record structure.


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"




void dumpEventRecord(const ENCODING_HEADER * pEncoH,
		     const EVENT_HEADER * pEH, const EVENT_RECORD * pER)
{
  static int j;
  static u16 *pcrist;
  static u64 timeInMillis = 0;

  printf("TIME =\n");
  if (pEH->coincidenceBool == FALSE) {
    for (j = 0; j < 8; j++)
      printf("%x\t", pER->timeStamp[j]);

    printf("\n---> %llu picoseconds\n", getTimeOfThisEVENT(pER));

  } else {
    for (j = 2; j >= 0; j--)
      printf("%x\t", pER->timeStamp[j]);

    timeInMillis = pER->timeStamp[2] +
	256 * pER->timeStamp[1] + 256 * 256 * pER->timeStamp[0];

    printf("\n---> %llu milliseconds\n", getTimeOfThisCOINCI(pER));
    printf("Time of flight = %.1f nano-seconds\n", getTimeOfFlightOfThisCOINCI(pER));
  }

  if (pEH->detectorIDBool == TRUE) {
    printf("\n");
    printf("Main Crystal ID = ");
    printf("%llx --->\n", pER->crystalIDs[0]);
    pcrist = demakeid(pER->crystalIDs[0], pEncoH);
    printf("Rsector = %d\n", pcrist[4]);
    printf("Module = %d\n", pcrist[3]);
    printf("Submodule = %d\n", pcrist[2]);
    printf("Crystal = %d\n", pcrist[1]);
    printf("Layer = %d\n", pcrist[0]);
    free(pcrist);
  }

  if ((pEH->detectorIDBool == TRUE) && (pEH->coincidenceBool == TRUE)) {
    printf("\n");
    printf("Second Crystal ID = ");
    printf("%llx --->\n", pER->crystalIDs[pEH->numberOfNeighbours + 1]);
    pcrist =
	demakeid(pER->crystalIDs[pEH->numberOfNeighbours + 1], pEncoH);
    printf("Rsector = %d\n", pcrist[4]);
    printf("Module = %d\n", pcrist[3]);
    printf("Submodule = %d\n", pcrist[2]);
    printf("Crystal = %d\n", pcrist[1]);
    printf("Layer = %d\n", pcrist[0]);
    free(pcrist);
  }
  if (pEH->gantryAngularPosBool == TRUE) {
    printf("\n");
    printf("Gantry Angular Position = %d\n", pER->gantryAngularPos);
  }
  if (pEH->gantryAxialPosBool == TRUE) {
    printf("\n");
    printf("Gantry Axial Position = %d\n", pER->gantryAxialPos);
  }
  if (pEH->sourcePosBool == TRUE) {
    printf("\n");
    printf("Source Axial Position = %d\n", pER->sourceAxialPos);
    printf("Source Angular Position = %d\n", pER->sourceAngularPos);
  }

  if (pEH->energyBool == TRUE) {
    printf("\n");
    printf("Energy in main crystal = %d --> %d keV\n", pER->energy[0],
	   pER->energy[0] * getEnergyStepFromCCH());

    if (pEH->neighbourBool == TRUE) {
      for (j = 1; j <= pEH->numberOfNeighbours; j++)
	printf("Energy in neighbour %d of main crystal = %d --> %d keV\n",
	       j, pER->energy[j],
	       pER->energy[j] * getEnergyStepFromCCH());
    }

    if (pEH->coincidenceBool == TRUE) {
      printf("\n");
      printf("Energy in second crystal = %d --> %d keV\n",
	     pER->energy[pEH->numberOfNeighbours + 1],
	     pER->energy[pEH->numberOfNeighbours +
			 1] * getEnergyStepFromCCH());
      if (pEH->neighbourBool == TRUE) {
	for (j = pEH->numberOfNeighbours + 2;
	     j <= (2 * pEH->numberOfNeighbours) + 1; j++)
	  printf
	      ("Energy in neighbour %d of second crystal = %d --> %d keV\n",
	       j, pER->energy[j],
	       pER->energy[j] * getEnergyStepFromCCH());
      }
    }

  }

  if (pEH->fpgaNeighBool) {
    printf("\n");
    printf("FPGA Neigh info = %d\n", pER->fpgaNeighInfo[0]);
    if (pEH->coincidenceBool == TRUE) {
      printf("FPGA Neigh info (2nd crystal) = %d\n",
	     pER->fpgaNeighInfo[1]);

    }
  }


}
