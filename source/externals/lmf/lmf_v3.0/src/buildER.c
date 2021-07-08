/*-------------------------------------------------------

           List Mode Format 
                        
     --  buildER.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of buildER.c:
     Called by LMFbuilder(). This function builds
     an event record. It needs the encoding header
     and event header structures (pEncoH and pEH)
     that affect the event record size.


-------------------------------------------------------*/
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

void buildER(const EVENT_HEADER * pEH,
	     u16 encodingIDSize, const EVENT_RECORD * pER, FILE * pf)
{
  u16 pbufi16[18];
  u8 pbufi8[50];
  i16 i = 0, j = 0, k = 0;	/*Counters */
  u16 size = 0;
  u64vsu16 crystalIDs;

  /* TAG, TIME STAMP, AND TIME OF FLIGHT */
  if (pEH->coincidenceBool == FALSE) {
    for (j = 0; j < 8; j++) {	/*     // we inverse the time i8 */
      pbufi8[7 - j] = pER->timeStamp[j];	/*  the greatest value is tagged */
    }
    pbufi8[0] &= ~BIT8;		/*    ...here */
    fwrite(pbufi8, sizeof(u8), 8, pf);	/*Writing time tagged */
  } else {			/* // we inverse the time i8 */
    pbufi8[0] = pER->timeStamp[0];
    pbufi8[0] &= ~BIT8;
    pbufi8[1] = pER->timeStamp[1];
    pbufi8[2] = pER->timeStamp[2];
    pbufi8[3] = pER->timeOfFlight;

    fwrite(pbufi8, sizeof(u8), 4, pf);	/*Writing time tagged */
  }

  i = 0;
  if (pEH->detectorIDBool == TRUE) {
    crystalIDs.w64 = pER->crystalIDs[0];
    for (size = 0; size < encodingIDSize + 1; size++) {
      pbufi16[i] = crystalIDs.w16[size];
      pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
      i++;
    }
  }

  if ((pEH->detectorIDBool == TRUE) && (pEH->coincidenceBool == TRUE)) {
    crystalIDs.w64 = pER->crystalIDs[pEH->numberOfNeighbours + 1];
    for (size = 0; size < encodingIDSize + 1; size++) {
      pbufi16[i] = crystalIDs.w16[size];
      pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
      i++;
    }
  }

  if (pEH->gantryAngularPosBool == TRUE) {
    pbufi16[i] = pER->gantryAngularPos;
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;
  }
  if (pEH->gantryAxialPosBool == TRUE) {
    pbufi16[i] = pER->gantryAxialPos;
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;
  }

  if (pEH->sourcePosBool == TRUE) {
    pbufi16[i] = pER->sourceAngularPos;
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;
    pbufi16[i] = pER->sourceAxialPos;
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;
  }
  fwrite(pbufi16, sizeof(u16), i, pf);	/*Writing the 2 bytes objects */

  j = 0;
  if (pEH->energyBool == TRUE) {

    pbufi8[j] = pER->energy[0];
    j++;
    if (pEH->neighbourBool == TRUE) {
      for (k = 1; k <= pEH->numberOfNeighbours; k++) {
	pbufi8[j] = pER->energy[k];
	j++;
      }
    }

    if (pEH->coincidenceBool == TRUE) {
      pbufi8[j] = pER->energy[pEH->numberOfNeighbours + 1];
      j++;
      if (pEH->neighbourBool == TRUE) {
	for (k = pEH->numberOfNeighbours + 2;
	     k <= (2 * pEH->numberOfNeighbours) + 1; k++) {
	  pbufi8[j] = pER->energy[k];
	  j++;
	}
      }
    }
  }

  if (pEH->fpgaNeighBool) {
    pbufi8[j] = pER->fpgaNeighInfo[0];
    j++;
    if (pEH->coincidenceBool == TRUE) {
      pbufi8[j] = pER->fpgaNeighInfo[1];
      j++;
    }
  }

  fwrite(pbufi8, sizeof(u8), j, pf);	/*Writing the 1 byte object */
}
