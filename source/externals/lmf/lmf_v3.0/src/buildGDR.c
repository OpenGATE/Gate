/*-------------------------------------------------------

           List Mode Format 
                        
     --  buildGDR.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of buildGDR.c:
     Called by LMFbuilder(). This function builds
     a gate digi record. It needs the encoding header
     thr event header and gate digi header structures 
     (pEncoH, pEH and pGDH)
     that affect the gate digi record size.


-------------------------------------------------------*/
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

u8 makeOneNumberOfCOmptonWithTwo(u8 a, u8 b)
{
  /*
     // if  a = 0000 1010
     // and b = 0000 0111
     // returns 1010 0111 (a then b)
   */
  if ((a > 15) || (b > 15)) {
    printf("*** WARNING : buildGDR.c : truncated number of compton\n");
    printf("one of %d or %d is bigger than 15\n", a, b);
    if (a > 15)
      a = 15;
    if (b > 15)
      b = 15;
  }

  a = a << 4;
  b = b | a;

  return (b);
}

void buildGDR(const GATE_DIGI_HEADER * pGDH,
	      const GATE_DIGI_RECORD * pGDR,
	      const EVENT_HEADER * pEH, FILE * pf)
{
#ifdef _64
  u32vsu8 fromU32toU16;
#else
  u32 pbufi32[10];		/* // 32 bits int */
  i16 l = 0;			/* Counter */
#endif
  u16 pbufi16[200];
  u8 pbufi8[50];
  i16 i = 0, j = 0, k = 0;	/*Counters */

  if (pGDH->runIDBool == TRUE) {
#ifdef _64
    fromU32toU16.w32 = htons(pGDR->runID);
    pbufi16[i] = fromU32toU16.w16[1];
    i++;
    pbufi16[i] = fromU32toU16.w16[0];
    i++;
#else
    pbufi32[l] = pGDR->runID;
    pbufi32[l] = htonl(pbufi32[l]);
    l++;
#endif
  }

  if (pGDH->eventIDBool == TRUE) {
#ifdef _64
    fromU32toU16.w32 = htons(pGDR->eventID[0]);
    pbufi16[i] = fromU32toU16.w16[1];
    i++;
    pbufi16[i] = fromU32toU16.w16[0];
    i++;
#else
    pbufi32[l] = pGDR->eventID[0];
    pbufi32[l] = htonl(pbufi32[l]);
    l++;
#endif
    if (pEH->coincidenceBool == TRUE) {
#ifdef _64
      fromU32toU16.w32 = htons(pGDR->eventID[1]);
      pbufi16[i] = fromU32toU16.w16[1];
      i++;
      pbufi16[i] = fromU32toU16.w16[0];
      i++;
#else
      pbufi32[l] = pGDR->eventID[1];
      pbufi32[l] = htonl(pbufi32[l]);
      l++;
#endif
    }
  }

  if ((pEH->coincidenceBool) && (pGDH->multipleIDBool)) {
#ifdef _64
    fromU32toU16.w32 = htons(pGDR->multipleID);
    pbufi16[i] = fromU32toU16.w16[1];
    i++;
    pbufi16[i] = fromU32toU16.w16[0];
    i++;
#else
    pbufi32[l] = pGDR->multipleID;
    pbufi32[l] = htonl(pbufi32[l]);	/* swap the i16 */
    l++;
#endif
  }
#ifndef _64
  fwrite(pbufi32, sizeof(u32), l, pf);	/*Write the 4 bytes objects */
#endif

  if (pGDH->sourceIDBool) {
    pbufi16[i] = pGDR->sourceID[0];
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;
    if (pEH->coincidenceBool == TRUE) {
      pbufi16[i] = pGDR->sourceID[1];
      pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
      i++;
    }
  }

  if (pGDH->sourceXYZPosBool == TRUE) {
    pbufi16[i] = pGDR->sourcePos[0].X;
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;
    pbufi16[i] = pGDR->sourcePos[0].Y;
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;
    pbufi16[i] = pGDR->sourcePos[0].Z;
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;
    if (pEH->coincidenceBool) {
      pbufi16[i] = pGDR->sourcePos[1].X;
      pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
      i++;
      pbufi16[i] = pGDR->sourcePos[1].Y;
      pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
      i++;
      pbufi16[i] = pGDR->sourcePos[1].Z;
      pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
      i++;
    }
  }

  if (pGDH->globalXYZPosBool == TRUE) {
    pbufi16[i] = pGDR->globalPos[0].X;
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;
    pbufi16[i] = pGDR->globalPos[0].Y;
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;
    pbufi16[i] = pGDR->globalPos[0].Z;
    pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
    i++;

    if (pEH->neighbourBool) {
      for (k = 1; k <= pEH->numberOfNeighbours; k++) {	/* first photon and neighbours */
	pbufi16[i] = pGDR->globalPos[k].X;
	pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
	i++;
	pbufi16[i] = pGDR->globalPos[k].Y;
	pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
	i++;
	pbufi16[i] = pGDR->globalPos[k].Z;
	pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
	i++;
      }
    }

    if (pEH->coincidenceBool) {

      pbufi16[i] = pGDR->globalPos[pEH->numberOfNeighbours + 1].X;
      pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
      i++;
      pbufi16[i] = pGDR->globalPos[pEH->numberOfNeighbours + 1].Y;
      pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
      i++;
      pbufi16[i] = pGDR->globalPos[pEH->numberOfNeighbours + 1].Z;
      pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
      i++;

      if (pEH->neighbourBool) {
	for (k = pEH->numberOfNeighbours + 2; k <= (2 * pEH->numberOfNeighbours) + 1; k++) {	/*second photon and neighbours */
	  pbufi16[i] = pGDR->globalPos[k].X;
	  pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
	  i++;
	  pbufi16[i] = pGDR->globalPos[k].Y;
	  pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
	  i++;
	  pbufi16[i] = pGDR->globalPos[k].Z;
	  pbufi16[i] = htons(pbufi16[i]);	/* swap the i16 */
	  i++;
	}
      }
    }
  }

  fwrite(pbufi16, sizeof(u16), i, pf);	/*Write the 2 bytes objects */

  j = 0;
  if (pGDH->comptonBool == TRUE) {
    if (pEH->coincidenceBool == TRUE) {	/* // number of compton is ok on a half byte */
      pbufi8[j] =
	  makeOneNumberOfCOmptonWithTwo(pGDR->numberCompton[0],
					pGDR->numberCompton[1]);
      j++;
    } else {
      pbufi8[j] = pGDR->numberCompton[0];
      j++;
    }
  }

  if (pGDH->comptonDetectorBool == TRUE) {
    if (pEH->coincidenceBool == TRUE) {	/* // number of compton is ok on a half byte */
      pbufi8[j] =
	  makeOneNumberOfCOmptonWithTwo(pGDR->numberDetectorCompton[0],
					pGDR->numberDetectorCompton[1]);
      j++;
    } else {

      pbufi8[j] = pGDR->numberDetectorCompton[0];
      j++;
    }
  }

  fwrite(pbufi8, sizeof(u8), j, pf);	/*Writing the 1 byte object */

  return;
}
