/*-------------------------------------------------------

List Mode Format 
                        
--  fillCoinciRecordForGate.c --

Martin.Rey@epfl.ch
Crystal Clear Collaboration
Copyright (C) 2005 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of fillCoinciRecord.c 
(function wich was created by Luc Simon 
in the oneList_cleanKit.c file)

make an event coincidence record 
with 2 EVENT_RECORD of the dlist P2
               
multipleID is 0 except if fillCoinciRecord 
is called from multiple manager functions
-------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

void fillCoinciRecordForGate(const ENCODING_HEADER * pEncoHC,
			     const EVENT_HEADER * pEHC,
			     const GATE_DIGI_HEADER * pGDHC,
			     EVENT_RECORD * first, EVENT_RECORD * second,
			     int verboseLevel, EVENT_RECORD * pERC)
{
  static int i, firstGreaterBool;
  static double bufD;
  u32 timeOF;
  static u8 *puc;
  static u64 tFirst, tSecond;

#ifdef WRITE_TOF
  static FILE *pCoinciFile = NULL;
  u16 sct1, sct2;

  if (!pCoinciFile)
    pCoinciFile = fopen("coinci.dat", "w");
#endif

  /*   tFirst = getTimeOfThisEVENT((EVENT_RECORD *)(first->data)); */
  /*   tSecond = getTimeOfThisEVENT((EVENT_RECORD *)(second->data)); */
  tFirst = u8ToU64(first->timeStamp);
  tSecond = u8ToU64(second->timeStamp);

  firstGreaterBool = (tFirst > tSecond) ? TRUE : FALSE;


  /* time stored is the greater */
  bufD = (firstGreaterBool) ? (double) tFirst : (double) tSecond;

  if (verboseLevel)
    printf("\ntime = %f ps\t", bufD);

  /* time conversion in milliseconds */
  bufD = bufD * CONVERT_TIME_COINCI;

  if (verboseLevel)
    printf("converted = %f ms\n", bufD);

  puc = doubleToU8(bufD);

  if (verboseLevel)
    for (i = 0; i < 8; i++)
      printf("\t%d", puc[i]);

  for (i = 3; i < 8; i++) {	/* check if the bytes 3,4,5,6 and 7 are 0 */
    if (puc[i] != 0)
      printf
	  ("*** warning : cleanP2 : the time stamp is too big for 3 bytes");
  }

  for (i = 0; i < 3; i++)	/* store bytes 0,1 and 2 */
    pERC->timeStamp[i] = puc[2 - i];

  timeOF = (firstGreaterBool) ? tFirst - tSecond : tSecond - tFirst;
  //  timeOF = (u32)(((double) (timeOF) * getTimeStepFromCCH() / CONVERT_TOF) + 0.5);
  timeOF = (u32)(((double) (timeOF) * 1000 / CONVERT_TOF) + 0.5);

#ifdef WRITE_TOF
  if (firstGreaterBool) {
    sct1 = getRsectorID(pEncoHC, first);
    sct2 = getRsectorID(pEncoHC, second);
  } else {
    sct1 = getRsectorID(pEncoHC, second);
    sct2 = getRsectorID(pEncoHC, first);
  }

  fprintf(pCoinciFile, "%d\t%d\t%lu\n", sct1, sct2, timeOF);
#endif

  if (timeOF > 255)
    timeOF = 255;

  if (verboseLevel)
    printf("time of flight = %lu\t", timeOF);


  if (verboseLevel)
    if (timeOF >= 255) {
      printf
	  ("\n*** Warning : cleanP2.c : fillcoinci : timeOfFlight >= 255 after conversion !\n");
      printf("time of flight = %lu", timeOF);
    }

  pERC->timeOfFlight = (u8) timeOF;

  if (pEHC->gantryAxialPosBool == TRUE)
    pERC->gantryAxialPos = first->gantryAxialPos;

  if (pEHC->gantryAngularPosBool == TRUE)
    pERC->gantryAngularPos = first->gantryAngularPos;

  if (pEHC->sourcePosBool == TRUE) {
    pERC->sourceAngularPos = first->sourceAngularPos;
    pERC->sourceAxialPos = first->sourceAxialPos;
  }


  if (pEHC->detectorIDBool == TRUE) {
    pERC->crystalIDs[0] = first->crystalIDs[0];
    pERC->crystalIDs[1] = second->crystalIDs[0];
  }

  if (pEHC->energyBool == TRUE) {
    pERC->energy[0] = first->energy[0];
    pERC->energy[1] = second->energy[0];
  }

  if (pEHC->fpgaNeighBool) {
    pERC->fpgaNeighInfo[0] = first->fpgaNeighInfo[0];
    pERC->fpgaNeighInfo[1] = second->fpgaNeighInfo[0];
  }

  if (pEHC->gateDigiBool) {
    if (pGDHC->runIDBool)
      pERC->pGDR->runID = first->pGDR->runID;

    if (pGDHC->multipleIDBool)
      pERC->pGDR->multipleID = 0;

    if (pGDHC->eventIDBool) {
      pERC->pGDR->eventID[0] = first->pGDR->eventID[0];
      pERC->pGDR->eventID[1] = second->pGDR->eventID[0];
    }

    if (pGDHC->sourceIDBool) {
      pERC->pGDR->sourceID[0] = first->pGDR->sourceID[0];
      pERC->pGDR->sourceID[1] = second->pGDR->sourceID[0];
    }
    if (pGDHC->sourceXYZPosBool) {
      pERC->pGDR->sourcePos[0].X = first->pGDR->sourcePos[0].X;
      pERC->pGDR->sourcePos[1].X = second->pGDR->sourcePos[0].X;
      pERC->pGDR->sourcePos[0].Y = first->pGDR->sourcePos[0].Y;
      pERC->pGDR->sourcePos[1].Y = second->pGDR->sourcePos[0].Y;
      pERC->pGDR->sourcePos[0].Z = first->pGDR->sourcePos[0].Z;
      pERC->pGDR->sourcePos[1].Z = second->pGDR->sourcePos[0].Z;
    }
    if (pGDHC->globalXYZPosBool) {
      pERC->pGDR->globalPos[0].X = first->pGDR->globalPos[0].X;
      pERC->pGDR->globalPos[1].X = second->pGDR->globalPos[0].X;
      pERC->pGDR->globalPos[0].Y = first->pGDR->globalPos[0].Y;
      pERC->pGDR->globalPos[1].Y = second->pGDR->globalPos[0].Y;
      pERC->pGDR->globalPos[0].Z = first->pGDR->globalPos[0].Z;
      pERC->pGDR->globalPos[1].Z = second->pGDR->globalPos[0].Z;
    }

    if (pGDHC->comptonBool) {
      pERC->pGDR->numberCompton[0] = first->pGDR->numberCompton[0];
      pERC->pGDR->numberCompton[1] = second->pGDR->numberCompton[0];
    }




    if (pGDHC->comptonDetectorBool) {
      pERC->pGDR->numberDetectorCompton[0] =
	  first->pGDR->numberDetectorCompton[0];
      pERC->pGDR->numberDetectorCompton[1] =
	  second->pGDR->numberDetectorCompton[0];
    }
  }

  return;
}
