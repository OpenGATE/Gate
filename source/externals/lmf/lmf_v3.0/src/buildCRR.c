/*-------------------------------------------------------

           List Mode Format 
                        
     --  buildCRR.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of buildCRR.c:
     Called by LMFbuilder(). This function builds
     a count rate record. It needs the encoding header
     and count rate header structures (pEncoH and pCRH)
     that affect the count rate record size.

-------------------------------------------------------*/
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>

#include "lmf.h"

void buildCRR(const ENCODING_HEADER * pEncoH,
	      const COUNT_RATE_HEADER * pCRH,
	      const COUNT_RATE_RECORD * pCRR, FILE * pf)
{
  u16 pbufi16[500], tnR = 0, tnM = 0, tnSM = 0;
  u8 pbufi8[4];
  i16 i, j = 0;			/*Counters */
  tnR = pEncoH->scannerTopology.totalNumberOfRsectors;
  tnM = tnR * pEncoH->scannerTopology.totalNumberOfModules;
  tnSM = tnM * pEncoH->scannerTopology.totalNumberOfSubmodules;



  /* TAG, TIME STAMP */
  pbufi8[0] = pCRR->timeStamp[0];
  pbufi8[0] |= BIT8;
  pbufi8[0] &= ~(BIT7 + BIT6 + BIT5);
  pbufi8[1] = pCRR->timeStamp[1];
  pbufi8[2] = pCRR->timeStamp[2];
  pbufi8[3] = 0;
  fwrite(pbufi8, sizeof(u8), 4, pf);	/* WRITE THE TIME TAGGED */
  i = 0;
  if (pCRH->singleRateBool == TRUE) {
    pbufi16[i] = pCRR->totalSingleRate[0];
    pbufi16[i] = htons(pbufi16[i]);
    pbufi16[i + 1] = pCRR->totalSingleRate[1];
    pbufi16[i + 1] = htons(pbufi16[i + 1]);
    i = i + 2;

    if (pCRH->singleRatePart == 1) {	/* RSECTOR RATE */
      for (j = 0; j < tnR; j++) {
	pbufi16[i + j] = pCRR->pRsectorRate[j];
	pbufi16[i + j] = htons(pbufi16[i + j]);
      }
      i = i + tnR;
    }
    if (pCRH->singleRatePart == 2) {	/* MODULE RATE */
      for (j = 0; j < tnM; j++) {
	pbufi16[i + j] = pCRR->pModuleRate[j];
	pbufi16[i + j] = htons(pbufi16[i + j]);
      }
      i = i + tnM;
    }


    if (pCRH->singleRatePart == 3) {	/* SUBMODULE RATE */
      /*      //      printf("\n\nYes it s cool \n\n");    */
      for (j = 0; j < tnSM; j++) {
	pbufi16[i + j] = pCRR->pSubmoduleRate[j];
	pbufi16[i + j] = htons(pbufi16[i + j]);
      }
      i = i + tnSM;
    }
  }

  if (pCRH->totalCoincidenceBool == TRUE) {
    pbufi16[i] = pCRR->coincidenceRate;
    pbufi16[i] = htons(pbufi16[i]);
    i++;
  }
  if (pCRH->totalRandomBool == TRUE) {
    pbufi16[i] = pCRR->randomRate;
    pbufi16[i] = htons(pbufi16[i]);
    i++;
  }
  fwrite(pbufi16, sizeof(u16), i, pf);

  i = 0;
  if (pCRH->angularSpeedBool == TRUE) {
    pbufi8[i] = pCRR->angularSpeed;
    i++;
  }
  if (pCRH->axialSpeedBool == TRUE) {
    pbufi8[i] = pCRR->axialSpeed;
    i++;
  }
  fwrite(pbufi8, sizeof(u8), i, pf);


}
