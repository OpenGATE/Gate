/*-------------------------------------------------------

           List Mode Format 
                        
     --  dumpCountRateRecord.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dumpCountRateRecord.c:
     This function called by dumpTheRecord()
     dispays on screen the containing of a
     count rate record structure.


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

void dumpCountRateRecord(const ENCODING_HEADER * pEncoH,
			 const COUNT_RATE_HEADER * pCRH,
			 const COUNT_RATE_RECORD * pCRR)
{
  static u8 time0;
  static i16 j = 0;		/*Counters */
  static int tnR, tnM, tnSM;

  tnR = pEncoH->scannerTopology.totalNumberOfRsectors;
  tnM = tnR * pEncoH->scannerTopology.totalNumberOfModules;
  tnSM = tnM * pEncoH->scannerTopology.totalNumberOfSubmodules;

  printf("TIME =\n");
  time0 = pCRR->timeStamp[0];
  time0 &= (BIT1 + BIT2 + BIT3 + BIT4);

  printf("%d\t", time0);
  printf("%d\t", pCRR->timeStamp[1]);
  printf("%d\n", pCRR->timeStamp[2]);


  if (pCRH->singleRateBool == TRUE) {
    printf("Total single rate = %d\t%d\n", pCRR->totalSingleRate[0],
	   pCRR->totalSingleRate[1]);

    if (pCRH->singleRatePart == 1) {	/* RSECTOR RATE */
      for (j = 0; j < tnR; j++) {
	printf("Rate in rsector %d = %d\n", j, pCRR->pRsectorRate[j]);
      }

    }
    if (pCRH->singleRatePart == 2) {	/* MODULE RATE */
      for (j = 0; j < tnM; j++) {
	printf("Rate in module %d = %d\n", j, pCRR->pModuleRate[j]);
      }
    }

    if (pCRH->singleRatePart == 3) {	/* SUBMODULE RATE */
      for (j = 0; j < tnSM; j++) {
	printf("Rate in submodule %d = %d\n", j, pCRR->pSubmoduleRate[j]);
      }
    }
  }

  if (pCRH->totalCoincidenceBool == TRUE) {
    printf("Coincidence rate = %d\n", pCRR->coincidenceRate);
  }
  if (pCRH->totalRandomBool == TRUE) {
    printf("Random rate = %d\n", pCRR->randomRate);
  }



  if (pCRH->angularSpeedBool == TRUE) {

    printf("Angular speed = %d\n", pCRR->angularSpeed);

  }

  if (pCRH->axialSpeedBool == TRUE) {
    printf("Axial speed = %d\n", pCRR->axialSpeed);
  }

}
