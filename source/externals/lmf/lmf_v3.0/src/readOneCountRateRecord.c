/*-------------------------------------------------------

           List Mode Format 
                        
     --  readOneCountRateRecord.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of readOneCountRateRecord.c:
     
     Fills the LMF_ccs_countRateRecord structure with 
     a block of read bytes (block size must be check before)
     The block is pointed by *pBufCountRate
-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include "lmf.h"


static int allocCRR = 0, SSpart = 0;
static u16 ntR = 0, ntM = 0, ntm = 0;
static COUNT_RATE_RECORD *pCRR;



COUNT_RATE_RECORD(*readOneCountRateRecord(ENCODING_HEADER * pEncoH,
					  u8 * pBufCountRate,
					  u16 CRpattern))
{

  int i;
  u8 *puni8 = NULL;
  u16 *punch = NULL;





  if (allocCRR == 0) {		/* just one time this block */
    /* Calcul de SS */
    SSpart = (CRpattern >> 9);
    SSpart &= 3;

    ntR = pEncoH->scannerTopology.totalNumberOfRsectors;
    ntM = ntR * pEncoH->scannerTopology.totalNumberOfModules;
    ntm = ntM * pEncoH->scannerTopology.totalNumberOfSubmodules;

    if ((pCRR =
	 (COUNT_RATE_RECORD *) malloc(sizeof(COUNT_RATE_RECORD))) == NULL)
      printf
	  ("\n***ERROR : in readOneCountRate.c : impossible to do : malloc()\n");

    switch (SSpart) {
    case 1:
      if ((pCRR->pRsectorRate = malloc(ntR * sizeof(u16))) == NULL)
	printf
	    ("\n***ERROR : in readOneCountRate.c : impossible to do : malloc()\n");
      break;
    case 2:
      if ((pCRR->pModuleRate = malloc(ntM * sizeof(u16))) == NULL)
	printf
	    ("\n***ERROR : in readOneCountRate.c : impossible to do : malloc()\n");
      break;
    case 3:
      if ((pCRR->pSubmoduleRate = malloc(ntm * sizeof(u16))) == NULL)
	printf
	    ("\n***ERROR : in readOneCountRate.c : impossible to do : malloc()\n");
      break;
    }
    allocCRR = 1;
  }


  puni8 = (u8 *) pBufCountRate;
  punch = (u16 *) puni8;

  /*
     TAG + TIME + not used bits
   */

  for (i = 0; i < 3; i++) {
    pCRR->timeStamp[i] = *puni8;
    puni8++;
  }

  pCRR->timeStamp[3] = 0;

  if (*puni8 != 0)
    printf("\nWARNING RESERVED BYTES ARE NOT 0 in countrate record !!!\n");
  if ((pCRR->timeStamp[0] & (BIT8 + BIT7 + BIT6 + BIT5)) != BIT8)
    printf("\nWARNING WRONG TAG OF COUNT RATE !!!\n");

  puni8++;
  punch = (u16 *) puni8;

/*      Total single rate     */
  if ((CRpattern & BIT12) == BIT12) {
    *punch = ntohs(*punch);	/* byte order */
    pCRR->totalSingleRate[0] = *punch;
    punch++;
    *punch = ntohs(*punch);	/* byte order */
    pCRR->totalSingleRate[1] = *punch;
    punch++;


    switch (SSpart) {
    case 0:
      break;
    case 1:
      for (i = 0; i < ntR; i++) {
	*punch = ntohs(*punch);	/* byte order */
	pCRR->pRsectorRate[i] = *punch;
	punch++;
      }
      break;
    case 2:
      for (i = 0; i < ntM; i++) {
	*punch = ntohs(*punch);	/* byte order */
	pCRR->pModuleRate[i] = *punch;
	punch++;
      }
      break;
    case 3:
      for (i = 0; i < ntm; i++) {
	*punch = ntohs(*punch);	/* byte order */
	pCRR->pSubmoduleRate[i] = *punch;
	punch++;
      }
      break;
    }
  }
  /*-=-=-=-=-=-=-=-=-=-=*
    Total coincidence rate
   *-=-=-=-=-=-=-=-=-=-*/
  if ((CRpattern & BIT9) == BIT9) {
    *punch = ntohs(*punch);	/* byte order */
    pCRR->coincidenceRate = *punch;
    punch++;
  }
  /*-=-=-=-=-=-=-=-=-=-=*
    Total random rate
   *-=-=-=-=-=-=-=-=-=-*/
  if ((CRpattern & BIT8) == BIT8) {
    *punch = ntohs(*punch);	/* byte order */
    pCRR->randomRate = *punch;
    punch++;
  }

  puni8 = (u8 *) punch;

 /*-=-=-=-=-=-=-=-=-=-=*
       Rotation speed
   *-=-=-=-=-=-=-=-=-=-*/
  if ((CRpattern & BIT7) == BIT7) {
    pCRR->angularSpeed = *puni8;
    puni8++;
  }
 /*-=-=-=-=-=-=-=-=-=-=*
       Bed speed
   *-=-=-=-=-=-=-=-=-=-*/
  if ((CRpattern & BIT6) == BIT6) {
    pCRR->axialSpeed = *puni8;
    puni8++;
  }

  return (pCRR);


}



/* //  be very careful we need pCRH to destroy that */

void destroyCRRreader(COUNT_RATE_HEADER * pCRH)
{
  allocCRR = 0;
  SSpart = 0;
  ntR = 0;
  ntM = 0;
  ntm = 0;

  if (pCRH->singleRatePart == 1)
    free(pCRR->pRsectorRate);
  if (pCRH->singleRatePart == 2)
    free(pCRR->pModuleRate);
  if (pCRH->singleRatePart == 3)
    free(pCRR->pSubmoduleRate);
  free(pCRR);
}
