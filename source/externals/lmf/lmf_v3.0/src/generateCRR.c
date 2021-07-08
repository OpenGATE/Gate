/*-------------------------------------------------------

           List Mode Format 
                        
     --  generateCRR.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of generateCRR.c:
     Example of filling of count rate record


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

static COUNT_RATE_RECORD *pCRR;


COUNT_RATE_RECORD(*generateCRR(ENCODING_HEADER * pEncoH,
			       COUNT_RATE_HEADER * pCRH))
{

  static int allocERdone = 0;
  i16 i = 0;
  static u16 ntRS = 0, ntM = 0, ntm = 0;

  if (allocERdone == 0) {
    allocERdone = 1;
    if ((pCRR =
	 (COUNT_RATE_RECORD *) malloc(sizeof(COUNT_RATE_RECORD))) == NULL)
      printf
	  ("\n*** ERROR : in generateCRR.c : impossible to do : malloc()\n");

    ntRS = pEncoH->scannerTopology.totalNumberOfRsectors;
    ntM = ntRS * pEncoH->scannerTopology.totalNumberOfModules;
    ntm = ntM * pEncoH->scannerTopology.totalNumberOfSubmodules;

    /*allocation for different levels of scanner */
    pCRR->pRsectorRate = malloc(ntRS * sizeof(u16));
    pCRR->pModuleRate = malloc(ntM * sizeof(u16));
    pCRR->pSubmoduleRate = malloc(ntm * sizeof(u16));

  }

  pCRR->timeStamp[0] = 4;	/*   time stamp on 31 bits but  maybe less... */
  pCRR->timeStamp[1] = 1;	/*   time stamp on 31 bits but  maybe less... */
  pCRR->timeStamp[2] = 5;	/*   time stamp on 31 bits but  maybe less... */

  pCRR->totalSingleRate[0] = 6;
  pCRR->totalSingleRate[1] = 4;


  for (i = 0; i < ntRS; i++)
    pCRR->pRsectorRate[i] = 2 * i + 3;
  for (i = 0; i < ntM; i++)
    pCRR->pModuleRate[i] = i + 2;
  for (i = 0; i < ntm; i++)
    pCRR->pSubmoduleRate[i] = i + 1;

  pCRR->coincidenceRate = 9;
  pCRR->randomRate = 3;
  pCRR->angularSpeed = 4;
  pCRR->axialSpeed = 5;
  return (pCRR);
}

void generateCRRDestructor()
{
  if (pCRR) {
    if (pCRR->pRsectorRate)
      free(pCRR->pRsectorRate);
    if (pCRR->pModuleRate)
      free(pCRR->pModuleRate);
    if (pCRR->pSubmoduleRate)
      free(pCRR->pSubmoduleRate);
    free(pCRR);
  }
}
