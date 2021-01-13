/*-------------------------------------------------------

           List Mode Format 
                        
     --  dumpCountRateHeader.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dumpCountRateHeader.c:
     This function called by dumpTheRecord()
     dispays on screen the containing of a
     count rate header structure.


-------------------------------------------------------*/
#include <stdio.h>
#include "lmf.h"

void dumpCountRateHeader(const COUNT_RATE_HEADER * pCRH)
{
  printf("\n");
  printf("--->  singleRateBool = %d\n", pCRH->singleRateBool);	/* singles countrate recorded if =1 */
  printf("--->  singleRatePart = %d\n", pCRH->singleRatePart);	/* Ring (1), sector(2), module(3) or total (0) */
  printf("--->  totalCoincidenceBool = %d\n", pCRH->totalCoincidenceBool);	/* total coincidence recorded if =1 */
  printf("--->  totalRandomBool = %d\n", pCRH->totalRandomBool);	/* total random rate recorded if =1 */
  printf("--->  angularSpeedBool = %d\n", pCRH->angularSpeedBool);	/* angular speed recorded if =1 */
  printf("--->  axialSpeedBool = %d\n", pCRH->axialSpeedBool);	/* axial speed recorded if =1 */
}
