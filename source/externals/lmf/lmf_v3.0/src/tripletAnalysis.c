/*-------------------------------------------------------

           List Mode Format 
                        
     --  tripletAnalysis.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of tripletAnalysis.c

     analyse a multiple coincidence concerning 3 singles
     return a coded u16
     xxxx xxxx xxab cdef
     a: compton on ER1
     b: compton on ER2
     c: compton on ER3
     d: coincidence concerns E1
     e: coincidence concerns E2
     f: coincidence concerns E3

     if (!d && !e && !f)
     there is no true, or scattered coincidence in this triplet
     
-------------------------------------------------------*/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "lmf.h"
static int doneOnce = FALSE;
static EVENT_RECORD *pER1, *pER2, *pER3;
static FILE *pfAscii = NULL;	// to write the multiples pairs

u16 tripletAnalysis(ENCODING_HEADER * pEncoH,
		    EVENT_HEADER * pEH, EVENT_RECORD ** ppER)
{
  u16 cC = 0, nScatEvents = 0;
  int ev1, ev2, ev3;		// event IDs
  int co1, co2, co3;		// number of compton
  // t1 < t2 < t3
  if (!doneOnce) {
    pER1 = newER(pEH);
    pER2 = newER(pEH);
    pER3 = newER(pEH);
    // luc 
    pfAscii = fopen("triplet.dat", "w");


    doneOnce = TRUE;
  }


  cC = 0;
  nScatEvents = 0;

  copyER(ppER[0], pER1, pEH);
  copyER(ppER[1], pER2, pEH);
  copyER(ppER[2], pER3, pEH);

  co1 = pER1->pGDR->numberCompton[0];
  co2 = pER2->pGDR->numberCompton[0];
  co3 = pER3->pGDR->numberCompton[0];

  ev1 = pER1->pGDR->eventID[0];
  ev2 = pER2->pGDR->eventID[0];
  ev3 = pER3->pGDR->eventID[0];


  if (co1) {
    cC |= BIT4;
    nScatEvents++;
  }
  if (co2) {
    cC |= BIT5;
    nScatEvents++;
  }
  if (co3) {
    cC |= BIT6;
    nScatEvents++;
  }
  if ((ev1 == ev2) && (ev1 == ev3)) {
    if (nScatEvents > 1)	// triplet with no more than one compton scatt 
    {
      if (co1)
	cC |= BIT2 + BIT3;
      else if (co2)
	cC |= BIT1 + BIT3;
      else if (co3)
	cC |= BIT1 + BIT2;
    }
    // else nothing : real triplet with more than 2 scatt.

  } else if ((ev1 == ev2) && (ev2 != ev3)) {
    cC |= BIT1 + BIT2;
  } else if ((ev1 != ev2) && (ev2 == ev3)) {
    cC |= BIT3 + BIT2;
  } else if ((ev1 != ev2) && (ev2 != ev3)) {
    if (ev1 == ev3) {
      cC |= BIT3 + BIT1;
    }
    // else nothing : if (ev1!=ev3) 3 different decays...
  }
  //printf("cc = %d\n",cC);

  fprintf(pfAscii, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%llu\t%llu\t%llu\n",
	  nScatEvents,
	  cC,
	  diffRsectorER(pER1, pER2, pEncoH),
	  diffRsectorER(pER2, pER3, pEncoH),
	  diffRsectorER(pER1, pER3, pEncoH),
	  pER1->energy[0] * getEnergyStepFromCCH(),
	  pER2->energy[0] * getEnergyStepFromCCH(),
	  pER3->energy[0] * getEnergyStepFromCCH(),
	  getTimeOfThisEVENT(pER1),
	  getTimeOfThisEVENT(pER2), getTimeOfThisEVENT(pER3));


  return (cC);
}


void destroyTripletAnalysis()
{
  if (doneOnce) {
    freeER(pER1);
    freeER(pER2);
    freeER(pER3);
  }
  if (pfAscii)
    fclose(pfAscii);

}
