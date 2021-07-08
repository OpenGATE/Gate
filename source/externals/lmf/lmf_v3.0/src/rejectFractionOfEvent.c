/*-------------------------------------------------------

List Mode Format 
                        
--  rejectFractionOfEvent.c  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2006 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of rejectFractionOfEvent


-------------------------------------------------------*/


#include <stdio.h>
#include "lmf.h"

static double fraction = 1;
static u32 tot = 0, rej = 0;

void setInitialParamForRejectFractionOfEvent(double inline_fraction)
{
  fraction = inline_fraction;

  if((fraction > 1) || (fraction < 0)) {
    printf("Fraction must be a number between 0 and 1\n\t-> EXIT\n");
    exit(EXIT_FAILURE);
  }

  return;
}

void rejectFractionOfEvent(EVENT_RECORD **ppER)
{
  double random;

  random = randd();

  if(random > fraction) {
    *ppER = NULL;
    rej++;
  }
  tot++;

  return;
}

void rejectFractionOfEventDestructor()
{
  double fract;

  fract = ((double)(rej)) / tot * 100;
  printf("%d rejected event over a total of %d\n", rej, tot);
  printf("\t-> %.0f%% (consign: fraction to keep %.0f%%)\n",
	 fract,fraction*100);

  rej = 0;
  tot = 0;
  fraction = 1;

  return;
}
