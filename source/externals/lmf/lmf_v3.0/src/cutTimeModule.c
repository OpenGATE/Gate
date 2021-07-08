/*-------------------------------------------------------

           List Mode Format 
                        
     --  cutTimeModule.c  --                      

     Martin.Rey@epfl.ch

     Crystal Clear Collaboration
     Copyright (C) 2004 LPHE/EPFL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of cutTimeModule.c


-------------------------------------------------------*/

#include <stdio.h>

#include "lmf.h"

static u8 startTimeBool = 0;
static u32 startTime = 0;
static u32 timeDuration = 0;

void setCutTimeModuleParams(u8 inStartTimeBool, u32 inStartTime, u32 inTimeDuration)
{
  startTimeBool = inStartTimeBool;
  startTime = inStartTime;
  timeDuration = inTimeDuration;

  return;
}

void cutTimeModule(const EVENT_HEADER * pEH,
		   EVENT_RECORD ** ppER)
{
  static u8 doneOnce;
  u32 actualTime;

  if(pEH->coincidenceBool)
    actualTime = getTimeOfThisCOINCI(*ppER);
  else
    actualTime = getTimeOfThisEVENT(*ppER) / 1000 / 1000 / 1000;

  if(!doneOnce) {
    if(!startTimeBool) {
      startTime = actualTime;
      timeDuration += actualTime;
    }
    else 
      timeDuration += startTime;

    printf("actual Time = %lu\n",actualTime);
    printf("start Time = %lu\n",startTime);
    printf("stop Time = %lu\n",timeDuration);
    doneOnce++;
  }

  if(actualTime < startTime)
    *ppER = NULL;
  else
    if (actualTime > timeDuration) {
      printf("CUT TIME FINISHED\n");
      printf("First Time = %lu LastTime = %lu\n",startTime, actualTime);
      exit(0);
    }

  return;
}
