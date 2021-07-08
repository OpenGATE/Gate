/*-------------------------------------------------------

           List Mode Format 
                        
     --  cutEventsModule.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of cutEventsModule.c

	crop some events (ex 100 firsts)
	(you have to set upLimit and downLimit)
	0 means no cut

-------------------------------------------------------*/

#include <stdio.h>

#include "lmf.h"

static int upLimit = 0;
static int downLimit = 0;
static int nRecord = 0;
EVENT_RECORD *cutEventsNumber(const ENCODING_HEADER * pEncoH,
			      const EVENT_HEADER * pEH,
			      const GATE_DIGI_HEADER * pGDH,
			      EVENT_RECORD * pER)
{
  int keepIT;			/*  = FALSE if we dont keep this event */
  keepIT = TRUE;
  nRecord++;

  if (upLimit) {
    if (nRecord > upLimit) {
      printf("CROP FINISHED\n");
      exit(0);
    } else {
      if (nRecord >= downLimit)
	keepIT = TRUE;
      else
	keepIT = FALSE;
    }
  } else			// keep til end of file if upLimit is NULL
  {
    if (nRecord > downLimit) {
      printf("CROP FINISHED\n");
      exit(0);
    }
  }




  if (keepIT)
    return (pER);
  else
    return (NULL);


}


void setRecordUpLimit(int limit)
{
  upLimit = limit;
  printf("Cut Events Module : upper limit set to %d\n", upLimit);
}
void setRecordDownLimit(int limit)
{
  downLimit = limit;
  printf("Cut Events Module : lower limit set to %d\n", downLimit);
}
