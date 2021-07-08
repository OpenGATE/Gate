/*-------------------------------------------------------

           List Mode Format 
                        
     --  coinciDeadTimeMgr.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of coinciDeadTimeMgr.c:
     Manage the dead time for a coincidence file. 
     This function is to simulate the processing of coincidence found
     with the Moisan model
     In most of commercial scanners, the coincidence are read 
     only once per period (ex. 256 ns)
     This dead time is applied on the total scanner.
     returns 1 if the event is seen 0 else
     WARNING : The coincidence must be chronologically sorted.
     

-------------------------------------------------------*/
#include <stdio.h>
#include "lmf.h"


static int doneOnce = FALSE;
static u64 currentTime = 0, nextAliveTime, deadTimeValue = 0;
static int deadTimeMode;
static u64 numberOfRejected = 0, numberOfAccepted = 0;


int deadTimeCoinciMgr(ELEMENT * first, ENCODING_HEADER * pEncoHC)
{


  int keepIT = TRUE;


  if (doneOnce == FALSE) {


    printf("Initialization of the coincidence dead time Manager:\n");


    deadTimeMode = setCSdtMode();
    deadTimeValue = setCSdtValue();



    printf("Mode set to %d\n ", deadTimeMode);
    printf("Value set to %llu\n", deadTimeValue);

    nextAliveTime = 0;
    doneOnce = TRUE;
  }

  currentTime = getTimeOfThisEVENT((EVENT_RECORD *) first->data);
  keepIT = TRUE;


  if (deadTimeMode != 0) {
    if (currentTime < nextAliveTime) {
      keepIT = FALSE;

      numberOfRejected++;
      if (deadTimeMode == 1)	/* // mode paralysable */
	nextAliveTime = currentTime + deadTimeValue;


    } else {
      numberOfAccepted++;
      keepIT = TRUE;
      nextAliveTime = currentTime + deadTimeValue;

    }
  }

  return (keepIT);


}


void deastroyDeadTimeCoinciMgr()
{
  doneOnce = FALSE;
  nextAliveTime = 0;



  printf("Destruction of the coincidence dead time Manager\n");
  printf("Accepted = %llu coincidences \n", numberOfAccepted);
  printf("Rejected = %llu coincidences \n", numberOfRejected);
  numberOfRejected = 0;
  numberOfAccepted = 0;

}
