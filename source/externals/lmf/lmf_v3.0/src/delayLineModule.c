/*-------------------------------------------------------

           List Mode Format 
                        
     --  delayLineModule.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of delayLineModule

	Apply a delay on chosen rsector
	Default delay is 20ns * rsectorID
	Ex. : delay on rsector 10 = 200 ns
-------------------------------------------------------*/

#include <stdio.h>

#include "lmf.h"

static u16 numberOfRsectors;

static int doneOnce = FALSE;
static EVENT_RECORD *pERout;
static u16 *sctName = NULL;
static u64 *myDelayList = NULL;

static u64 myDelayBase = 0;
static u8 myDelayBaseIsSet = FALSE;

void setMyDelayBase(u64 value)
{
  myDelayBase = value;
  myDelayBaseIsSet = TRUE;
}

void initDelayList(u16 numberOfSectors)
{
  u16 j;

  numberOfRsectors = numberOfSectors;

  sctName = malloc(numberOfRsectors * sizeof(u16));
  myDelayList = malloc(sizeof(u64) * numberOfSectors);

  for (j = 0; j < numberOfSectors; j++)
    myDelayList[0] = 0;
}

EVENT_RECORD *delayLine(const ENCODING_HEADER * pEncoH,
			const EVENT_HEADER * pEH,
			const GATE_DIGI_HEADER * pGDH,
			const EVENT_RECORD * pER)
{
  u16 j;
  static u32 viewOnce = 0;
  static u8 check = 0;
  u16 index;

  static u64 timeStepFromCCH;
  u64 timeSingle;
  u16 rsectorID;
  u8 *bufCharTime = NULL;

  if (!doneOnce) {
    if (pEH->coincidenceBool != FALSE) {
      printf("*** ERROR : delayLineTime.c : Juelich dead time can ");
      printf("\n be applied only on singles file\n\n");
      exit(0);
    }
    if (pEH->detectorIDBool == FALSE) {
      printf("*** ERROR : delayLine.c : Juelich dead time can ");
      printf
	  ("\n be applied only on singles file containing detectorID\n\n");
      exit(0);
    }

    timeStepFromCCH = getTimeStepFromCCH();

    for (j = 0; j < numberOfRsectors; j++) {
      if (myDelayBaseIsSet) {
	myDelayList[j] = (u64) (j * myDelayBase);
	printf("myDelayList = %llu\t time step = %llu\n",
	       myDelayList[j], timeStepFromCCH);
	myDelayList[j] = (myDelayList[j] * 1000) / timeStepFromCCH;
	printf("myDelayList -> %llu\n", myDelayList[j]);
      } else
	myDelayList[j] = (u64) (j * DEFAULT_DELAY_BY_RSECYOR);
      /* 
         default is 20 ns * rsector ID

       */

    }

    pERout = newER(pEH);
    doneOnce = TRUE;
  }

  /* This new event is copied in a local structure */
  copyER(pER, pERout, pEH);	// cp pER to pERout but safe

  /* get time of this single in long long format */
  timeSingle = u8ToU64(pERout->timeStamp);
  /* get the rsector of this single */
  rsectorID = getRsectorID(pEncoH, pER);

  if (!((viewOnce >> rsectorID) & 1)) {
    viewOnce |= 1 << rsectorID;
    check = 0;
    for (index = 0; index < 8 * sizeof(u32); index++)
      check += (viewOnce >> index) & 1;
    sctName[check - 1] = rsectorID;
    if (check > numberOfRsectors) {
      printf
	  ("nb of sector in file is greater than the one introduced\nPlease re-run\n");
      exit(0);
    }
  }

  for (index = 0; index < check; index++)
    if (rsectorID == sctName[index])
      break;


  /* * * * * * *     TIME TREATMENT   * * * * */


  timeSingle += myDelayList[index];




  /* come back to 8 char array format for time */
  bufCharTime = u64ToU8(timeSingle);
  for (j = 0; j < 8; j++)
    pERout->timeStamp[j] = bufCharTime[j];


  return (pERout);
}


void destroyDelayLine(void)
{

  if (pERout)
    freeER(pERout);

  if (myDelayList)
    free(myDelayList);
  if (sctName)
    free(sctName);

  doneOnce = FALSE;

  myDelayBase = 0;
  myDelayBaseIsSet = FALSE;
}
