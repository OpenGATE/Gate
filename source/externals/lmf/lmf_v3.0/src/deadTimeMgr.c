/*-------------------------------------------------------

List Mode Format 
                        
--  deadTimeMgr.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of deadTimeMgr.c:
Apply a dead time on singles event record. 
The dead time can be paralysable or not
and the depth level of application can also be chosen
(levels are layer, crystal, submodule, module, rsector
and scanner)

-------------------------------------------------------*/
#include <stdio.h>
#include "lmf.h"


static int doneOnce = FALSE;
static u64 *pDeadTime = NULL, currentTime = 0, currentDeadTime;
static u16 *pcrist;
static int deadTimeMode;
static int verbose = 0;
static float ratioOfModeParalysable;
static int setDone = FALSE;
static int depthOfDeadTime = 4, dodDone = FALSE;
static int tnl = 0, tnc = 0, tns = 0, tnm = 0, tnr = 0;

/* 
   0 layer
   1 crystal
   2 submodule
   3 module
   4 rsector (default)
   5 scanner
*/

#ifndef DT_TIME_CONV
#define DT_TIME_CONV 1000000	/*  picos to microsecond */
#endif


void setDepthOfDeadTime(int depth)
{
  if (dodDone == FALSE) {
    depthOfDeadTime = depth;
    dodDone = TRUE;
    printf("Dead time applied at level %d:\n ", depthOfDeadTime);
  }

}





void setDeadTimeModeWithThis(int mode, u64 value, float ratio)
{
  if (!setDone) {
    deadTimeMode = mode;
    if (mode > 3) {
      printf("deadTimeMgr.c :dead time mode can just be 0 1 2 or 3\n");
      exit(0);
    }
    currentDeadTime = value;
    ratioOfModeParalysable = ratio;
    printf("mode %d dt %llu ratio %f\n", mode, value, ratio);
  }
  setDone = TRUE;

}

void setDeadTimeMode(void)
{

  if (!setDone) {
    /*
       mode = 0 no dead time
       mode = 1 paralysable
       mode = 2 non paralysable
       mode = 3 combined mode with a part of ratioOfModeParalysable (between 0 and 1)
     */


    printf("Set dead time mode : \n");
    printf("mode = 0 no dead time\n");
    printf("mode = 1 paralysable\n");
    printf("mode = 2 non paralysable\n");
    printf
	("mode = 3 combined mode with a part of ratioOfModeParalysable (between 0 and 1)\n");
    deadTimeMode = hardgeti16(0, 3);
    printf("deadTime mode = %d\n", deadTimeMode);

    printf("Set dead time (nano seconds) : \n");

    scanf("%llu", &currentDeadTime);
    currentDeadTime = currentDeadTime * 1000;

    printf("\ndeadTime = %llu\n", currentDeadTime);


    if (deadTimeMode == 3) {
      printf("Set ratio of paralysable (x10%%):\n");
      ratioOfModeParalysable = hardgeti16(0, 9);
      printf("combined mode : ratio of paralysable = %f %%\n",
	     ratioOfModeParalysable * 10);
    } else
      ratioOfModeParalysable = 0;

    setDone = TRUE;

  }

}

int getElementID(const ENCODING_HEADER * pEncoH, const EVENT_RECORD * pER)
{
  /* return a unique value for each element ID depending on depth of DT application */
  int value;
  pcrist = demakeid(pER->crystalIDs[0], pEncoH);
  if (depthOfDeadTime <= 5) {
    switch (depthOfDeadTime) {
    case 0:
      value =
	  tnm * (int) pcrist[4] + tns * (int) pcrist[3] +
	  tnc * (int) pcrist[2] + tnl * (int) pcrist[1] + (int) pcrist[0];
      break;
    case 1:
      value =
	  tnm * (int) pcrist[4] + tns * (int) pcrist[3] +
	  tnc * (int) pcrist[2] + (int) pcrist[1];
      break;
    case 2:
      value = tnm * (int) pcrist[4] + tns * (int) pcrist[3] + pcrist[2];
      break;
    case 3:
      value = tnm * (int) pcrist[4] + (int) pcrist[3];
      break;
    case 4:
      value = (int) pcrist[4];
      break;
    case 5:
      value = 0;
      break;
    }






  } else {
    printf
	("*** error: deadTimeMgr.c : depth must be < 6 ; here depth = %d\n",
	 depthOfDeadTime);
    exit(0);
  }
  free(pcrist);



  return (value);
}

/*
  returns 1 if the event is accepted 0 else
*/
int deadTimeMgr(const ENCODING_HEADER * pEncoH,
		const EVENT_HEADER * pEH,
		const GATE_DIGI_HEADER * pGDH, const EVENT_RECORD * pER)
{
  int i = 0;
  static int nElement = 0, elementID = 0;
  int keepIt = TRUE;


  if (deadTimeMode) {		/*  have we got to manage dead time : mode 0 = no dead time  */
    if (doneOnce == FALSE) {

      tnl = pEncoH->scannerTopology.totalNumberOfLayers;
      tnc = pEncoH->scannerTopology.totalNumberOfCrystals;
      tns = pEncoH->scannerTopology.totalNumberOfSubmodules;
      tnm = pEncoH->scannerTopology.totalNumberOfModules;
      tnr = pEncoH->scannerTopology.totalNumberOfRsectors;

      switch (depthOfDeadTime) {
      case 0:
	nElement = tnr * tnm * tns * tnc * tnl;
	break;
      case 1:
	nElement = tnr * tnm * tns * tnc;
	break;
      case 2:
	nElement = tnr * tnm * tns;
	break;
      case 3:
	nElement = tnr * tnm;
	break;
      case 4:
	nElement = tnr;
	break;
      case 5:
	nElement = 1;
	break;
      }


      pDeadTime = (u64 *) malloc(sizeof(u64) * nElement);
      if (pDeadTime == NULL)
	printf("*** ERROR : deadTimeMgr.c : malloc\n");

      for (i = 0; i < nElement; i++) {
	pDeadTime[i] = 0;	/*  set to 0 */
      }

      doneOnce = TRUE;
    }

    elementID = getElementID(pEncoH, pER);
    currentTime = getTimeOfThisEVENT(pER);


    if (verbose)
      printf("This element (%d) will be alive at %llu \n", elementID,
	     pDeadTime[elementID]);

    if (currentTime < pDeadTime[elementID]) {


      keepIt = FALSE;
      if (deadTimeMode == 1)	/*  mode paralysable */
	pDeadTime[elementID] = currentTime + currentDeadTime;
      if (deadTimeMode == 3) {	/*  combined mode */
	if (randd() < ratioOfModeParalysable) {
	  pDeadTime[elementID] = currentTime + currentDeadTime;
	}
      }

    } else {
      pDeadTime[elementID] = currentTime + currentDeadTime;


      keepIt = TRUE;
    }
  }

  if (verbose) {
    printf("Dead time manager : \n This event is ");
    if (keepIt)
      printf("taken in account \n");
    else
      printf("not taken in account \n");

    printf("Time of event %llu \n", currentTime);
    printf("Dead Time %llu \n", currentDeadTime);
    printf
	("This element will be alive again at  %llu us : element %d\n\n\n",
	 pDeadTime[elementID], elementID);


    getchar();
  }
  return (keepIt);
}



void destroyDeadTimeMgr()
{
  if (pDeadTime)
    free(pDeadTime);
  doneOnce = FALSE;
  setDone = FALSE;
  dodDone = FALSE;
  depthOfDeadTime = 4;

}
