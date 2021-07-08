/*-------------------------------------------------------

           List Mode Format 
                        
     --  juelichDT.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of juelichDT.c:


Here are the rules for three detected singles E1, E2,E3 in same FPGA
with resp. time stamp :
t1 < t2 < t3
FPGA is rsector
PM is module 

Requirements : 

SAME PM AND SAME FPGA RULES : 
if(t2 - t1) > 400 ns   OK (we keep E1 and E2)
if(t2-t1) < 250 ns --> We kill the 2 events
if (t2 -t1) < 400ns AND (t2-t1) > 250 ns
---> we keep E1 and E2 if PileUpAnalysis == True 
and kill E1 and E2 if PileUpAnalysis == False

SAME FPGA DIFFERENT PMS
if(t2 - t1) > 250 ns ---> OK 
else kill the 2


What is implemented now
if (FPGA alive)  (250 ns with previous)
   {
    if (PM alive) (400 ns with previous)
       write all events waiting in FPGA
     else 
       kill it and the one waiting in the PM
    
     increase nextAlive time for PM (400) and others PM of FPGA (250)

   }
else
   {
     kill all events waitin in FPGA
     increase nextAlive time for PM (400) and others PM of FPGA (250)
   }

// not implemented yet
/ if(t2 - t1) < 250 ns ---> 
// OK if (t3 - t1) > 600 ns
//    else we kill E1, E2 and E3


-------------------------------------------------------*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"
static FILE *pf;
static int doneOnce = FALSE;
static u64 erTime = 0, PMDT = 400000, FPGADT = 250000, threeSingleDT = 600000;	// picosecond
static int pileUpAnalysis = TRUE;



static int nSinglesTreated = 0;	// verbose counter

int verbose = 0;
static int tnm = 0, tnr = 0;	/* PET number of module, PET number of rsector */
static int nElement = 0, erSectorID = 0, erModuleID = 0;

struct bufferJuelich {
  EVENT_RECORD *pERbuf;
  u8 unfreeBool;
  u64 nextTimeFPGAalive;
  u64 nextTimePMalive;

};
static struct bufferJuelich **pBufJuelich;

void setPileUpAnalysis(u8 bool)
{
  pileUpAnalysis = bool;
}

void setHighDepthDT(u64 value)
{
  PMDT = value;
}

void setLowDepthDT(u64 value)
{
  FPGADT = value;
}

void setThreeSingleDT(u64 value)
{
  threeSingleDT = value;
}

void juelichDT(const ENCODING_HEADER * pEncoH,
	       const EVENT_HEADER * pEH,
	       const COUNT_RATE_HEADER * pCRH,
	       const GATE_DIGI_HEADER * pGDH,
	       const CURRENT_CONTENT * pcC,
	       const EVENT_RECORD * pER,
	       const COUNT_RATE_RECORD * pCRR, const i8 * nameOfFile)
{
  int i = 0, j = 0;
  static u64 diffP = 0;

  if (doneOnce == FALSE) {
    if (pEH->coincidenceBool != FALSE) {
      printf("*** ERROR : juelichDeadTime.c : Juelich dead time can ");
      printf("\n be applied only on singles file\n\n");
      exit(0);
    }
    if (pEH->detectorIDBool == FALSE) {
      printf("*** ERROR : juelichDeadTime.c : Juelich dead time can ");
      printf
	  ("\n be applied only on singles file containing detectorID\n\n");
      exit(0);
    }

    /* get scanner number of elements ( sector and modules only) */
    tnm = (int) pEncoH->scannerTopology.totalNumberOfModules;
    tnr = (int) pEncoH->scannerTopology.totalNumberOfRsectors;
    nElement = tnm * tnr;

    /* allocation of a 2D array of bufferJuelich */
    pBufJuelich =
	(struct bufferJuelich **) malloc(sizeof(struct bufferJuelich) *
					 tnr);
    if (pBufJuelich == NULL)
      printf("*** ERROR : juelichDT.c : malloc\n");
    for (i = 0; i < tnr; i++) {
      pBufJuelich[i] =
	  (struct bufferJuelich *) malloc(sizeof(struct bufferJuelich) *
					  tnm);
      if (pBufJuelich[i] == NULL)
	printf("*** ERROR : juelichDT.c : malloc\n");
    }

    for (i = 0; i < tnr; i++) {	/* initialisation of buffer juelich */
      for (j = 0; j < tnm; j++) {
	pBufJuelich[i][j].unfreeBool = 0;
	pBufJuelich[i][j].nextTimeFPGAalive = 0;
	pBufJuelich[i][j].nextTimePMalive = 0;
	pBufJuelich[i][j].pERbuf = NULL;
	pBufJuelich[i][j].pERbuf = newER(pEH);
	if (verbose)
	  printf("allocation of pbuf[%d][%d] ok \n", i, j);
      }
    }

    doneOnce = TRUE;

    printf("FPGADT = %llu PMDT = %llu\n", FPGADT, PMDT);
  }
  //if (getRsectorID(pEncoH,pER) ==0) verbose =9;
  //else verbose =0;

  if (verbose)
    printf("A new event has come = %d %llu %d %d (number,time,pm,fpga)\n",
	   nSinglesTreated++, getTimeOfThisEVENT(pER), getModuleID(pEncoH,
								   pER),
	   getRsectorID(pEncoH, pER));


  if (verbose)
    printf("\t\tdiff = %llu\n", getTimeOfThisEVENT(pER) - diffP);

  if (verbose)
    diffP = getTimeOfThisEVENT(pER);

  erSectorID = getRsectorID(pEncoH, pER);
  erModuleID = getModuleID(pEncoH, pER);
  erTime = getTimeOfThisEVENT(pER);




  if (sectorIsAlive(erSectorID, erTime)) {
    if (verbose)
      printf("sector %d is alive \n", erSectorID);

    if (moduleIsAlive(erSectorID, erModuleID, erTime)) {
      if (verbose)
	printf("module %d is alive \n", erModuleID);

      for (j = 0; j < tnm; j++) {
	if (pBufJuelich[erSectorID][j].unfreeBool == TRUE)	// if an event was waitin write it
	{
	  if (verbose)
	    printf
		("an event was waiting in pBufJuelich[%d][%d] ...write it\n",
		 erSectorID, j);
	  LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC,
		     pBufJuelich[erSectorID][j].pERbuf, pCRR, &pf,
		     bis_ccsFileName);
	} else {

	  if (verbose)
	    printf("no event was waiting in pBufJuelich[%d][%d] \n",
		   erSectorID, j);
	}
	pBufJuelich[erSectorID][j].unfreeBool = FALSE;
      }

      copyER(pER, pBufJuelich[erSectorID][erModuleID].pERbuf, pEH);	/* store new event in this location */
      pBufJuelich[erSectorID][erModuleID].unfreeBool = TRUE;	/* tag unfree bool */
      if (verbose)
	printf
	    ("copy this new event in  pBufJuelich[%d][%d] and unfreeBool is tag as true\n",
	     erSectorID, erModuleID);
      /* tune next alive time */

      for (j = 0; j < tnm; j++) {	/* increase next time alive of all PMs of the FPGA */
	pBufJuelich[erSectorID][j].nextTimeFPGAalive = erTime + FPGADT;
      }
      /* increase next time alive of this PM */
      if (!pileUpAnalysis)
	pBufJuelich[erSectorID][erModuleID].nextTimePMalive =
	    erTime + PMDT;
      else
	pBufJuelich[erSectorID][erModuleID].nextTimePMalive =
	    erTime + FPGADT;
      // PMDT=400000, FPGADT=250000

      if (verbose)
	printf
	    ("tune next alive time for sector %d to %llu and module %d to %llu\n",
	     erSectorID,
	     pBufJuelich[erSectorID][erModuleID].nextTimeFPGAalive,
	     erModuleID,
	     pBufJuelich[erSectorID][erModuleID].nextTimePMalive);
    } else {

      pBufJuelich[erSectorID][erModuleID].unfreeBool = FALSE;	// ignore event and kill the waitin one
      if (verbose)
	printf("module %d is dead tag pBuf[%d][%d],unfree to FALSE \n",
	       erModuleID, erSectorID, erModuleID);

      /* tune next alive time */

      for (j = 0; j < tnm; j++) {	/* increase next time alive of all PMs of the FPGA */
	pBufJuelich[erSectorID][j].nextTimeFPGAalive = erTime + FPGADT;	// = erTime ?;
      }
      /* increase next time alive of this PM */
      if (!pileUpAnalysis)
	pBufJuelich[erSectorID][erModuleID].nextTimePMalive = erTime + PMDT;	// = erTime ?;
      else
	pBufJuelich[erSectorID][erModuleID].nextTimePMalive = erTime + FPGADT;	// = erTime ?;

      // PMDT=400000, FPGADT=250000
      if (verbose)
	printf
	    ("tune next alive time for sector %d to %llu and module %d to %llu\n",
	     erSectorID,
	     pBufJuelich[erSectorID][erModuleID].nextTimeFPGAalive,
	     erModuleID,
	     pBufJuelich[erSectorID][erModuleID].nextTimePMalive);


    }

  } else {
    if (verbose)
      printf("sector %d is dead\n", erSectorID);

    for (j = 0; j < tnm; j++)
      pBufJuelich[erSectorID][j].unfreeBool = FALSE;	//ignore event and kill evry waitin events of the FPGA 

    if (verbose)
      printf("tag all module of sector %d to false\n", erSectorID);

    /* tune next alive time */
    for (j = 0; j < tnm; j++) {	/* increase next time alive of all PMs of the FPGA */
      pBufJuelich[erSectorID][j].nextTimeFPGAalive = erTime + FPGADT;	// = erTime ?;
    }
    /* increase next time alive of this PM */
    if (!pileUpAnalysis)
      pBufJuelich[erSectorID][erModuleID].nextTimePMalive = erTime + PMDT;	// = erTime ?;
    else
      pBufJuelich[erSectorID][erModuleID].nextTimePMalive = erTime + FPGADT;	// = erTime ?;

    if (verbose)
      printf
	  ("tune next alive time for sector %d to %llu and module %d to %llu\n",
	   erSectorID,
	   pBufJuelich[erSectorID][erModuleID].nextTimeFPGAalive,
	   erModuleID,
	   pBufJuelich[erSectorID][erModuleID].nextTimePMalive);


  }

  if (verbose)
    printf("\nend of story\n\n\n\n");
  if (verbose)
    getchar();
}


u8 sectorIsAlive(int erSector, u64 timeER)
{				/* check if one PM of this FPGA is dead */
  int iMod = 0;
  u8 alive = TRUE;
  for (iMod = 0; iMod < tnm; iMod++) {
    if (timeER < pBufJuelich[erSector][iMod].nextTimeFPGAalive) {
      alive = FALSE;
    }
  }
  return (alive);
}

u8 moduleIsAlive(int erSector, int erModule, u64 timeER)
{				/* check if this PM is dead */
  u8 alive = TRUE;
  if (timeER < pBufJuelich[erSector][erModule].nextTimePMalive)
    alive = FALSE;
  return (alive);
}






void destroyJuelichDeadTime(void)
{
  /* purge table */
  int i = 0, j = 0;
  int nIgnoredEvent = 0;


  for (i = 0; i < tnr; i++) {
    for (j = 0; j < tnm; j++) {
      if (pBufJuelich[i][j].unfreeBool == TRUE) {
	nIgnoredEvent++;
	//LMFbuilder(pEncoH,pEH,pCRH,pGDH,pcC,pBufJuelich[i][j].pERbuf,pCRR,pf,bis_ccsFileName);          
      }
    }
  }

  printf("Destroy juelich Dead time ignoring %d events\n", nIgnoredEvent);
  for (i = 0; i < tnr; i++) {
    for (j = 0; j < tnm; j++) {
      pBufJuelich[i][j].unfreeBool = 0;
      pBufJuelich[i][j].nextTimeFPGAalive = 0;
      pBufJuelich[i][j].nextTimePMalive = 0;
      freeER(pBufJuelich[i][j].pERbuf);
    }
  }
  for (i = 0; i < tnr; i++) {
    free(pBufJuelich[i]);
  }
  free(pBufJuelich);
  tnr = tnm = nElement = 0;
  doneOnce = FALSE;


}
