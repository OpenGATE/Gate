/*-------------------------------------------------------

List Mode Format 
                        
--  processRecordCarrier.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of processRecordCarrier.c:
LMFreader() reads one by one the records in a 
binary file, fill the record carriers 
and send them to this processRecordCarrier() function 
with an option (string) that specify wich treatment 
must be applied on the record carrier.
Options are

sortTime
treatAndCopy
analyseCoinci
outputAscii
dump
countRecords
sortCoincidence
locateIdInScanner

-------------------------------------------------------*/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


calculOfEventPosition resultOfCalculOfEventPosition;
static FILE *pf = NULL;
static ENCODING_HEADER *pEncoHC = NULL;

void processRecordCarrier(const ENCODING_HEADER * pEncoH,
			  EVENT_HEADER * pEH,
			  const GATE_DIGI_HEADER * pGDH,
			  const COUNT_RATE_HEADER * pCRH,
			  const CURRENT_CONTENT * pcC,
			  EVENT_RECORD * pER,
			  const COUNT_RATE_RECORD * pCRR,
			  const i8 * processingMode, FILE * pfread)
{
  if (strcmp("treatAndCopy", processingMode) == 0) {

    /* 
       This option is just copy and paste the LMF file
       but you can do a treatment in your event record with 
       this function.
       Be very careful not to touch the pER pointer 
       in this function, but just what it contains.
       You can edit your own treatEventRecord function
       to define a cut for exemple...

       The output file is a XXX_bis.ccs
    */
    treatEventRecord(pEncoH, pEH, pGDH, &pER);

    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("keepOnlyTrue", processingMode) == 0) {
    pER = keepOnlyTrue(pEncoH, pEH, pGDH, pER);

    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("juelichDeadTime", processingMode) == 0)
    juelichDT(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, bis_ccsFileName);

  if (strcmp("energyWindow", processingMode) == 0) {
    /*        
	      Apply an energy window (you have to set upLimit and downLimit, default 350-650 keV)
	      0,0 means no cut
    */
    pER = cutEnergy(pEncoH, pEH, pGDH, pER);

    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("delayLine", processingMode) == 0) {
    pER = delayLine(pEncoH, pEH, pGDH, pER);

    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("eventNumerSelection", processingMode) == 0) {
    /*        
	      crop some events (ex 100 firsts)(you have to set upLimit and downLimit default)
	      0 means no cut
    */
    pER = cutEventsNumber(pEncoH, pEH, pGDH, pER);

    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("sortTime", processingMode) == 0) {
    /*        
	      Sort chronologically a singles file
	      The output file is a XXX_bis.ccs
    */
    sortTime(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, pf, bis_ccsFileName);
  }

  if (strcmp("analyseCoinci", processingMode) == 0)
    coincidenceAnalyser(pEncoH, pEH, pGDH, pER);

  if (strcmp("outputAscii", processingMode) == 0)
    outputAscii(pEncoH, pEH, pGDH, pER);

  if (strcmp("outputRoot", processingMode) == 0)
    outputRoot(pEncoH, pEH, pGDH, pER);

  if (strcmp("dump", processingMode) == 0)
    dumpTheRecord(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR);

  if (strcmp("countRecords", processingMode) == 0)
    countRecords(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR);

  if (strcmp("sortCoincidence", processingMode) == 0) {
    /*     // Sort coincidences in a file of singles */
    if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {
      if (pEH->coincidenceBool == TRUE) {
	printf
	  ("\n*** WARNING : processRecordCarrier.c : cannot find coincidences in a coincidences .ccs file\n");
	printf("exit\n");
	exit(0);
      } else
	sortCoincidence(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR);
    }
  }

  if (strcmp("locateIdInScanner", processingMode) == 0) {
    /*     // display XYZ pos for an event record */
    if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {

      resultOfCalculOfEventPosition =
	locateEventInLaboratory(pEncoH, pER, 0);

      printf("ID first location searcher : %llu\n", pER->crystalIDs[0]);
      printf("x = %f\n",
	     resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	     radial);
      printf("y = %f\n",
	     resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	     tangential);
      printf("z = %f\n",
	     resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	     axial);

      if (pEH->coincidenceBool == 1) {
	resultOfCalculOfEventPosition =
	  locateEventInLaboratory(pEncoH, pER, 1);

	printf("ID second location searcher : %llu\n", pER->crystalIDs[1]);
	printf("x = %f\n",
	       resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	       radial);
	printf("y = %f\n",
	       resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	       tangential);
	printf("z = %f\n",
	       resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	       axial);
      }
    }
  }

  if (strcmp("setGantryPosition", processingMode) == 0) {
    /*        
	      Set the position (angle and translation pos) of the gantry of your event (default 0 0) 
    */
    if (pER)
      pER = setGantryPosition(pEncoH, pEH, pGDH, pER);

    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("shiftTime", processingMode) == 0) {
    if (pER) {
      shiftTime(pEncoH, pER);
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
    }
  }

  if (strcmp("sortBlocks", processingMode) == 0) {
    if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {
      if (pEH->coincidenceBool == TRUE) {
	printf
	  ("\n*** WARNING : processRecordCarrier.c : cannot do it in a coincidences .ccs file\n");
	printf("exit\n");
	exit(0);
      } else
	sortBlocks(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR,
		   bis_ccsFileName);
    }
  }

  if (strcmp("mergeLMFfiles", processingMode) == 0)
    if (pER)
      mergeLMFfiles(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf);

  if (strcmp("geometrySelector", processingMode) == 0) {
    geometrySelector(pEncoH, &pER);
    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("correctDaqTime", processingMode) == 0) {
    if (pEH->coincidenceBool == TRUE) {
      printf
	("\n*** WARNING : processRecordCarrier.c : cannot do it in a coincidences .ccs file\n");
      printf("exit\n");
      exit(0);
    } else
      correctDaqTime(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR,
		     bis_ccsFileName);
  }

  if (strcmp("5rings", processingMode) == 0) {
    make5rings(pEncoH, pEH, &pER, &pEncoHC);
    if (pER)
      LMFbuilder(pEncoHC, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("sortMultiCoincidences", processingMode) == 0)
    multipleCoincidencesSorter(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR);

  if (strcmp("timeWindow", processingMode) == 0) {
    cutTimeModule(pEH, &pER);
    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("keep2detectors", processingMode) == 0) {
    onlyKeep2Detectors(pEncoH, &pER);
    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("changeAxialPos", processingMode) == 0) {
    changeAxialPos(pER);
    LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
	       bis_ccsFileName);
  }

  if (strcmp("changeAngularPos", processingMode) == 0) {
    changeAngularPos(pER);
    LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
	       bis_ccsFileName);
  }

  if (strcmp("daqBuffer", processingMode) == 0) {
    daqBuffer(pEncoH, &pER);
    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("timeAnalyser", processingMode) == 0)
    if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag)
      timeAnalyser(pEncoH, pER);
  
  if (strcmp("followCountRates", processingMode) == 0)
    if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {
      followCountRates(pEncoH, &pER);
      if (pER)
	LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		   bis_ccsFileName);
    }

  if (strcmp("temporalResolution", processingMode) == 0)
    if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {
      temporalResolution(pEncoH, pER);
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
    }

  if (strcmp("energyResolution", processingMode) == 0)
    if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {
      energyResolution(pEncoH, &pER);
      if(pER)
	LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		   bis_ccsFileName);
    }

  if (strcmp("dispersePeaks", processingMode) == 0) {
    peakPositionDispersion(pEncoH, &pER);
    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("sigmoidCut", processingMode) == 0) {
    sigmoidCut(pEncoH, &pER);
    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("rejectRandomlyEvent", processingMode) == 0) {
    rejectFractionOfEvent(&pER);
    if (pER)
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
		 bis_ccsFileName);
  }

  if (strcmp("misidentification", processingMode) == 0) {
    DOImisID(pEncoH, pER);
    LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf,
	       bis_ccsFileName);
  }

  return;
}
