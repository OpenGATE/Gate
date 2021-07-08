/*-------------------------------------------------------

           List Mode Format 
                        
     --  coincidenceAnalyser.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of coincidenceAnalyser.c:
     This module just count the number of true, scatter, 
     random coincidences.
     A coincidence can only be one of these 3 possibilities
     First we search if it s random, then scatter and then true.
     These informations can only be found in simulation datas.

-------------------------------------------------------*/
#include <stdio.h>
#include "lmf.h"


static u64 nTotalCoincidences = 0;
static u64 nTrueCoincidences = 0;
static u64 nScatterCoincidences = 0, nScatterDetectorCoincidences = 0;
static u64 nRandomCoincidences = 0;
static u64 nMultiplesTrues = 0;
static u64 nMultiplesScatter = 0;
static u64 nMultiplesRandom = 0;
static u64 nMultiples = 0;

static float factorK = 0.17;	// default for a rat phantom in a P4
static float NEC = 0;
static int doneOnce = FALSE;
static int fileOK = TRUE;	/* // the file is a coincidence file ? */
static int fileContainsDetectorCompton = FALSE;

void coincidenceAnalyser(const ENCODING_HEADER * pEncoH,
			 const EVENT_HEADER * pEH,
			 const GATE_DIGI_HEADER * pGDH,
			 const EVENT_RECORD * pER)
{


  if (doneOnce == FALSE) {

    if (pEncoH->scanContent.eventRecordBool == FALSE) {
      printf
	  ("*** error : coincidenceAnalyser.c : not an event record file\n");
      fileOK = FALSE;
      printf("<ENTER> to continue\n");
      getchar();

    }

    if (pEH->coincidenceBool == FALSE) {
      printf
	  ("*** error : coincidenceAnalyser.c : not a coincidence file\n");
      fileOK = FALSE;
      printf("<ENTER> to continue\n");
      getchar();
    }
    if (pEncoH->scanContent.gateDigiRecordBool == FALSE) {
      printf
	  ("*** warning : coincidenceAnalyser.c : no gate digi in this file\n");
      fileOK = FALSE;
      printf("<ENTER> to continue\n");
      getchar();
    }
    if (pGDH->comptonBool == FALSE) {
      printf
	  ("*** warning : coincidenceAnalyser.c : no number of compton in this file\n");
      fileOK = FALSE;
      printf("<ENTER> to continue\n");
      getchar();
    }

    if (pGDH->comptonDetectorBool == TRUE)
      fileContainsDetectorCompton = TRUE;
    else
      fileContainsDetectorCompton = FALSE;


    if (pGDH->eventIDBool == FALSE) {
      printf
	  ("*** warning : coincidenceAnalyser.c : no event ID  in this file\n");
      fileOK = FALSE;
      printf("<ENTER> to continue\n");
      getchar();
    }
    doneOnce = TRUE;
  }
  if (fileOK) {

    nTotalCoincidences++;


    if (pER->pGDR->eventID[0] != pER->pGDR->eventID[1]) {
      nRandomCoincidences++;

    } else if (pER->pGDR->numberCompton[0] || pER->pGDR->numberCompton[1]) {
      nScatterCoincidences++;
    } else if ((fileContainsDetectorCompton) &&
	       (pER->pGDR->numberDetectorCompton[0]
		|| pER->pGDR->numberDetectorCompton[1])) {
      nScatterCoincidences++;
      nScatterDetectorCoincidences++;

    } else {
      nTrueCoincidences++;

    }


    if (pGDH->multipleIDBool) {
      if (pER->pGDR->multipleID) {
	nMultiples++;

	if (pER->pGDR->eventID[0] != pER->pGDR->eventID[1]) {
	  nMultiplesRandom++;

	} else if (pER->pGDR->numberCompton[0]
		   || pER->pGDR->numberCompton[1]) {
	  nMultiplesScatter++;
	} else {
	  nMultiplesTrues++;

	}
      }
    }
  }


}




void destroyCoincidenceAnalyser()
{


  doneOnce = FALSE;
  printf("______________________________________________\n");
  printf("______________________________________________\n");
  printf("Analyse of %llu coincidence result :\n", nTotalCoincidences);
  printf("trues = %llu\t %f %%\n", nTrueCoincidences,
	 100 * ((float) nTrueCoincidences) / ((float) nTotalCoincidences));
  printf
      ("scatter = %llu\t %f %% : phantom scat = %llu detector scat. = %llu\n",
       nScatterCoincidences,
       100 * ((float) (nScatterDetectorCoincidences)) /
       ((float) nTotalCoincidences),
       nScatterCoincidences - nScatterDetectorCoincidences,
       nScatterDetectorCoincidences);
  printf("random = %llu\t %f %%\n", nRandomCoincidences,
	 100 * ((float) nRandomCoincidences) /
	 ((float) nTotalCoincidences));


  NEC = ((float) nTrueCoincidences * (float) nTrueCoincidences);	// T*T
  NEC =
      NEC / ((float) nTrueCoincidences + (float) nScatterCoincidences +
	     2 * factorK * (float) nRandomCoincidences);

  //  printf("NEC (time = 1s and k = %f)  = %f --> \n\n", factorK,NEC);


  printf("multiples = %llu\t %f %%\n", nMultiples,
	 100 * ((float) nMultiples) / ((float) nTotalCoincidences));

  if (!nMultiples)
    nMultiples = 1;		/* no  division by 0 allowed */

  printf("\ttrues = %llu\t %f %% (%f %%/tof total)\n", nMultiplesTrues,
	 100 * ((float) nMultiplesTrues) / ((float) nMultiples),
	 100 * ((float) nMultiplesTrues) / ((float) nTotalCoincidences));
  printf("\tscatter = %llu\t %f %% (%f %%/tof total)\n", nMultiplesScatter,
	 100 * ((float) nMultiplesScatter) / ((float) nMultiples),
	 100 * ((float) nMultiplesScatter) / ((float) nTotalCoincidences));

  printf("\trandom = %llu\t %f %% (%f %%/tof total)\n", nMultiplesRandom,
	 100 * ((float) nMultiplesRandom) / ((float) nMultiples),
	 100 * ((float) nMultiplesRandom) / ((float) nTotalCoincidences));

  printf("______________________________________________\n");
  printf("______________________________________________\n");


  if (OF_is_Set()) {		/* // write in an ascii file */
    fprintf(getAndSetOutputFileName(), "%llu\n", nTrueCoincidences);
    fprintf(getAndSetOutputFileName(), "%llu\n", nScatterCoincidences);
    fprintf(getAndSetOutputFileName(), "%llu\n", nRandomCoincidences);
  }



  nTotalCoincidences = 0;
  nTrueCoincidences = 0;
  nScatterCoincidences = 0;
  nRandomCoincidences = 0;






}
