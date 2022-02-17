/*-------------------------------------------------------

List Mode Format 
                        
--  outputAsciiMgr.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of outputAsciiMgr.c:
This module allows to write in ASCII file some LMF infos
of Event record only 


-------------------------------------------------------*/

#include <stdio.h>
#include "lmf.h"

calculOfEventPosition resultOfCalculOfEventPosition;	/* JMV modif */


static u16 randomBool;		/*  // 1 if random coinci */
static int doneOnce = FALSE;
static int fileOK = TRUE;	/* // the file is a coincidence file ? */

static FILE *pfile = NULL;
u16 *pcrist;

static u64 timeMillis = 0;
int j;
GATE_DIGI_RECORD *pGDR;

void outputAscii(const ENCODING_HEADER * pEncoH,
		 const EVENT_HEADER * pEH,
		 const GATE_DIGI_HEADER * pGDH, const EVENT_RECORD * pER)
{
  int energyStep;

  if (doneOnce == FALSE) {
    if (pEncoH == NULL)
      printf
	  ("*** warning : outputAsciiMgr.c : no encoding pointer pEncoH defined, please check \n");
    else {
      if (pEncoH->scanContent.eventRecordBool == FALSE) {
	printf
	    ("*** error : outputAsciiMgr.c : not an event record file\n");
	//fileOK = FALSE;
	printf("<ENTER> to continue\n");
	getchar();

      }
      if (pEncoH->scanContent.gateDigiRecordBool == FALSE) {
	printf
	    ("*** warning : outputAsciiMgr.c : no gate digi in this file\n");
	//fileOK = FALSE; 
	printf("<ENTER> to continue\n");
	getchar();
      }
    }

    if (pGDH == NULL)
      printf
	  ("*** warning : outputAsciiMgr.c : no encoding pointer pGDH defined, please check \n");
    else {
      if (pGDH->comptonBool == FALSE) {
	printf
	    ("*** warning : outputAsciiMgr.c : no number of compton in this file\n");
	//fileOK = FALSE; 
	printf("<ENTER> to continue\n");
	getchar();
      }
      if (pGDH->eventIDBool == FALSE) {
	printf
	    ("*** warning : outputAsciiMgr.c : no event ID  in this file\n");
	//fileOK = FALSE; 
	printf("<ENTER> to continue\n");
	getchar();
      }
    }

    /*     //destroy the ascii file if it already exists */
    if (pEH->coincidenceBool) {
      pfile = fopen("ascii_coinci.txt", "w");
      fclose(pfile);
      pfile = fopen("ascii_coinci.txt", "a");
    } else {
      pfile = fopen("ascii_single.txt", "w");
      fclose(pfile);
      pfile = fopen("ascii_single.txt", "a");
    }
    energyStep = getEnergyStepFromCCH();
    doneOnce = TRUE;
  }

  if (fileOK) {
    // coincidences --------------------------------------------------------
    if (pEH->coincidenceBool) {
      /*      //time_in_millis / TOF / crystalIDS0  crystalID1 / EventID0 /  EventID1 */
      timeMillis = getTimeOfThisCOINCI(pER);
      /* MOdif to dump crystals XYZ positions in lab frame JMV/30-4-03 */
      fprintf(pfile, "%llu \t%.1f \t%llx \t%llx",
	      timeMillis,
	      getTimeOfFlightOfThisCOINCI(pER),
	      pER->crystalIDs[0], pER->crystalIDs[1]);
      if (pGDR != NULL)
	fprintf(pfile, "\t%ld \t%ld", pER->pGDR->eventID[0],
		pER->pGDR->eventID[1]);

      if (pGDR != NULL) {
	if (pER->pGDR->eventID[0] == pER->pGDR->eventID[1])
	  randomBool = 0;
	else
	  randomBool = 1;
      } else
	randomBool = 999;

      fprintf(pfile, "\t%d", randomBool);

      /*     // display XYZ pos for an event record, adapted from processRecordCarrier */
      resultOfCalculOfEventPosition =
	  locateEventInLaboratory(pEncoH, pER, 0);
      fprintf(pfile, "\t%5.2f\t%5.2f\t%5.2f",
	      resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	      radial,
	      resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	      tangential,
	      resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	      axial);

      if (pEH->energyBool == TRUE) {
	fprintf(pfile, "E %d %d keV", pER->energy[0],
		pER->energy[0] * energyStep);
	if (pEH->neighbourBool == TRUE) {
	  for (j = 1; j <= pEH->numberOfNeighbours; j++)
	    fprintf(pfile, "Engbr %d %d %d keV", j, pER->energy[j],
		    pER->energy[j] * energyStep);
	}
      }
      fprintf(pfile, "\n");
    }
    // singles ------------------------------------------------------------
    else {
      /*   // time_in_picos    crystalID    eventID */
      fprintf(pfile, "Ti: %llu \t%llx ", getTimeOfThisEVENT(pER),
	      pER->crystalIDs[0]);
      if (pGDR != NULL)
	fprintf(pfile, "\t%ld", pER->pGDR->eventID[0]);

      // Crystal adress
      pcrist = demakeid(pER->crystalIDs[0], pEncoH);
      fprintf(pfile, " \tAd: %2d  %2d  %2d  %2d  %2d",
	      pcrist[4], pcrist[3], pcrist[2], pcrist[0], pcrist[1]);
      free(pcrist);

      // display XYZ pos for an event record, adapted from processRecordCarrier
      resultOfCalculOfEventPosition =
	  locateEventInLaboratory(pEncoH, pER, 0);
      fprintf(pfile, " \tPn: \t%5.2f\t%5.2f\t%5.2f",
	      resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	      radial,
	      resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	      tangential,
	      resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	      axial);

      if (pEH->energyBool == TRUE) {
	fprintf(pfile, " Ey: \t%d  \t%d keV", pER->energy[0],
		pER->energy[0] * energyStep);
	if (pEH->neighbourBool == TRUE) {
	  for (j = 1; j <= pEH->numberOfNeighbours; j++)
	    fprintf(pfile, " \tEngbr % d %d %d keV", j, pER->energy[j],
		    pER->energy[j] * energyStep);
	}
      }
      fprintf(pfile, "\n");
    }
  }
}				/* outputAscii */

void destroyOutputAsciiMgr()
{
  fclose(pfile);
  doneOnce = FALSE;
  printf("<ENTER> to continue\n");
  getchar();
}
