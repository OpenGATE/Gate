/*-------------------------------------------------------

           List Mode Format 
                        
     --  countRecords.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of countRecords.c:
     Called for each record read by LMFreader in mode
     countRecords
     This function returns at destruction, on sceen, and 
     eventually in a ASCII output file : 

     Number of records 
     Number of event records 
        Number of Singles 
	Number of Coincidences
     Number of count rate records
     Number of singles non chronologically sorted
 

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include "lmf.h"

static int headalreadyDone = FALSE;
static u64 recordNumber = 0, 
  countRateNumber = 0, 
  eventNumber = 0;
static u64 numberOfSingles = 0, 
  numberOfNonSortedSingles = 0,
  numberOfCoincidences = 0,
  nTrueCoincidences = 0,
  nScatterCoincidences = 0, 
  nRandomCoincidences = 0;
static u64 previousEventTime = 0;

static u16 layNb = 0;
static u32* singlesNb = NULL;
static u32* coinciNb = NULL;

static u8 coinciBool = 0;

void countRecords(const ENCODING_HEADER * pEncoH,
		  const EVENT_HEADER * pEH,
		  const COUNT_RATE_HEADER * pCRH,
		  const GATE_DIGI_HEADER * pGDH,
		  const CURRENT_CONTENT * pcC,
		  const EVENT_RECORD * pER, const COUNT_RATE_RECORD * pCRR)
{
  int test = 0;
  u16 l, l2;

  if (headalreadyDone == FALSE) {
    printf("\n");
    printf("*****************************\n");
    printf("* COUNT THE LMF BINARY FILE *\n");
    printf("*****************************\n");
    printf("\n");
    printf("******     HEAD      *******\n");
    printf("\n");

    //      dumpHead(pEncoH,pEH,pGDH,pCRH); 
    headalreadyDone = TRUE;
    printf("\n");
    printf("******    BODY      ********\n");
    printf("\n");

    coinciBool = pEH->coincidenceBool;
    layNb = pEncoH->scannerTopology.totalNumberOfLayers;

    if(coinciBool) {
      coinciNb = malloc(sizeof(u32)*layNb);
      for(l=0;l<layNb;l++)
	coinciNb[l] = 0;
    }
    else {
      singlesNb = malloc(sizeof(u32)*layNb);
    for(l=0;l<layNb;l++)
      singlesNb[l] = 0;
    }
  }
  recordNumber++;

  if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {
    l = getLayerID(pEncoH, pER);
    eventNumber++;
    if (coinciBool) {
      numberOfCoincidences++;
      l2 = getLayerID2(pEncoH, pER);
      if(l==l2)
	coinciNb[l]++;
      if (pEncoH->scanContent.gateDigiRecordBool) {
	if (pGDH->eventIDBool)
	  if (pER->pGDR->eventID[0] != pER->pGDR->eventID[1]) {
	    nRandomCoincidences++;
	    test++;
	  }
	if ((pGDH->comptonBool) && (!test))
	  if (pER->pGDR->numberCompton[0] || pER->pGDR->numberCompton[1]) {
	    nScatterCoincidences++;
	    test++;
	  }
	if(!test)
	  nTrueCoincidences++;
      }
    }
    else {
      numberOfSingles++;
      singlesNb[l]++;

      if (previousEventTime == 0)
	previousEventTime = getTimeOfThisEVENT(pER);
      else {
	if (previousEventTime > getTimeOfThisEVENT(pER))
	  numberOfNonSortedSingles++;
	else
	  previousEventTime = getTimeOfThisEVENT(pER);
      }
    }
  }
  if (pcC->typeOfCarrier == pEncoH->scanContent.countRateRecordTag)
    countRateNumber++;
}


void destroyCounter(void)
{
  u16 l;

  headalreadyDone = FALSE;
  printf("\n\n\n\n");
  printf("***************************************************\n");
  printf("All the file have been read :\n");
  printf("Number of records =\t%llu\n", recordNumber);
  recordNumber = 0;
  printf("\nNumber of event records =\t%llu\t\n", eventNumber);

  if (OF_is_Set())		/*  // write event number in an ascii file */
    fprintf(getAndSetOutputFileName(), "%llu\n", eventNumber);

  eventNumber = 0;
  if (coinciBool) {
    printf("\t\t\tCoincidences :\t%llu\n", numberOfCoincidences);
    for(l=0;l<layNb;l++)
      printf("\t\t\t\tlay %d:\t%lu\n", l, coinciNb[l]);
    free(coinciNb);
    printf("\t\t\t\ttrues : %llu\t %f %%\n", nTrueCoincidences,
	   100 * ((float) nTrueCoincidences) / ((float) numberOfCoincidences));
    nTrueCoincidences = 0;
    printf("\t\t\t\trandoms : %llu\t %f %%\n", nRandomCoincidences,
	   100 * ((float) nRandomCoincidences) /
	   ((float) numberOfCoincidences));
    nRandomCoincidences = 0;
    printf("\t\t\t\tscatters : %llu\t %f %%\n", nScatterCoincidences,
	   100 * ((float) (nScatterCoincidences)) /
	   ((float) numberOfCoincidences));
    nScatterCoincidences = 0;
    numberOfCoincidences = 0;
  }
  else {
    printf("\t\t\tSingles :\t%llu\n", numberOfSingles);
    numberOfSingles = 0;
    for(l=0;l<layNb;l++)
      printf("\t\t\t\tlay %d:\t%lu\n", l, singlesNb[l]);
    free(singlesNb);

    printf("Number of singles non chronologically sorted = %llu\n",
	   numberOfNonSortedSingles);
    numberOfNonSortedSingles = 0;
    previousEventTime = 0;
  }
  layNb = 0;

  printf("\nNumber of count rate records :\t%llu\n", countRateNumber);
  countRateNumber = 0;
  printf("***************************************************\n\n");

  return;
}
