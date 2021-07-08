/*-------------------------------------------------------

List Mode Format 
                        
--  daqTimeCorrector.c  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2004 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of daqTimeCorrector

-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

#define SIZE 10000
#define TABLESIZE 2 * SIZE

#define AVERAGE_SLOPE_THRESHOLD -1e6
#define EVENT_TIME_THRESHOLD 1e11
#define MARGIN 2e6

static u16 sctNb;
static u16 *sctName = NULL;

static LIST *eventList = NULL;

static double *oldtime = NULL;
static u32 *deltaN = NULL;
static u32 *strongBitsTime = NULL;

static double *averageSlope = NULL;
static int *slopeNb = NULL;


static u32 count = 0;

static CURRENT_CONTENT *pcCC = NULL;
static ENCODING_HEADER *pEncoHC = NULL;
static EVENT_HEADER *pEHC = NULL;
static GATE_DIGI_HEADER *pGDHC = NULL;	/* local copies */
static COUNT_RATE_HEADER *pCRHC = NULL;
static COUNT_RATE_RECORD *pCRRC = NULL;
static FILE *pfileC = NULL;
static char *fileNameC;

void timeSorter(LIST *, u16);
double calculateAverageSlope(LIST *);
void copyDaqHeads(const ENCODING_HEADER *,
		  const EVENT_HEADER *,
		  const GATE_DIGI_HEADER *,
		  const COUNT_RATE_HEADER *,
		  const CURRENT_CONTENT *,
		  const COUNT_RATE_RECORD *, char *);

void setNbOfDaqSct(u16 inlineSctNb)
{
  sctNb = inlineSctNb;
}

void correctDaqTime(const ENCODING_HEADER * pEncoH,
		    const EVENT_HEADER * pEH,
		    const COUNT_RATE_HEADER * pCRH,
		    const GATE_DIGI_HEADER * pGDH,
		    const CURRENT_CONTENT * pcC,
		    const EVENT_RECORD * pER,
		    const COUNT_RATE_RECORD * pCRR, char *fileNameCCS)
{
  static u32 viewOnce = 0;
  static u8 doneonce = 0;
  static u8 check = 0;

  u16 sct, index;

  EVENT_RECORD *pERin;

  if (!doneonce) {
    copyDaqHeads(pEncoH, pEH, pGDH, pCRH, pcC, pCRR, fileNameCCS);

    if (!(sctName = (u16 *) malloc(sctNb * sizeof(u16))))
      printf
	  ("\n *** error : daqTimeCorrector.c : correctDaqTime : malloc\n");
    if (!(eventList = (LIST *) malloc(sctNb * sizeof(LIST))))
      printf("\n *** error : blockSorter.c : sortBlocks : malloc\n");
    if (!(oldtime = malloc(sctNb * sizeof(double))))
      printf
	  ("\n *** error : daqTimeCorrector.c : correctDaqTime : malloc\n");
    if (!(deltaN = malloc(sctNb * sizeof(u32))))
      printf
	  ("\n *** error : daqTimeCorrector.c : correctDaqTime : malloc\n");
    if (!(strongBitsTime = malloc(sctNb * sizeof(u32))))
      printf
	  ("\n *** error : daqTimeCorrector.c : correctDaqTime : malloc\n");
    if (!(averageSlope = malloc(sctNb * sizeof(double))))
      printf
	  ("\n *** error : daqTimeCorrector.c : correctDaqTime : malloc\n");
    if (!(slopeNb = malloc(sctNb * sizeof(int))))
      printf
	  ("\n *** error : daqTimeCorrector.c : correctDaqTime : malloc\n");

    for (sct = 0; sct < sctNb; sct++) {
      oldtime[sct] = 0;
      strongBitsTime[sct] = 0;
      averageSlope[sct] = 0;
      slopeNb[sct] = 0;
    }

    doneonce++;
  }

  pERin = newER(pEHC);

  copyER(pER, pERin, pEHC);

  sct = getRsectorID(pEncoHC, pER);

  if (!((viewOnce >> sct) & 1)) {
    viewOnce |= 1 << sct;
    check = 0;
    for (index = 0; index < 8 * sizeof(u32); index++)
      check += (viewOnce >> index) & 1;
    sctName[check - 1] = sct;
    dlist_init(&(eventList[check - 1]), (void *) freeER);
    if (check > sctNb) {
      printf
	  ("nb of sector in file is greater than the one introduced\nPlease re-run\n");
      exit(0);
    }
  }

  for (index = 0; index < check; index++)
    if (sct == sctName[index])
      break;

  if (dlist_ins_next
      (&(eventList[index]), dlist_tail(&(eventList[index])), pERin)) {
    printf
	("\n *** error : daqTimeCorrector.c : impossible to insert event in list\n");
    exit(0);
  }

  if ((eventList[index]).size == TABLESIZE)
    timeSorter(&(eventList[index]), index);
}


void finalCleanTable()
{
  u16 index;

  for (index = 0; index < sctNb; index++) {
    if ((eventList[index]).size > 0)
      timeSorter(&(eventList[index]), index);
    dlist_destroy(&(eventList[index]));
  }
  printf("%lu events were written in %s", count, fileNameC);
  free(eventList);


  free(sctName);
  free(deltaN);
  free(strongBitsTime);
  free(averageSlope);
  free(slopeNb);

  if (pEncoHC->scanContent.eventRecordBool == 1) {
    free(pEHC);
    if (pEncoHC->scanContent.gateDigiRecordBool == 1)
      if (pGDHC)
	free(pGDHC);
  }

  if (pcCC)
    free(pcCC);
  if (pEncoHC)
    free(pEncoHC);
  if (pGDHC)
    free(pGDHC);
  if (pCRHC)
    free(pCRHC);
  if (pCRRC)
    free(pCRRC);


  fclose(pfileC);
}

void copyDaqHeads(const ENCODING_HEADER * pEncoH,
		  const EVENT_HEADER * pEH,
		  const GATE_DIGI_HEADER * pGDH,
		  const COUNT_RATE_HEADER * pCRH,
		  const CURRENT_CONTENT * pcC,
		  const COUNT_RATE_RECORD * pCRR, char *fileNameCCS)
{
  pEncoHC = (ENCODING_HEADER *) malloc(sizeof(ENCODING_HEADER));
  pEHC = (EVENT_HEADER *) malloc(sizeof(EVENT_HEADER));
  pGDHC = (GATE_DIGI_HEADER *) malloc(sizeof(GATE_DIGI_HEADER));
  pCRHC = (COUNT_RATE_HEADER *) malloc(sizeof(COUNT_RATE_HEADER));
  pcCC = (CURRENT_CONTENT *) malloc(sizeof(CURRENT_CONTENT));
  pCRRC = (COUNT_RATE_RECORD *) malloc(sizeof(COUNT_RATE_RECORD));

  if ((pEncoHC == NULL) || (pEHC == NULL) || (pGDHC == NULL)
      || (pCRHC == NULL) || (pcCC == NULL) || (pCRRC == NULL))
    printf("\n *** error : daqTimeCorrector.c : copyDaqHeads : malloc\n");

  if (pEncoH)
    *pEncoHC = *pEncoH;		/* no pointer in this structure, it is safe */
  if (pEH)
    *pEHC = *pEH;		/* no pointer in this structure, it is safe */
  if (pCRH)
    *pCRHC = *pCRH;		/*  no pointer in this structure, it is safe */
  if (pGDH)
    *pGDHC = *pGDH;		/*  no pointer in this structure, it is safe */
  if (pcC)
    *pcCC = *pcC;		/*  no pointer in this structure, it is safe */

  fileNameC = fileNameCCS;
}

double calculateAverageSlope(LIST * table)
{
  double slope, upBound, downBound;
  u32 index, half;
  ELEMENT *element;


  half = table->size / 2;

  element = dlist_head(table);
  index = 0;
  upBound = 0;
  downBound = 0;

  while (1) {
    if (index < half)
      downBound +=
	  (double) (i64) u8ToU64(((EVENT_RECORD *) ((element)->data))->
				 timeStamp);
    else
      upBound +=
	  (double) (i64) u8ToU64(((EVENT_RECORD *) ((element)->data))->
				 timeStamp);

    index++;
    if (dlist_is_tail(element))
      break;
    element = dlist_next(element);
  }

  downBound /= half;
  upBound /= half;
  slope = (upBound - downBound) / half;

  return slope;
}

void timeSorter(LIST * list, u16 index)
{
  ELEMENT *element1 = NULL;
/*   ELEMENT *element2 = NULL; */
  void *data = NULL;

  double slope, currentSlope;
  u8 strongBitFlag;

  /*
     static FILE *plog = NULL;
     static FILE *plog2 = NULL;
     static u32 cnt = 0;

     if(!plog)
     plog = fopen("averageSlope.dat","w");
     if(!plog2)
     plog2 = fopen("diff.dat","w");
   */


  // First compute the slope average over the event sample
  slope = calculateAverageSlope(list);


  strongBitFlag = 0;
  if (averageSlope[index]) {
    //if((slope < averageSlope[index] + MARGIN) & (slope > averageSlope[index] - MARGIN))
    if (slope > averageSlope[index] - MARGIN) {
      averageSlope[index] *= slopeNb[index];
      averageSlope[index] += slope;
      slopeNb[index]++;
      averageSlope[index] /= slopeNb[index];
    }
/*       else */
/* 	{ */
/* 	  strongBitFlag = 1; */
/* 	  //printf("A possible discontinuity for sector %d\n",sctName[index]); */
/* 	} */
  } else
    averageSlope[index] = slope;

  //  cnt++;


  // Then find out in the fixed length sample the realy event where the dynamic jump has occured
  // if then, add the strong bit

  while (1) {
    element1 = dlist_head(list);

/*       if(strongBitFlag) */
/* 	{ */
/* 	  element2 = dlist_next(element1); */
/* 	  currentSlope = (double)(i64) u8ToU64(((EVENT_RECORD*)(element2->data))->timeStamp) */
/* 	    - ((double)(i64) u8ToU64(((EVENT_RECORD*)(element1->data))->timeStamp)); */
/* 	  if(currentSlope < - EVENT_TIME_THRESHOLD)  */
/* 	    { */
/* 	      printf("disconituity for sector %d\n",sctName[index]); */
/* 	      strongBitsTime[index]++; */
/* 	      strongBitFlag = 0; */
/* 	    } */
/* 	  if (dlist_is_tail(element2)) */
/* 	    strongBitFlag = 0; */
/* 	} */
/*       time38bitShifter(((EVENT_RECORD*)(element1->data))->timeStamp,strongBitsTime[index]); */

    if (oldtime[index]) {
      currentSlope = ((double) (i64)
		      (u8ToU64
		       (((EVENT_RECORD *) (element1->data))->timeStamp)) -
		      oldtime[index]) / deltaN[index];
      if ((currentSlope < averageSlope[index] + MARGIN) & (currentSlope >
							   averageSlope
							   [index] -
							   MARGIN)) {
	oldtime[index] =
	    (double) (i64) (u8ToU64
			    (((EVENT_RECORD *) (element1->data))->
			     timeStamp));
	deltaN[index] = 1;
	LMFbuilder(pEncoHC, pEHC, pCRHC, pGDHC, pcCC,
		   (EVENT_RECORD *) (element1->data), pCRRC, &pfileC,
		   fileNameC);
	count++;
      } else
	deltaN[index]++;
    } else {
      oldtime[index] =
	  (double) (i64) (u8ToU64
			  (((EVENT_RECORD *) (element1->data))->
			   timeStamp));
      deltaN[index] = 1;

      LMFbuilder(pEncoHC, pEHC, pCRHC, pGDHC, pcCC,
		 (EVENT_RECORD *) (element1->data), pCRRC, &pfileC,
		 fileNameC);
      count++;
    }

    if (dlist_is_tail(element1))
      break;

    if ((dlist_remove(&(eventList[index]), element1, (void **) &data)) == 0
	&& (eventList[index].destroy != NULL))
      eventList[index].destroy(data);
  }
  if ((dlist_remove(&(eventList[index]), element1, (void **) &data)) == 0
      && (eventList[index].destroy != NULL))
    eventList[index].destroy(data);
}
