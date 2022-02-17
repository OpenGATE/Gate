/*-------------------------------------------------------

List Mode Format 
                        
--  oneList_cleanKit.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of oneList_cleanKit.c:
When the list used to sort coincidences
contain too old elements we clean them here.

-------------------------------------------------------*/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "lmf.h"

static int verboseLevel, coincidence_window;
static u64 stack_cut_time;
static int deadTimeForCoincidenceSortingMode = 0;
static u64 deadTimeForCoincidenceSortingValue = 0;
static int saveAutoCoinciBool = 0;	/* if = 0 don 't store the Auto coincidence (same rsector) */
/*                                    if = 1 keep them */

static int rsectorNeighbourOrder = 0;	/* 
					   if auto coinci bool is 0 
					   we can also remove coincidences occurs
					   in neighbour rsector:
					   0 accept coinci in neigh. rsector
					   1 removes if |rsector1 - rsector2| <= 1
					   2 removes if |rsector1 - rsector2| <= 2
					 */

static int saveMultipleBool = 0;	/* if = 0 don 't store the multiples */
/*                                  if = 1 keep them but associate 2 by 2 in chronological order */
static u32 multipleID = 0;	// stored in LMF coinci file if M && C
static CURRENT_CONTENT *pcCC = NULL;
static ENCODING_HEADER *pEncoHC;	/* to build coinci file head */
static EVENT_HEADER *pEHC;	/* to build coinci file head */
static GATE_DIGI_HEADER *pGDHC;	/*     to build coinci file head */
static COUNT_RATE_HEADER *pCRHC;	/*   to build coinci file head */
static EVENT_RECORD *pERC;	/* to build coinci file head */
static int nN;			/* number of neighs */
/* static const i8 nameOfCoincidenceFile[] = COINCIDENCE_FILE; */
static int headDone = FALSE, okForWriteInLMF = TRUE;

static u64 numberOfCoinci = 0, numberOfMultiple = 0;
static u64 numberOfSingleton = 0, numberOfTooLateSingleton = 0;
static u64 numberOfRejectedSinglesMultiples =
    0, numberOfAcceptedSinglesMultiples = 0;
static u64 arrayMultiples[10];
static u64 nMultiplesWritten = 0;

static u64 nMulRandomCoincidences = 0;
static u64 nMulScatterCoincidences = 0;
static u64 nMulTrueCoincidences = 0;



static u64 nAutos = 0, nNonAutos = 0;




void initCleanKit()
{

  int i;

  for (i = 0; i < 10; i++)
    arrayMultiples[i] = 0;	// counts number of triplets, number of quadruplets,...

  verboseLevel = setCSverboseLevel();

  coincidence_window = setCScoincidencewindow();
  stack_cut_time = setCSstackcuttime();
  saveMultipleBool = setCSsaveMultipleBool();

  saveAutoCoinciBool = setCSsaveAutoCoinciBool();
  rsectorNeighbourOrder = setCSrsectorNeighOrder();


  deadTimeForCoincidenceSortingMode = setCSdtMode();
  if (deadTimeForCoincidenceSortingMode) {
    deadTimeForCoincidenceSortingValue = setCSdtValue();

  }



  numberOfCoinci = 0;
  numberOfMultiple = 0;


}


/*******************************************************

                    InitCoinciFile
 
              build the head of coinci file
             


********************************************************/
void initCoinciFile(const ENCODING_HEADER * pEncoH,
		    const EVENT_HEADER * pEH,
		    const GATE_DIGI_HEADER * pGDH,
		    const COUNT_RATE_HEADER * pCRH,
		    const CURRENT_CONTENT * pcC)
{


  if (headDone == FALSE) {
    headDone = TRUE;

    pEncoHC = (ENCODING_HEADER *) malloc(sizeof(ENCODING_HEADER));
    pEHC = (EVENT_HEADER *) malloc(sizeof(EVENT_HEADER));
    pGDHC = (GATE_DIGI_HEADER *) malloc(sizeof(GATE_DIGI_HEADER));
    pCRHC = (COUNT_RATE_HEADER *) malloc(sizeof(COUNT_RATE_HEADER));
    pcCC = (CURRENT_CONTENT *) malloc(sizeof(CURRENT_CONTENT));


    if ((pEncoHC == NULL) || (pEHC == NULL) || (pCRHC == NULL))
      printf("\n *** error : cleanKit.c : initCoiciFile : malloc\n");

    if (pEncoH)
      *pEncoHC = *pEncoH;	/* no pointer in this structure, it is safe */
    if (pEH)
      *pEHC = *pEH;		/* no pointer in this structure, it is safe */
    if (pCRH)
      *pCRHC = *pCRH;		/*  no pointer in this structure, it is safe */
    if (pGDH)
      *pGDHC = *pGDH;		/*  no pointer in this structure, it is safe */
    if (pcC)
      *pcCC = *pcC;		/*  no pointer in this structure, it is safe */

    nN = pEHC->numberOfNeighbours;
    pEHC->coincidenceBool = 1;	/*  yes */
    if (pGDH)
      pGDHC->multipleIDBool = 1;
    pERC = newER(pEHC);



  }
  /* */
}

/* canal 22_1 fi 0 10000 1000000000 1 0 1 1 0 */
/*******************************************************

                    DestroyCoinciFile
 
                  Destroy the coinci file
             

********************************************************/
void destroyCoinciFile(void)
{

  int i;
  u64 totalSM = 0;
  destroyTripletAnalysis();
  printf("number of multiples = %llu\n\n", numberOfMultiple);
  numberOfMultiple = 0;
  printf("number of singles rejected as multiples = %llu\n",
	 numberOfRejectedSinglesMultiples);
  printf("number of singles accepted as multiples = %llu\n",
	 numberOfAcceptedSinglesMultiples);
  printf
      ("number of singletons = %llu (%llu too late and %llu normal singleton)\n",
       numberOfSingleton, numberOfTooLateSingleton,
       numberOfSingleton - numberOfTooLateSingleton);
  for (i = 3; i < 9; i++)
    printf("number of multiples concerning %d singles = %llu\n", i, arrayMultiples[i]);	// counts number of triplets, number of quadruplets,...
  printf("number of multiples concerning more than 9 singles = %llu\n", arrayMultiples[9]);	// counts number of triplets, numbe

  for (i = 0; i <= 9; i++)
    totalSM = totalSM + i * arrayMultiples[i];
  printf("Singles concerned by multiples = %llu \n", totalSM);
  printf("number of true coincidence written = %llu\n", numberOfCoinci);
  printf("number of multiple coincidences accepted as trues = %llu\n",
	 nMultiplesWritten);
  printf("Total of written coincidences = %llu\n",
	 nMultiplesWritten + numberOfCoinci);
  printf("number of auto coincidence rejected = %llu \n", nAutos);

  printf("accepted multiple coincidence analyser \n");
  printf("\tTrues = %llu\n", nMulTrueCoincidences);
  printf("\tScatter = %llu\n", nMulScatterCoincidences);
  printf("\tRandom = %llu\n", nMulRandomCoincidences);



  numberOfRejectedSinglesMultiples = 0;
  numberOfAcceptedSinglesMultiples = 0;
  numberOfCoinci = 0;
  nMultiplesWritten = 0;
  nAutos = 0;
  nNonAutos = 0;
  FreeLMFCBuilderCarrier(pEncoHC, pEHC, pGDHC, pcCC, pERC);

  closeOutputCoincidenceFile();

}




/*******************************************************

                        cleanListP1

             remove all the elements en of P1
            for which t(pERin) - t(en) > Delta    
           if they re not associated with an element
              for which this condition is false
         The non multiple coincidences are sent to LMF 
               coincdence events file.
           Multiples and singles are killed.


********************************************************/
int cleanListP1(LIST * plistP1, EVENT_RECORD * pERin)
{
  static u64 tEn = 0;
  int i, count = 1, countN = 0;	/* to count how much events are associated */
  static ELEMENT *searcher, *searcher2, *destroyer;
  void *data;
  tEn = getTimeOfThisEVENT(pERin);
  if (verboseLevel)
    printf("\n*  * * * * * * * * * clean P1 * * * * * * * * *\n");


  if (plistP1->size != 0) {
    searcher = (plistP1)->tail;

    if (searcher->prev != NULL)	/* check if there is more than 1 Elt */
      while (tEn >=
	     (stack_cut_time +
	      getTimeOfThisEVENT((EVENT_RECORD *) (searcher->prev->
						   data)))) {
	searcher = searcher->prev;

	if (searcher->prev == NULL) {
	  if (verboseLevel)
	    printf("All the list needs to be clean !!!\n");
	  break;		/* head */
	}
    } else {
      if (verboseLevel)
	printf("\njust an element to clean...done\n");
      if ((dlist_remove(plistP1, searcher, (void **) &data)) == 0 &&
	  (plistP1->destroy != NULL)) {
	plistP1->destroy(data);
      }

      return (0);
    }
    /*       searcher is now the highest element that is "too old" and is not NULL */


    if (verboseLevel) {
      if (searcher)
	printf("latest too old element = %llu\n",
	       getTimeOfThisEVENT((EVENT_RECORD *) (searcher->data)));
      else
	printf("1 searcher  null strange ?");
    }
    searcher2 = searcher;
    countN = 0;


    if (searcher2->prev != NULL) {
      while (searcher2->prev->CWN == 1) {
	countN++;
	searcher2 = searcher2->next;
	if (searcher2 == NULL) {
	  if (verboseLevel)
	    printf("searcher 2 null\n");
	  break;		/* tail */
	}

      }
    } else {
      if (verboseLevel)
	printf("\nAll the list needs to be clean !!!\n");
    }

    /*       searcher2 is now the highest element that is "too old" */
    /*       but not associated with another one that is not "too old" */
    /*       All elements under searcher2 (included) must be processed */
    /*       if searcher2 is NULL (tail) we don't process  */
    /*       exept in High-activity case */
    if (verboseLevel) {
      if (searcher2)
	printf
	    ("latest too old element but not associated with a not too old one = %llu\n",
	     getTimeOfThisEVENT((EVENT_RECORD *) (searcher2->data)));
      else
	printf("searcher  null -> tail reached -> no clean possible");
    }
    if (countN > HIGH_ACTIVITY_CUT_NUMBER) {	/* high activity case */
      if (verboseLevel) {
	printf("Warning : High activity in Coincidence Sorter \n");
	printf("\ttoo much associated elements\n");
	printf("\tkill not too old elements\n");
      }
      if (verboseLevel > 2)
	getchar();


      if (searcher2) {		/*  not at tail */
	destroyer = searcher2->prev;
	for (i = 0; i < countN; i++) {
	  if ((dlist_remove(plistP1, destroyer, (void **) &data)) == 0 &&
	      (plistP1->destroy != NULL)) {
	    plistP1->destroy(data);
	  }
	  destroyer = searcher2->prev;
	}
      } else {			/* tail */

	destroyer = searcher;
	searcher = searcher->prev;
	if (searcher) {
	  while (destroyer != NULL) {	/* destroy all till tail */
	    if ((dlist_remove(plistP1, destroyer, (void **) &data)) == 0 &&
		(plistP1->destroy != NULL)) {
	      plistP1->destroy(data);
	    }
	    destroyer = searcher->next;

	  }
	  dlist_tail(plistP1)->CWN = 0;

	  if (tEn >=
	      stack_cut_time +
	      getTimeOfThisEVENT((EVENT_RECORD *) dlist_tail(plistP1)->
				 data))
	    cleanListP1(plistP1, pERin);
	  else
	    return (0);
	} else {		/* all the elements of this list must be destroyed */

	  for (i = 0; i < plistP1->size; i++) {
	    if ((dlist_remove
		 (plistP1, dlist_head(plistP1), (void **) &data)) == 0
		&& (plistP1->destroy != NULL)) {
	      plistP1->destroy(data);
	    }
	  }
	  return (0);
	}
      }
    }

    if (!searcher2) {		/*  all the oldest elements are associated with a "not too old" */
      if (verboseLevel) {
	printf(" ...cannot remove elements\n");
	if (verboseLevel > 2)
	  getchar();
      }
      return (0);
    }

    while (searcher2 != NULL) {	/* while we are not in the tail continue to process */

      while (searcher2->next != NULL) {
	if (searcher2->CWN == 1) {

	  searcher2 = searcher2->next;
	  count++;
	} else
	  break;
      }

      if (verboseLevel) {
	if (searcher2)
	  printf("Processing until %llu\n",
		 getTimeOfThisEVENT((EVENT_RECORD *) (searcher2->data)));
	printf("count = %d\n", count);
      }


      if (count > 2) {		/* multiple ->> destroy these ones */

	if (verboseLevel) {
	  printf("Find a multiple (%d)  \n", count);
	}

	if (count > 8)
	  arrayMultiples[9]++;
	else
	  arrayMultiples[count] = arrayMultiples[count] + 1;

	numberOfMultiple++;


	destroyer = searcher2;
	searcher2 = searcher2->next;

	if (!saveMultipleBool) {	/* if don t save the multiples */

	  if (verboseLevel) {
	    printf("destroy this  multiple (%d singles)  \n", count);
	  }


	  if (searcher2 != NULL) {	/* not the tail */



	    for (i = 0; i < count; i++) {

	      numberOfRejectedSinglesMultiples++;

	      if ((dlist_remove(plistP1, destroyer, (void **) &data)) == 0
		  && (plistP1->destroy != NULL)) {
		plistP1->destroy(data);
	      }
	      destroyer = searcher2->prev;
	    }
	  } else {		/*  destroy tail coun times */


	    for (i = 0; i < count; i++) {

	      numberOfRejectedSinglesMultiples++;

	      if ((dlist_remove
		   (plistP1, dlist_tail(plistP1), (void **) &data)) == 0
		  && (plistP1->destroy != NULL)) {
		plistP1->destroy(data);
	      }
	    }
	  }
	} else if (saveMultipleBool) {
	  multipleCoincidenceMgr(plistP1, destroyer, count);
	  multipleCombinatCoincidenceMgr(plistP1, destroyer, count);


	}



      } else if (count == 2) {	/*   well it s a coincidence ! */
	destroyer = searcher2;
	if (verboseLevel)
	  printf
	      ("Find a coincidence between 2 singles  => %llu and %llu \n",
	       getTimeOfThisEVENT((EVENT_RECORD *) (destroyer->data)),
	       getTimeOfThisEVENT((EVENT_RECORD *) (destroyer->prev->
						    data)));

	searcher2 = searcher2->next;


	/*    Now, destroyer and destroyer->prev are in coincidence ! */


	/*              If we don t want to keep the auto coincidences (same rsector) */
	/*              we must check if it is one or not. */
	okForWriteInLMF = TRUE;


	if (!saveAutoCoinciBool) {
	  okForWriteInLMF =
	      checkForAutoCoinci(destroyer, destroyer->prev, pEncoHC);

	}



	/*              Eventually applied a dead time on the coincidence  */
	/*              but with the single time accuracy */
	if (okForWriteInLMF) {
	  if (deadTimeForCoincidenceSortingMode) {
	    okForWriteInLMF = deadTimeCoinciMgr(destroyer, pEncoHC);
	    if (verboseLevel)
	      printf("okforwrite (DT coinci mgr) = %d\n\n\n",
		     okForWriteInLMF);

	  }
	}


	if (okForWriteInLMF) {
	  fillCoinciRecord(pEncoHC, pEHC, pGDHC,
			   (EVENT_RECORD *) (destroyer->data),
			   (EVENT_RECORD *) (destroyer->prev->data), 0, nN,
			   verboseLevel, pERC);
	  outputCoincidence(pEncoHC, pEHC, pGDHC, pcCC, pERC,
			    getCoincidenceOutputMode());

	  numberOfCoinci++;	/* count coinci     */
	  if (verboseLevel)
	    printf("... coincidence sent to LMF\n ");
	}


	/* remove these 2 of P1 */
	if ((dlist_remove(plistP1, destroyer->prev, (void **) &data)) == 0
	    && (plistP1->destroy != NULL)) {
	  plistP1->destroy(data);
	}
	if ((dlist_remove(plistP1, destroyer, (void **) &data)) == 0 &&
	    (plistP1->destroy != NULL)) {
	  plistP1->destroy(data);
	}




	if (verboseLevel)
	  printf("\n...and destroyed in P1 \n");
      } else if ((count < 2) && (count > 0)) {	/*  singles */

	if (verboseLevel)
	  printf("\nFind and destroying a single \n");

	numberOfSingleton++;
	destroyer = searcher2;
	searcher2 = searcher2->next;
	if ((dlist_remove(plistP1, destroyer, (void **) &data)) == 0 &&
	    (plistP1->destroy != NULL)) {
	  plistP1->destroy(data);
	}
      }
      count = 1;

    }

  } else if (verboseLevel)
    printf("\nP1 empty...\n");



  if (verboseLevel) {
    printf("\n* * * * * * * * * *clean P1 done... * * * * * * * \n");
    if (verboseLevel > 5)
      getchar();
  }
  return (0);
}

/*******************************************************

                        finalCleanListP1

              called just before P1 destruction
               remove all the elements en of P1
         if they re not associated with an element
              for which this condition is false
         The non multiple coincidences are sent to LMF 
               coincdence events file
            If we find a single , print a warning


********************************************************/
u64 finalCleanListP1(LIST * plistP1)
{
  int i, count = 1;		/* to count how much events are associated */
  static ELEMENT *searcher, *destroyer, *saver;
  void *data;

  /*   verboseLevel = 9; */
  printf
      ("\n\n****************    FINAL CLEAN    **********************  \n\n");
  if (verboseLevel) {
    print_list(plistP1);
  }

  if (plistP1->size)
    searcher = (plistP1)->head;
  else
    return (numberOfCoinci);

  while (searcher != NULL) {	/*  while we are not in the tail continue to process */
    searcher = (plistP1)->head;

    while (searcher->next != NULL) {
      if (searcher->CWN == 1) {
	searcher = searcher->next;
	count++;
      } else
	break;
    }


    if (count > 2) {		/* multiple ->> destroy these ones */
      if (verboseLevel)
	printf("multiple\n");

      if (count > 8)
	arrayMultiples[9]++;
      else
	arrayMultiples[count] = arrayMultiples[count] + 1;

      numberOfMultiple++;

      searcher = searcher->next;

      if (!saveMultipleBool) {	/* if we don t keep the multiples */

	for (i = 0; i < count; i++) {	/* destroy head count times */

	  numberOfRejectedSinglesMultiples++;
	  if ((dlist_remove(plistP1, dlist_head(plistP1), (void **) &data))
	      == 0 && (plistP1->destroy != NULL)) {
	    plistP1->destroy(data);
	  }
	}
      } else if (saveMultipleBool) {
	saver = dlist_head(plistP1);
	for (i = 0; i < (count - 1); i++)
	  saver = saver->next;
	//multipleCoincidenceMgr(plistP1,saver,count);
	multipleCombinatCoincidenceMgr(plistP1, saver, count);
      }

    } else if (count == 2) {	/*  well it s a coincidence ! */
      if (verboseLevel)
	printf("coincidence\n");

      destroyer = searcher;
      searcher = searcher->next;

      /*      Now, destroyer and destroyer->prev are in coincidence ! */
      /*      If we don t want to keep the auto coincidences (same rsector) */
      /*      we must check if it is one or not. */
      okForWriteInLMF = TRUE;

      if (!saveAutoCoinciBool)
	okForWriteInLMF =
	    checkForAutoCoinci(destroyer, destroyer->prev, pEncoHC);

      /*      Eventually applied a dead time on the coincidence  */
      /*      but with the single time accuracy */
      if (okForWriteInLMF) {
	if (deadTimeForCoincidenceSortingMode) {
	  okForWriteInLMF = deadTimeCoinciMgr(destroyer, pEncoHC);
	}
      }


      if (okForWriteInLMF) {
	fillCoinciRecord(pEncoHC, pEHC, pGDHC,
			 (EVENT_RECORD *) (destroyer->data),
			 (EVENT_RECORD *) (destroyer->prev->data), 0, nN,
			 verboseLevel, pERC);
	outputCoincidence(pEncoHC, pEHC, pGDHC, pcCC, pERC,
			  getCoincidenceOutputMode());
	numberOfCoinci++;	/*  count coinci */
	if (verboseLevel)
	  printf("coincidence sent to LMF ");
      }

      /*      remove these 2 of P1 */
      if ((dlist_remove(plistP1, destroyer->prev, (void **) &data)) == 0 &&
	  (plistP1->destroy != NULL)) {
	plistP1->destroy(data);
      }
      if ((dlist_remove(plistP1, destroyer, (void **) &data)) == 0 &&
	  (plistP1->destroy != NULL)) {
	plistP1->destroy(data);
      }

      if (verboseLevel)
	printf("\n...and destroyed in P1 \n");
    } else if ((count < 2) && (count > 0)) {	/* singles */


      numberOfSingleton++;

      destroyer = searcher;
      searcher = searcher->next;

      if ((dlist_remove(plistP1, destroyer, (void **) &data)) == 0 &&
	  (plistP1->destroy != NULL)) {
	plistP1->destroy(data);
      }
    }

    count = 1;
  }

  if (verboseLevel)
    printf("\nfinal cleanP1 done...");
  printf("coincidence found :  %llu\n", numberOfCoinci);
  return (numberOfCoinci);
}

/*******************************************************

                     multipleCoincidenceMgr

              Called if you do not want to simply remove
              multiple coincidences (saveMultipleBool == 1)
              destroyer is the oldest element to destroy. there is count-1
              elements to manage on his "head"
              Ex:  count = 4
              
              ...
              elt to manage
              elt to manage
              elt to manage
              destroyer to manage
              ...
               
              The coincidence are associated 2 by 2, searching the most probable...
              For example if you have a 5 event multiple, it search the longest
              distance between two sectors. Write them if they are 
              distant enough (diff > rsector neigh order)
              and continue until there is nothin else to process.


********************************************************/
void multipleCoincidenceMgr(LIST * plistP1, ELEMENT * destroyer, int count)
{
  u16 *pcrist, rsID1, rsID2;
  void *data;
  ELEMENT *searcher, *first, *second, *mostProb1, *mostProb2;
  int nCouple = 0, localCount, i, j;
  int stillToProcess = 0;	// number of singles still to process
  int maxDiffR = 0, diffR = 0;
  u64 diffT = 0;

  if (verboseLevel)
    printf
	("\nMultiple manager for this multiple................................. \n");
  multipleID++;
  localCount = count;
  nCouple = count / 2;
  searcher = destroyer->next;
  stillToProcess = localCount;


  if (verboseLevel) {
    printf("\nMultiple manager starts to check %d singles:\n", localCount);
    second = destroyer;
    for (i = 0; i < count; i++) {
      pcrist =
	  demakeid(((EVENT_RECORD *) (second->data))->crystalIDs[0],
		   pEncoHC);
      printf("rsector %d\n", pcrist[4]);
      free(pcrist);
      second = second->prev;
    }
  }


  while (stillToProcess > 0) {	/* while all singles are not destroyed */
    if (stillToProcess > 1) {
      if (searcher)
	first = searcher->prev;
      else
	first = dlist_tail(plistP1);
      second = first;

      maxDiffR = 0;
      diffR = 0;
      mostProb1 = NULL;
      mostProb2 = NULL;
      for (j = 0; j < stillToProcess - 1; j++) {	/*look for maximum probability on rsector ID */
	second = first;
	for (i = 0; i < stillToProcess - j - 1; i++) {
	  second = second->prev;
	  diffR = diffRsector(first, second, pEncoHC);
	  //              diffT = diffTime(first,second,pEncoHC);
	  if ((diffR >= maxDiffR) && (diffT < coincidence_window))
	    if (diffR >= maxDiffR) {
	      maxDiffR = diffR;
	      mostProb1 = first;
	      mostProb2 = second;
	    }
	}
	first = first->prev;
      }				/* after this "for loop", mostProb1 and mostProb2 are supposed to be
				   the most probable coinci.  */

      if (verboseLevel > 1) {
	printf("most1 = %p 2= %p stil =%d\n", mostProb1, mostProb2,
	       stillToProcess);
	pcrist =
	    demakeid(((EVENT_RECORD *) (mostProb1->data))->crystalIDs[0],
		     pEncoHC);
	rsID1 = pcrist[4];
	free(pcrist);
	pcrist =
	    demakeid(((EVENT_RECORD *) (mostProb2->data))->crystalIDs[0],
		     pEncoHC);
	rsID2 = pcrist[4];
	free(pcrist);
	printf("max diff found in this multiple = %d : (%d and %d)\n",
	       maxDiffR, rsID1, rsID2);
      }

      if (maxDiffR > rsectorNeighbourOrder) {	/* Are the 2 concerned restor distant enough ? */
	/* yes they are : write them after an eventual dead time check */

	okForWriteInLMF = TRUE;
	if (deadTimeForCoincidenceSortingMode)
	  okForWriteInLMF = deadTimeCoinciMgr(mostProb1, pEncoHC);
	if (!okForWriteInLMF) {
	  if (verboseLevel > 1)
	    printf
		("this multiple is not accepted by coinci. dead time manager\n");

	  numberOfRejectedSinglesMultiples += 2;

	}

	if (okForWriteInLMF) {
	  if (verboseLevel > 1) {
	    printf
		("\nmultiple manager writes the most probable coincidence:\n");
	    printf("diff between rsector = %d (minimum neigh = %d)\n",
		   maxDiffR, rsectorNeighbourOrder);
	  }
	  fillCoinciRecord(pEncoHC, pEHC, pGDHC,
			   (EVENT_RECORD *) (mostProb1->data),
			   (EVENT_RECORD *) (mostProb2->data), multipleID,
			   nN, verboseLevel, pERC);
	  outputCoincidence(pEncoHC, pEHC, pGDHC, pcCC, pERC,
			    getCoincidenceOutputMode());
	  nMultiplesWritten++;
	  numberOfAcceptedSinglesMultiples += 2;

	  if (pEHC->gateDigiBool == TRUE) {
	    if (pERC->pGDR->eventID[0] != pERC->pGDR->eventID[1])
	      nMulRandomCoincidences++;
	    else if (pERC->pGDR->numberCompton[0]
		     || pERC->pGDR->numberCompton[1])
	      nMulScatterCoincidences++;
	    else
	      nMulTrueCoincidences++;
	  }
	}
      } else {
	if (verboseLevel > 1) {
	  printf
	      ("\nmultiple manager doesn't write the most probable coincidence:\n");
	  printf("diff between rsector = %d (minimum neigh = %d)\n",
		 maxDiffR, rsectorNeighbourOrder);
	}
	numberOfRejectedSinglesMultiples += 2;
      }

      /* kill'em all */
      if ((dlist_remove(plistP1, mostProb1, (void **) &data)) == 0 &&
	  (plistP1->destroy != NULL)) {
	plistP1->destroy(data);
      }
      if ((dlist_remove(plistP1, mostProb2, (void **) &data)) == 0 &&
	  (plistP1->destroy != NULL)) {
	plistP1->destroy(data);
      }
      stillToProcess -= 2;
    } else if (stillToProcess == 1) {	/* one single left in this multiple ; kill it */
      if (verboseLevel > 1)
	printf("multiple manager has killed this single\n");
      if (searcher)
	first = searcher->prev;
      else
	first = dlist_tail(plistP1);
      if ((dlist_remove(plistP1, first, (void **) &data)) == 0 &&
	  (plistP1->destroy != NULL)) {
	plistP1->destroy(data);
      }
      numberOfRejectedSinglesMultiples++;
      stillToProcess = 0;
    }
  }
}

/*******************************************************

                     multipleCombinatCoincidenceMgr

              Called if you do not want to simply remove
              multiple coincidences (saveMultipleBool == 1)
              destroyer is the oldest element to destroy. there is count-1
              elements to manage on his "head"
              Ex:  count = 4
              
              ...
              elt to manage
              elt to manage
              elt to manage
              destroyer to manage
              ...
               
              Associated e1 with e2  and e2 with e3 
              and also e1 with e3 if e1,e2,e3 are a triplet
              doesnt associate e1 with e3 
              if |time(e1) - time(e3)| > CW
              

********************************************************/
void multipleCombinatCoincidenceMgr(LIST * plistP1, ELEMENT * destroyer,
				    int count)
{
  void *data;
  ELEMENT *searcher, *first, *second;
  int localCount, i, j;
  int stillToProcess = 0;	// number of singles still to process
  u64 diffT = 0;
  /*   EVENT_RECORD *triplet[3]; */

  localCount = count;
  searcher = destroyer->next;
  stillToProcess = localCount;

  multipleID++;
  // luc
  /*if(count == 3)
     {
     second = destroyer;
     for(i=0 ; i < count ; i++)
     {
     triplet[i] = (EVENT_RECORD*)second->data;
     second = second->prev;
     }
     tripletAnalysis(pEncoHC,pEHC,triplet);
     }
   */
  if (verboseLevel) {
    printf("\nMultiple manager starts to check %d singles:\n", localCount);
    second = destroyer;
    for (i = 0; i < count; i++) {
      printf("time [%d] =%llu\n", i,
	     getTimeOfThisEVENT((EVENT_RECORD *) second->data));
      second = second->prev;
    }
    if (verboseLevel > 3)
      getchar();
  }

  while (stillToProcess > 0) {	/* while all singles are not destroyed */
    if (stillToProcess > 1) {
      if (searcher)
	first = searcher->prev;
      else
	first = dlist_tail(plistP1);
      second = first;

      for (j = 0; j < stillToProcess - 1; j++) {
	second = first;
	for (i = 0; i < stillToProcess - j - 1; i++) {
	  second = second->prev;
	  diffT = diffTime(first, second, pEncoHC);
	  if (diffT < coincidence_window) {

	    okForWriteInLMF = TRUE;

	    if (!saveAutoCoinciBool)
	      okForWriteInLMF = checkForAutoCoinci(first, second, pEncoHC);

	    /*              Eventually applied a dead time on the coincidence  */
	    /*               but with the single time accuracy */
	    if (okForWriteInLMF) {
	      if (deadTimeForCoincidenceSortingMode)
		okForWriteInLMF = deadTimeCoinciMgr(first, pEncoHC);
	    }
	    if (okForWriteInLMF) {
	      if (verboseLevel) {
		printf
		    ("write a coinci %llu with %llu (diff = %llu) and rsector = %d and %d\n",
		     getTimeOfThisEVENT((EVENT_RECORD *) first->data),
		     getTimeOfThisEVENT((EVENT_RECORD *) second->data),
		     diffTime(first, second, pEncoHC),
		     getRsectorID(pEncoHC, (EVENT_RECORD *) first->data),
		     getRsectorID(pEncoHC, (EVENT_RECORD *) second->data));
	      }
	      //     statistic
	      if (pEHC->gateDigiBool == TRUE) {
		if (pERC->pGDR->eventID[0] != pERC->pGDR->eventID[1])
		  nMulRandomCoincidences++;
		else if (pERC->pGDR->numberCompton[0]
			 || pERC->pGDR->numberCompton[1])
		  nMulScatterCoincidences++;
		else
		  nMulTrueCoincidences++;
	      }

	      fillCoinciRecord(pEncoHC, pEHC, pGDHC,
			       (EVENT_RECORD *) (first->data),
			       (EVENT_RECORD *) (second->data), multipleID,
			       nN, verboseLevel, pERC);
	      outputCoincidence(pEncoHC, pEHC, pGDHC, pcCC, pERC,
				getCoincidenceOutputMode());
	      nMultiplesWritten++;
	    }
	  }
	}
	first = first->prev;
	/* kill one */
	if (first->next) {
	  if ((dlist_remove(plistP1, first->next, (void **) &data)) == 0 &&
	      (plistP1->destroy != NULL)) {
	    plistP1->destroy(data);
	  }
	  stillToProcess -= 1;
	}
      }

    } else if (stillToProcess == 1) {	/* one single left in this multiple ; kill it */
      if (verboseLevel > 1)
	printf("multiple manager has killed this single\n");
      if (searcher)
	first = searcher->prev;
      else
	first = dlist_tail(plistP1);
      if ((dlist_remove(plistP1, first, (void **) &data)) == 0 &&
	  (plistP1->destroy != NULL)) {
	plistP1->destroy(data);
      }
      numberOfRejectedSinglesMultiples++;
      stillToProcess = 0;
    }
  }
}



/*******************************************************

                    checkForAutoCoinci

              Called if you do want to remove
              auto coincidences (coinci in same Rsector) 
              (saveAutoCoinciBool must be 0)
              This function checks if the 2 rsector ID difference is
              less than rsectorNeigh.
              Returns 1 if it is and 0 else

********************************************************/

int checkForAutoCoinci(ELEMENT * e1, ELEMENT * e2,
		       const ENCODING_HEADER * pEncoHC)
{
  int okForWrite;
  u16 diffSector;

  if (verboseLevel) {
    printf("\t < Checker for auto coincidence\t");
  }


  diffSector = diffRsector(e1, e2, pEncoHC);



  if (diffSector == 0) {
    okForWrite = 0;
    nAutos++;
    if (verboseLevel)
      printf(": This is an auto coinci  >\n");

  } else if (rsectorNeighbourOrder != 0) {

    if (diffSector > rsectorNeighbourOrder) {
      //ok 
      okForWrite = 1;
      nNonAutos++;
      if (verboseLevel)
	printf(": This is not  a neighbour rsector coinci  diff = %d >\n",
	       diffSector);
    } else {

      if (verboseLevel)
	printf("coincidence removed because neigh diff = %d order =%d\n",
	       diffSector, rsectorNeighbourOrder);
      // not ok
      okForWrite = 0;
      nAutos++;
    }


  } else {
    okForWrite = 1;
    nNonAutos++;
    if (verboseLevel)
      printf(": This is not an auto coinci >\n");

  }

  if (verboseLevel)
    printf("okforwrite after checking for autos and neigh = %d\n",
	   okForWrite);


  return (okForWrite);

}

/*******************************************************

                    incrementNumberOfSingleton



********************************************************/

void incrementNumberOfSingleton(void)
{
  numberOfSingleton++;
  numberOfTooLateSingleton++;
}

/*******************************************************

                    diffRsector

              Compute difference between 2 rsectorID


********************************************************/

int diffRsector(ELEMENT * e1, ELEMENT * e2,
		const ENCODING_HEADER * pEncoHC)
{


  static u8 diffRsectordoneOnce = FALSE;
  static u16 numberOfSectors = 0;
  u16 *pcrist;
  u16 rsector1, rsector2, buf, diff1, diff2;
  if (diffRsectordoneOnce == FALSE) {
    if (pEncoHC->scannerTopology.numberOfRings > 1) {
      printf
	  ("*** ERROR : oneList_cleanKit.c : diffRsector not implemented\n");
      printf("yet for more than 1 axial rsector scanner\n");
      exit(0);
    }
    numberOfSectors = pEncoHC->scannerTopology.numberOfSectors;
    diffRsectordoneOnce = TRUE;

  }


  /*   get rsector 1 ID */
  pcrist = demakeid(((EVENT_RECORD *) (e1->data))->crystalIDs[0], pEncoHC);
  rsector1 = pcrist[4];
  free(pcrist);


  /*   get rsector 2 ID */
  pcrist = demakeid(((EVENT_RECORD *) (e2->data))->crystalIDs[0], pEncoHC);
  rsector2 = pcrist[4];
  free(pcrist);

  if (rsector2 < rsector1) {	/* makes 1 < 2 */
    buf = rsector1;
    rsector1 = rsector2;
    rsector2 = buf;
  }

  /* find the shortest way between rsector */
  diff1 = rsector2 - rsector1;
  diff2 = rsector1 + (numberOfSectors - rsector2);

  /* return shortest rsector diff */
  if (diff1 >= diff2)
    return (diff2);
  else
    return (diff1);


}

/*******************************************************

                    diffRsectorER

              same than diffRsector but parameters are
              ER and not ELEMENT


********************************************************/

int diffRsectorER(EVENT_RECORD * e1, EVENT_RECORD * e2,
		  const ENCODING_HEADER * pEncoHC)
{


  static u8 diffRsectordoneOnce = FALSE;
  static u16 numberOfSectors = 0;
  u16 *pcrist;
  u16 rsector1, rsector2, buf, diff1, diff2;
  if (diffRsectordoneOnce == FALSE) {
    if (pEncoHC->scannerTopology.numberOfRings > 1) {
      printf
	  ("*** ERROR : oneList_cleanKit.c : diffRsectorER not implemented\n");
      printf("yet for more than 1 axial rsector scanner\n");
      exit(0);
    }
    numberOfSectors = pEncoHC->scannerTopology.numberOfSectors;
    diffRsectordoneOnce = TRUE;

  }


  /*   get rsector 1 ID */
  pcrist = demakeid(e1->crystalIDs[0], pEncoHC);
  rsector1 = pcrist[4];
  free(pcrist);


  /*   get rsector 2 ID */
  pcrist = demakeid(e2->crystalIDs[0], pEncoHC);
  rsector2 = pcrist[4];
  free(pcrist);

  if (rsector2 < rsector1)	// makes 1 < 2
  {
    buf = rsector1;
    rsector1 = rsector2;
    rsector2 = buf;
  }

  /* find the shortest way between rsector */
  diff1 = rsector2 - rsector1;
  diff2 = rsector1 + (numberOfSectors - rsector2);

  /* return shortest rsector diff */
  if (diff1 >= diff2)
    return (diff2);
  else
    return (diff1);


}

/*******************************************************

                    diffTime

              Compute time difference between 2 event


********************************************************/

u64 diffTime(ELEMENT * e1, ELEMENT * e2, const ENCODING_HEADER * pEncoHC)
{

  u64 time1, time2, diff;


  /*    get time1  */
  time1 = getTimeOfThisEVENT((EVENT_RECORD *) e1->data);
  /*    get time2  */
  time2 = getTimeOfThisEVENT((EVENT_RECORD *) e2->data);

  if (time2 < time1)		// makes 1 < 2
    diff = time1 - time2;
  else
    diff = time2 - time1;
  return (diff);

}
