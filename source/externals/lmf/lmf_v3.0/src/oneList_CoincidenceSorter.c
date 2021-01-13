/*-------------------------------------------------------

List Mode Format 
                        
--  oneList_CoincidenceSorter.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of oneList_CoincidenceSorter.c:

coincidence sorter

You must read the document "Coincidence Sorter Implementation" of 
ClearPET Project, to understand this code.
Get it : http://www-iphe.unil.ch/~PET
created by luc.simon@iphe.unil.ch on june 2002 for CCC collaboration


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"



static int headDone = FALSE;
static int verboseLevel = 0, coincidence_window = 0;
static u64 stack_cut_time = 0;

static int verboseSet = FALSE, vL = 0;
static int coinciWinSet = FALSE, cw;
static int sctSet = FALSE;
static int multipleBoolSet = FALSE, mb;
static int autoCoinciBoolSet = FALSE, acb;
static int rsectorNeighOrderSet = FALSE, rno;
static int searchingModeSet = FALSE, sm;


static int coincidenceDeadTimeValueSet = FALSE;
static u64 cdtv;

static int coincidenceDeadTimeModeSet = FALSE, cdtm;


static u64 sct;

static LIST listP1;

static int maxlistsize;




void setAllCSparameters(int inlineverbose,
			int inlineCW,
			u64 inlineSCT,
			int inlineSearchMode,
			int inlineMB,
			int inlineAB,
			int inlineRNO, u64 inlineCDTV, int inlineCDTM)
{
  /*
     set all CS parameter, but you have to respct the units

     * verbose (between 0 and 9)
     * CW coincidence window in picosecond
     * SCT stack cut time in picosecond
     * Searchmode 1 or 2 1 recursif 2 non recursif
     * Multiple bool (keep the multiple coinci if 1)
     * Auto bool (keep the auto coinci if 1)

   */
  if (!verboseSet) {
    vL = inlineverbose;
    verboseSet = TRUE;
  }

  if (!coinciWinSet) {
    cw = inlineCW;
    coinciWinSet = TRUE;
  }

  if (!sctSet) {
    sct = inlineSCT;
    sctSet = TRUE;
  }

  if (!searchingModeSet) {
    sm = inlineSearchMode;
    searchingModeSet = TRUE;
  }

  if (!multipleBoolSet) {
    mb = inlineMB;
    multipleBoolSet = TRUE;
  }
  if (!autoCoinciBoolSet) {
    acb = inlineAB;
    autoCoinciBoolSet = TRUE;
  }
  if (!rsectorNeighOrderSet) {
    rno = inlineRNO;
    rsectorNeighOrderSet = TRUE;
  }



  cdtv = inlineCDTV;
  coincidenceDeadTimeValueSet = TRUE;
  cdtm = inlineCDTM;
  coincidenceDeadTimeModeSet = TRUE;


  printf("All parameters for coincidence sorting :\n");
  printf("verbosity = %d\n", vL);
  printf("window = %d ns\n", cw / 1000);
  printf("stack cut time = %d ns\n", (int) (sct / 1000));
  if (sm == 1)
    printf("Recursive search mode\n");
  if (sm == 2)
    printf("Iterative search mode\n");
  printf("multiple keep bool = %d\n", mb);
  printf("order of neighbour for sector = %d\n", rno);
  printf("auto coincidence keep bool = %d\n", acb);

  if (cdtm) {
    printf("Dead time ");
    if (cdtm == 1)
      printf("paralysable");
    if (cdtm == 2)
      printf("non paralysable");
    printf(" for coincidence writing : %llu\n", cdtv);
  }
}
void setSearchMode(int searchingMode)
{
  if (!searchingModeSet) {
    sm = searchingMode;
    searchingModeSet = TRUE;
  }
}



int getAndSetCSsearchMode()
{

  if (!searchingModeSet) {
    printf
	("Choose searching mode to locate elements in list (recursive mode recommanded) :\n");
    printf("1 Recursive mode\t");
    printf("2 Iterative mode\n");
    sm = hardgeti16(1, 2);
    searchingModeSet = TRUE;
  }

  return (sm);

}


u64 setCSdtValue()
{

  if (!coincidenceDeadTimeValueSet) {

    if (cdtm) {
      printf
	  (" Value of dead time for coincidence writting (picoseconds):");
      scanf("%llu", &cdtv);
      getchar();		/*   user choice */
    } else
      cdtv = 0;

    printf("\ndead time value for coincidence writting  %llu (ps)\n",
	   cdtv);

    coincidenceDeadTimeValueSet = TRUE;
  }

  return (cdtv);
}


int setCSdtMode()
{

  if (!coincidenceDeadTimeModeSet) {
    printf("Dead time mode for coincidence writting :\n");
    printf("1 Paralysable \n");
    printf("2 Non paralysable\n");
    printf("0 No dead time on coinci writing\n");

    cdtm = hardgeti16(0, 2);
    printf("dead time mode for coincidence writting  %d\n", cdtm);

    coincidenceDeadTimeModeSet = TRUE;
  }

  return (cdtm);
}



int setCSverboseLevel()
{
  maxlistsize = 0;
  if (!verboseSet) {
    printf
	("Enter verbose level for coincidence sorter : (0 recommanded)\n");
    /*       vL = 0; */
    vL = hardgeti16(0, 9);
    printf("verbose level = %d\n", vL);
    verboseSet = TRUE;
  }


  return (vL);

}


int setCSsaveMultipleBool()
{


  if (!multipleBoolSet) {


    /*     the 3 possibilities to choose  */

    /*        ----------------------------- */
    /*        1. automatically set : */
    /*        mb = 0;    set to 0 */
    /*        ------------------------------ */
    /*        2. ask to user : */
    printf("Do you want to save multiples ? (no recommanded) :");
    mb = hardgetyesno();	/*  user choice */
    /*       printf("\n"); */
    /*        ------------------------------ */
    /*        3. symbolic constant : */
    /*        mb = SAVE_MULTIPLE_BOOL;  in constant_CCS.h */
    /*        ------------------------------      */
    printf("multiple bool  = %d  \n", mb);
    multipleBoolSet = TRUE;
  }
  return (mb);
}

int setCSsaveAutoCoinciBool()
{

  if (!autoCoinciBoolSet) {
    /*    the 3 possibilities to choose  */
    /*        ----------------------------- */
    /*        1. automatically set : */
    /*        acb = 0;    set to 0 */
    /*        ------------------------------ */
    /*        2. ask to user : */
    printf
	("Do you want to save auto coincidences (same Rsector) ? (no recommanded) :");
    acb = hardgetyesno();	/*  user choice */
    /*        printf("\n"); */
    /*        ------------------------------ */
    /*        3. symbolic constant : */
    /*        mb = SAVE_AUTO_COINCI_BOOL;  in constant_CCS.h */
    /*        ------------------------------      */
    printf("auto coincidences bool  = %d  \n", acb);
    autoCoinciBoolSet = TRUE;
  }
  return (acb);
}

int setCSrsectorNeighOrder()
{

  if (!rsectorNeighOrderSet) {
    printf("Do you want to save coincidences in neighb. rsectors\n ");
    printf("(ignored if 'y' to previous question) ? \n");
    printf("Choose the order of rejection (Ex.: if 2 \n");
    printf
	("coincidences are accepted if |rsectorID1 - rsectorID2}| > 2 ) ");
    rno = hardgeti16(1, 9);	/*  user choice */
    printf(" %d \n", rno);
    rsectorNeighOrderSet = TRUE;
  }
  return (rno);
}

int setCScoincidencewindow()
{


  if (!coinciWinSet) {


    /*       the 3 possibilities to choose coincidence window */

    /*        ----------------------------- */
    /*        1. automatically set : */
    /*        cw = 2;    set to 2 */
    /*        ------------------------------ */
    /*        2. ask to user : */
    printf("Enter coincidence window (nanoseconds, ex : 10) : ");

    scanf("%d", &cw);		/*  user choice */
    cw = 1000 * cw;
    /*       printf("\n"); */
    /*        ------------------------------ */
    /*        3. symbolic constant : */
    /*        cw = COINCIDENCE_WINDOW;  in constant_CCS.h */
    /*        ------------------------------      */
    printf("coincidence window = %d picoseconds \n", cw);

    if (OF_is_Set())		/*  write in an ascii file */
      fprintf(getAndSetOutputFileName(), "%d\n", cw);

    coinciWinSet = TRUE;
  }
  return cw;
}

int setCSstackcuttime()
{

  if (!sctSet) {
    printf
	("Enter stack cut time (microseconds) ( 10 to 50 recommanded and 0 for help) : ");
    sct = 10;

    /*       the 3 possibilities to choose coincidence window */

    /*        ----------------------------- */
    /*        1. automatically set : */
    /*              sct = 500000;    set to 500000 */
    /*        ------------------------------ */
    /*        2. ask to user in micros seconds: */
    scanf("%llu", &sct);
    getchar();			/*  user choice */

    while (!sct) {
      helpForStackCutTime();
      printf
	  ("Enter stack cut time (microseconds) ( 10 to 50 recommanded and 0 for help) : ");
      scanf("%llu", &sct);
      getchar();		/*  user choice */
    }




    sct = sct * 1000000;
    printf("\n");

    /*        ------------------------------ */
    /*        3. symbolic constant : */
    /*        sct = STACK_CUT_TIME;  in constant_CCS.h */
    /*        ------------------------------      */
    printf("stack cut time = %llu picoseconds (%llu microseconds) \n", sct,
	   sct / 1000000);

    if (OF_is_Set())		/*  write in an ascii file */
      fprintf(getAndSetOutputFileName(), "%llu\n", sct);


    sctSet = TRUE;
  }
  return sct;
}




int sortCoincidence(const ENCODING_HEADER * pEncoH,
		    const EVENT_HEADER * pEH,
		    const COUNT_RATE_HEADER * pCRH,
		    const GATE_DIGI_HEADER * pGDH,
		    const CURRENT_CONTENT * pcC,
		    const EVENT_RECORD * pER,
		    const COUNT_RATE_RECORD * pCRR)
{

  EVENT_RECORD *pERin;
  static u64 tEn = 0;

  maxlistsize = 0;
  /****************************************

             Initialize the list

  *****************************************/

  if (headDone == FALSE) {	/*   only once-done block  */

    verboseLevel = setCSverboseLevel();
    coincidence_window = setCScoincidencewindow();
    stack_cut_time = setCSstackcuttime();
    getAndSetCSsearchMode();

    initBonusKit();
    initCleanKit();
    initFlowChart();


    headDone = TRUE;
    printf("\n");

    dlist_init(&listP1, (void *) freeER);	/*  init the listP1  */

    initCoinciFile(pEncoH, pEH, pGDH, pCRH, pcC);
    pERin = newER(pEH);		/*  complete allocatation for the very first element */

    copyER(pER, pERin, pEH);	/*   *pERin = *pER but safe */


    if (dlist_ins_prev(&listP1, dlist_head(&listP1), pERin) != 0)	/*  insert the first in dlistP1 */
      return 1;
    dlist_head(&listP1)->CWN = 0;

  } else {			/*  block not done the first time but the other times */


      /***************************************
                 show me new event
      ***************************************/
    tEn = getTimeOfThisEVENT(pER);

    if (listP1.size > maxlistsize)
      maxlistsize = listP1.size;

    if (verboseLevel) {
      printf("\nA new event has come ; %llu", tEn);
      printf("----------------------------------\n");

    }



    if (verboseLevel)
      printf("Diamond 1\n");
      /****************************************

                   DIAMOND 1  
                   P1 not empty ?

      *****************************************/

    if (listP1.size > 0) {

      if (verboseLevel)
	printf("Diamond 5\n");
	  /****************************************

                   Diamond 5
                   Send to trash the ones 
                   that come too late 

	  *****************************************/

      if (getTimeOfThisEVENT((EVENT_RECORD *) dlist_head(&listP1)->data) >=
	  (tEn + stack_cut_time)) {

	incrementNumberOfSingleton();

	if (verboseLevel) {
	  printf
	      ("\nThis element is not processed because it comes too late :%llu\n",
	       tEn);
	  printf("SCT = %llu\n", stack_cut_time);
	  printf("head P1 = %llu\n",
		 getTimeOfThisEVENT((EVENT_RECORD *) dlist_head(&listP1)->
				    data));
	  if (verboseLevel > 2)
	    getchar();
	}
	return (0);

      } else {

	pERin = newER(pEH);	/*  complete allocatation for an element */

	copyER(pER, pERin, pEH);	/*  *pERin = *pER but safe */

	fcWAY_1Y(&listP1, pERin);	/*  enter the flow i8t way 1 -> yes */

      }
    } else {

      pERin = newER(pEH);	/*  complete allocatation for an element */
      copyER(pER, pERin, pEH);	/*  *pERin = *pER but safe */
      if (dlist_ins_prev(&listP1, dlist_head(&listP1), pERin) != 0)	/* insert the first in dlistP1 */
	return 1;		/*  insert as first element */

      dlist_head(&listP1)->CWN = 0;
    }


  }


  /****************************************

              PRINT THEN END 

  *****************************************/


  if (verboseLevel) {

    print_list(&listP1);
    if (verboseLevel > 2)
      getchar();
  }
  return (0);
}


void destroyList()
{

  printf("\n\nMax list size was : %d\n\n\n", maxlistsize);
  headDone = FALSE;

  verboseSet = FALSE;
  coinciWinSet = FALSE;
  sctSet = FALSE;
  multipleBoolSet = FALSE;
  autoCoinciBoolSet = FALSE;
  searchingModeSet = FALSE;

  finalCleanListP1(&listP1);

  dlist_destroy(&listP1);

  fprintf(stdout, " list P1 destoyed\n");

  deastroyDeadTimeCoinciMgr();



}
