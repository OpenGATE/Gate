/*-------------------------------------------------------

           List Mode Format 
                        
     --  oneList_flowchartt.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of oneList_flowchartt.c:

     This file contains all the functions of the 
     coincidence sorter flowchartt.
     This coincidence sorting is optimized and the 
     flow i8t is not very intuitive
     You must read the document "Coincidence Sorter 
     Implementation" of 
     ClearPET Project, to understand this code.
     Get it at : http://www-iphe.unil.ch/~PET
     created by luc.simon@iphe.unil.ch on june 2002 for CCC 
     collaboration

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

static int listInitDone = FALSE;
static int verboseLevel, coincidence_window;
static u64 stack_cut_time;

static u64 tEn = 0, tTail = 0, tLocation = 0, tLocation_next = 0;
static ELEMENT *location;
static EVENT_RECORD *pERin;
static LIST *plistP1;



void initFlowChart()
{
  verboseLevel = setCSverboseLevel();

  coincidence_window = setCScoincidencewindow();
  stack_cut_time = setCSstackcuttime();
  printf("sorting coincidences ...\n");
}


void fcWAY_1Y(LIST * plistP1fromStart, EVENT_RECORD * pERinFromStart)
{


  if (listInitDone == FALSE) {
    plistP1 = plistP1fromStart;
    listInitDone = TRUE;

  }


  pERin = pERinFromStart;	/* // to acces from evrywhere to pERin */
  tEn = getTimeOfThisEVENT(pERin);





  tTail = getTimeOfThisEVENT((EVENT_RECORD *) dlist_tail(plistP1)->data);


  if (verboseLevel)
    printf("Diamond 2\n");
  /****************************************

              DIAMOND 2 
         Is this element at tail ?

  *****************************************/


  if (tEn >= tTail) {
    fcWAY_2Y();			/*  // pERin not at tail */

  } else {
    fcWAY_2N();			/* // element at tail  */

  }

}


void fcWAY_2N(void)
{
  static u64 tTail_prev;

  insertOK(plistP1, dlist_tail(plistP1), pERin);	/* // insert at tail */

  if (verboseLevel)
    printf("Diamond 10\n");
  /****************************************

              DIAMOND 10 
         Coincidence with Tail ?

  *****************************************/
  tTail_prev =
      getTimeOfThisEVENT((EVENT_RECORD *) dlist_tail(plistP1)->prev->data);

  if (tTail_prev <= tEn + coincidence_window)	/* // tEn is tail now! */
    dlist_tail(plistP1)->prev->CWN = 1;	/* // tag the tail->prev */
  else
    dlist_tail(plistP1)->prev->CWN = 0;
  /* // else nothing for 10 -> no */

}
void fcWAY_2Y(void)
{

  if (verboseLevel)
    printf("Diamond 3\n");
  /****************************************

              DIAMOND 3 
          need to clean P1 ?

  *****************************************/
  if (tEn >= tTail + stack_cut_time) {

      /****************************************

                          CLEAN P1  

      *****************************************/
    if (verboseLevel) {
      printf("verbose = %d\n", verboseLevel);
      getchar();
      printf("\nThis %llu", tTail);
      printf("\nis too old with this new  %llu \n",
	     getTimeOfThisEVENT(pERin));
    }
    cleanListP1(plistP1, pERin);	/*  // clean too old elmts of P1 */

    fcWAY_3Y();			/* // after cleanP1 */

  } else {			/*  // no need to clean P1 */


    fcWAY_3N();

  }
}

void fcWAY_3N(void)
{

  location = locateInList(plistP1, pERin);

  if (verboseLevel)
    printf("Diamond 6\n");
 /****************************************

              DIAMOND 6 
         pERin position is head ?

 *****************************************/
  if (location != NULL) {
    fcWAY_6Y();			/*  // not on head */
  } else {
    fcWAY_6N();			/* // we are on head */
  }
}

void fcWAY_3Y(void)
{				/* // after cleaning */
  if (verboseLevel)
    printf("Diamond 4\n");
  /****************************************

              DIAMOND 4 
           P1 empty (after cleaning) ?

  *****************************************/
  if (plistP1->size > 0) {	/* // not empty */

    fcWAY_3N();
  } else {			/*  // empty */

    if (dlist_ins_prev(plistP1, dlist_head(plistP1), pERin) != 0)	/* //insert as first in dlistP1  */
      printf("\n*** error flowchartt.c : diamond 4\n");

    dlist_head(plistP1)->CWN = 0;
  }
}


void fcWAY_6N(void)
{
  static u64 tHead_next;


  insertOK(plistP1, NULL, pERin);	/* // push on head */

  tHead_next =
      getTimeOfThisEVENT((EVENT_RECORD *) dlist_head(plistP1)->next->data);

  if (verboseLevel)
    printf("Diamond 9\n");
  /****************************************

              DIAMOND 9 
        coincidence on head ?

  *****************************************/
  if (tEn <= tHead_next + coincidence_window) {	/* // tEn is head now !! */
    dlist_head(plistP1)->CWN = 1;	/* // tag the head */
  } else
    dlist_head(plistP1)->CWN = 0;
  /*  // else nothing for 9 -> no */

}
void fcWAY_6Y(void)
{

  if (verboseLevel)
    printf("Diamond 7\n");
  /****************************************

              DIAMOND 7 
       pERin position is tail ?

  *****************************************/

  if (location->next != NULL) {
    fcWAY_7Y();			/* // not at tail */
  } else {
    fcWAY_2N();			/* // we are at tail */
  }


}


void fcWAY_7Y(void)
{
  tLocation_next =
      getTimeOfThisEVENT((EVENT_RECORD *) location->next->data);
  tLocation = getTimeOfThisEVENT((EVENT_RECORD *) location->data);

  insertOK(plistP1, location, pERin);	/*  // insert at good position */
  if (verboseLevel)
    printf("Diamond 8\n");
  /****************************************

              DIAMOND 8 
       coincidence with prev

  *****************************************/
  if (tLocation <= tEn + coincidence_window) {
    location->CWN = 1;
  } else
    location->CWN = 0;
  /*  // else nothing */

  if (verboseLevel)
    printf("Diamond 11\n");
  /****************************************

              DIAMOND 11 
       coincidence with next

  *****************************************/
  if (tEn <= tLocation_next + coincidence_window) {
    location->next->CWN = 1;
  }
  /* // else nothing */

}
