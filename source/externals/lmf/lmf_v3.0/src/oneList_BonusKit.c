/*-------------------------------------------------------

List Mode Format 
                        
--  oneList_BonusKit.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of oneList_BonusKit.c:

This file contains a set of functions used to manage 
the doubly linked list (LIST) and the ELEMENT structure
Very useful for the coincidence sorter.
-------------------------------------------------------*/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "lmf.h"


static int verboseLevel, coincidence_window;

static u64 stack_cut_time;



void initBonusKit()
{

  verboseLevel = setCSverboseLevel();
  coincidence_window = setCScoincidencewindow();
  stack_cut_time = setCSstackcuttime();

}



/*******************************************************

                         newER

               A complete dynamic allocation
                   for an EVENT_RECORD

********************************************************/
EVENT_RECORD *newER(const EVENT_HEADER * pEH)
{
  EVENT_RECORD *pER;

  if ((pER = (EVENT_RECORD *) malloc(sizeof(EVENT_RECORD))) == NULL) {
    printf("\n*** ERROR : oneList_bonus.c : impossible to do malloc\n");
    exit(1);
  }
  if ((pER->crystalIDs =
       malloc((pEH->coincidenceBool + 1) * (pEH->numberOfNeighbours +
					    1) * sizeof(u64))) == NULL) {
    printf("\n*** ERROR : oneList_bonus.c : impossible to do malloc\n");
    exit(1);
  }

  if ((pER->energy =
       malloc((pEH->coincidenceBool + 1) * (pEH->numberOfNeighbours +
					    1) * sizeof(u8))) == NULL) {
    printf("\n*** ERROR : oneList_bonus.c : impossible to do malloc\n");
    exit(1);
  }

  if (pEH->gateDigiBool) {
    pER->pGDR = newGDR();

  } else
    pER->pGDR = NULL;

  return (pER);
}


/*******************************************************

                         newGDR

               A complete dynamic allocation
                   for an GATE_DIGI_RECORD

********************************************************/
GATE_DIGI_RECORD(*newGDR(void))
{
  GATE_DIGI_RECORD *pGDR;

  if ((pGDR =
       (GATE_DIGI_RECORD *) malloc(sizeof(GATE_DIGI_RECORD))) == NULL) {
    printf("\n*** ERROR : oneList_bonus.c : impossible to do malloc\n");
    exit(1);
  }
  return (pGDR);
}



/*******************************************************

                        freeGDR

                It frees completly the 
                      GATE_DIGI_RECORD

********************************************************/
void freeGDR(GATE_DIGI_RECORD * pGDR)
{
  if (pGDR)
    free(pGDR);


}

/*******************************************************

                        freeER

                It frees completly the 
                      EVENT_RECORD

********************************************************/
void freeER(EVENT_RECORD * pER)
{

  free(pER->crystalIDs);
  free(pER->energy);
  freeGDR(pER->pGDR);
  free(pER);


}


/*******************************************************

                        copyER

                  It copies completly the 
               containing of an EVENT_RECORD
                     in another one

********************************************************/
void copyER(const EVENT_RECORD * pER_source,
	    EVENT_RECORD * pER_destination, const EVENT_HEADER * pEH)
{
  int i;
  for (i = 0; i < 8; i++)
    pER_destination->timeStamp[i] = pER_source->timeStamp[i];

  pER_destination->timeOfFlight = pER_source->timeOfFlight;
  pER_destination->gantryAxialPos = pER_source->gantryAxialPos;
  pER_destination->gantryAngularPos = pER_source->gantryAngularPos;
  pER_destination->sourceAngularPos = pER_source->sourceAngularPos;
  pER_destination->sourceAxialPos = pER_source->sourceAxialPos;
  pER_destination->fpgaNeighInfo[0] = pER_source->fpgaNeighInfo[0];

  for (i = 0;
       i < ((pEH->coincidenceBool + 1) * (pEH->numberOfNeighbours + 1));
       i++) {
    pER_destination->crystalIDs[i] = pER_source->crystalIDs[i];

    pER_destination->energy[i] = pER_source->energy[i];
  }

  if (pEH->gateDigiBool)
    copyGDR(pER_source->pGDR, pER_destination->pGDR);
  return;
}

/*******************************************************

                        copyGDR

                  It copies completly the 
               containing of a GATE_DIGI_RECORD
                     in another one

********************************************************/
void copyGDR(GATE_DIGI_RECORD * pGDR_source,
	     GATE_DIGI_RECORD * pGDR_destination)
{
  *pGDR_destination = *pGDR_source;


}

/*******************************************************

                        print_list

                      It displays the 
                    doubly-linked list

********************************************************/
void print_list(const LIST * list)
{
  ELEMENT *element;
  EVENT_RECORD *pERcurrent;
  int i;
  fprintf(stdout, "List size is %d\n", dlist_size(list));
  if (dlist_size(list) != 0) {
    i = 0;
    element = dlist_head(list);
    while (1) {
      pERcurrent = element->data;
      printf("list[%d]\t time = %llu nanos\t\t\t %llu picos\n",
	     i,
	     (getTimeOfThisEVENT(pERcurrent) / 1000),
	     getTimeOfThisEVENT(pERcurrent));

      i++;
      if (dlist_is_tail(element))
	break;
      else {

	if (getTimeOfThisEVENT((EVENT_RECORD *) element->data) <=
	    (getTimeOfThisEVENT((EVENT_RECORD *) element->next->data) +
	     coincidence_window)) {
	  printf("\t+");
	}
	element = dlist_next(element);
      }
    }
    return;
  }

}

/*******************************************************

                      locateInList

              Returns the chronological position
          of an EVENT_RECORD in a doubly-linked list
       as his previous element. NULL if position is head           

********************************************************/
LINK locateInList(const LIST * plist, EVENT_RECORD * pNew)
{
  static int mode = 0;
  static ELEMENT *myPrev;

  if (!mode)
    mode = getAndSetCSsearchMode();
  else if (mode == 1)
    myPrev = searchRecursive(plist, pNew);
  else if (mode == 2)
    myPrev = searchIterative(plist, pNew);

  return (myPrev);

}



/*******************************************************

                     searchRecursive

              Returns the chronological position
          of an EVENT_RECORD in a doubly-linked list
       as his previous element. NULL if position is head           
          Recursive search. Start from the head of list.

********************************************************/
LINK searchRecursive(const LIST * plist, EVENT_RECORD * pNew)
{
  static ELEMENT *myPrev;
  static int tailCloserBool;
  static u64 tNew, tTail, tHead;


  if (dlist_head(plist)) {	/*  if list not empty */
    tNew = getTimeOfThisEVENT(pNew);

    tTail = getTimeOfThisEVENT((EVENT_RECORD *) dlist_tail(plist)->data);
    tHead = getTimeOfThisEVENT((EVENT_RECORD *) dlist_head(plist)->data);

    if (tHead <= tNew)		/*  tnew > head */
      return (myPrev = NULL);
    else if (tNew <= tTail)	/*  tail > new */
      return (myPrev = dlist_tail(plist));
    else {			/*   tNew is between head and tail  */


      tailCloserBool = tailCloser(plist, tNew);

      if (tailCloserBool) {
	if (verboseLevel)
	  printf("\nrecur from tail\n");
	myPrev = searchRecursiveFromTail(plist, pNew);
      } else {
	if (verboseLevel)
	  printf("\nrecur from head\n");
	myPrev = searchRecursiveFromHead(plist, pNew);

      }
      return (myPrev);
    }
  } else			/*   if list empty   */
    return (myPrev = NULL);
}




/*******************************************************

                     searchRecursiveFromHead

              Returns the chronological position
          of an EVENT_RECORD in a doubly-linked list
       as his previous element. NULL if position is head           
          Recursive search. Start from the head of list.

********************************************************/
LINK searchRecursiveFromHead(const LIST * plist, EVENT_RECORD * pNew)
{
  static ELEMENT *myPrev;
  static int prevAssigned = FALSE;
  static u64 tNew;



  if (dlist_head(plist) != NULL) {

    if (prevAssigned == FALSE) {

      tNew = getTimeOfThisEVENT(pNew);
      myPrev = dlist_head(plist);
      if (tNew >= getTimeOfThisEVENT((EVENT_RECORD *) (myPrev->data))) {
	prevAssigned = FALSE;
	return (NULL);
      } else
	prevAssigned = TRUE;
    } else
      myPrev = myPrev->next;

    if (myPrev == NULL) {
      prevAssigned = FALSE;
      return (NULL);
    } else if (myPrev->next == NULL) {
      prevAssigned = FALSE;
      return (myPrev);
    }


    if ((tNew <= getTimeOfThisEVENT((EVENT_RECORD *) (myPrev->data))) &&
	(tNew >=
	 getTimeOfThisEVENT((EVENT_RECORD *) (myPrev->next->data)))) {
      prevAssigned = FALSE;
      return (myPrev);
    } else {
      locateInList(plist, pNew);
    }
  } else {
    prevAssigned = FALSE;
    return (NULL);
  }


  if (myPrev)
    return (myPrev);
  else
    return (NULL);
}

/*******************************************************

                     searchRecursiveFromTail

              Returns the chronological position
          of an EVENT_RECORD in a doubly-linked list
       as his previous element. NULL if position is head           
          Recursive search. Start from the head of list.

********************************************************/
LINK searchRecursiveFromTail(const LIST * plist, EVENT_RECORD * pNew)
{
  static ELEMENT *myNext;
  static int nextAssigned = FALSE;
  static u64 tNew;

  if (dlist_head(plist) != NULL) {	/*  list empty ? */

    if (nextAssigned == FALSE) {	/*  first time */

      tNew = getTimeOfThisEVENT(pNew);
      myNext = dlist_tail(plist);
      if (tNew <= getTimeOfThisEVENT((EVENT_RECORD *) (myNext->data))) {
	nextAssigned = FALSE;
	return (myNext);
      } else
	nextAssigned = TRUE;
    } else			/*  other times  */
      myNext = myNext->prev;

    if (myNext == NULL) {	/*  normally it doesn't happen  */
      nextAssigned = FALSE;
      return (NULL);
    } else if (myNext->prev == NULL) {	/*  position is head */
      nextAssigned = FALSE;
      return (NULL);
    }

    if ((tNew <= getTimeOfThisEVENT((EVENT_RECORD *) (myNext->prev->data)))
	&& (tNew >= getTimeOfThisEVENT((EVENT_RECORD *) (myNext->data)))) {
      nextAssigned = FALSE;
      return (myNext->prev);
    } else {
      locateInList(plist, pNew);
    }
  } else {
    nextAssigned = FALSE;
    return (NULL);
  }

  if (myNext) {
    if (myNext->prev)
      return (myNext->prev);
    else
      return (NULL);
  } else
    return (NULL);
}

/*******************************************************

                     searchIterative

              Returns the chronological position
          of an EVENT_RECORD in a doubly-linked list
       as his previous element. NULL if position is head           
                Used if the list is i32. 

                     searchIterativeFromHead
                     searchIterativeFromTail
               (no need to explain more)

********************************************************/
LINK searchIterative(const LIST * plist, EVENT_RECORD * pNew)
{
  static ELEMENT *myPrev;
  static int tailCloserBool;
  static u64 tNew, tTail, tHead;


  if (dlist_head(plist)) {	/*  if list not empty */
    tNew = getTimeOfThisEVENT(pNew);

    tTail = getTimeOfThisEVENT((EVENT_RECORD *) dlist_tail(plist)->data);
    tHead = getTimeOfThisEVENT((EVENT_RECORD *) dlist_head(plist)->data);

    if (tHead <= tNew)		/*  tnew > head */
      return (myPrev = NULL);
    else if (tNew <= tTail)	/*  tail > new */
      return (myPrev = dlist_tail(plist));
    else {			/*   tNew is between head and tail  */


      tailCloserBool = tailCloser(plist, tNew);

      if (tailCloserBool) {
	myPrev = searchIterativeFromTail(plist, pNew);
	if (verboseLevel)
	  printf("\niter  from tail\n");
      } else {
	myPrev = searchIterativeFromHead(plist, pNew);
	if (verboseLevel)
	  printf("\niter  from head\n");

      }
      return (myPrev);
    }
  } else			/*  if list empty   */
    return (myPrev = NULL);
}


LINK searchIterativeFromTail(const LIST * plist, EVENT_RECORD * pNew)
{
  static ELEMENT *current;
  static int order;
  static u64 tNew, tCurrent, tCurrentNext;


  order = getOrderOfListSize(plist);	/*  order = 100 if 1000 > list_size > 100  */

  tNew = getTimeOfThisEVENT(pNew);
  current = dlist_tail(plist)->prev;

  while (1) {

    if (!current)
      break;
    if (!current->next) {	/*  well it s a tail */
      tCurrent = getTimeOfThisEVENT((EVENT_RECORD *) current->data);
      if (tCurrent >= tNew)
	break;
      else
	current = bigStepUp(current, order, tNew);
    } else {


      tCurrent = getTimeOfThisEVENT((EVENT_RECORD *) current->data);
      tCurrentNext =
	  getTimeOfThisEVENT((EVENT_RECORD *) current->next->data);
      if ((tCurrent >= tNew) && (tNew >= tCurrentNext))
	break;
      else if (tCurrent <= tNew) {

	current = bigStepUp(current, order, tNew);


      } else if (tCurrentNext >= tNew) {

	current = bigStepDown(current, order, tNew);

      }
    }
    order = order / 10;
  }



  return (current);

}


LINK searchIterativeFromHead(const LIST * plist, EVENT_RECORD * pNew)
{
  static ELEMENT *current;
  static int order;
  static u64 tNew, tCurrent, tCurrentNext;


  order = getOrderOfListSize(plist);	/*  order = 100 if 1000 > list_size > 100 */

  tNew = getTimeOfThisEVENT(pNew);
  current = dlist_head(plist);

  while (1) {

    if (!current)
      break;
    if (!current->next) {	/*  well it s a tail */
      tCurrent = getTimeOfThisEVENT((EVENT_RECORD *) current->data);
      if (tCurrent >= tNew)
	break;
      else
	current = bigStepUp(current, order, tNew);
    } else {
      tCurrent = getTimeOfThisEVENT((EVENT_RECORD *) current->data);
      tCurrentNext =
	  getTimeOfThisEVENT((EVENT_RECORD *) current->next->data);


      if ((tCurrent >= tNew) && (tNew >= tCurrentNext))
	break;
      else if (tCurrent <= tNew) {

	current = bigStepUp(current, order, tNew);

      } else if (tCurrentNext >= tNew) {

	current = bigStepDown(current, order, tNew);

      }
    }
    order = order / 10;
  }



  return (current);

}





/*******************************************************

                   bigStepUp and bigStepDown

          with big step (order elements) go up or down the stack
               till current is inf. or sup. than tNew

********************************************************/


LINK bigStepUp(LINK current, int order, u64 tNew)
{


  while (1) {


    if (current) {

      if (getTimeOfThisEVENT((EVENT_RECORD *) current->data) >= tNew)
	break;
      else {
	current = upInList(order, current);

      }
    } else
      break;
  }



  return (current);
}

LINK bigStepDown(LINK current, int order, u64 tNew)
{


  while (1) {

    if (current->next) {
      if (getTimeOfThisEVENT((EVENT_RECORD *) current->next->data) <= tNew)
	break;
      else {
	current = downInList(order, current);

      }
    } else
      break;
  }


  return (current);
}




/*******************************************************

                   upInList & downInList

               go for order steps up (or down) in a list
                from the current element
                 if it reaches head or tail return null

********************************************************/

LINK upInList(int order, LINK current)
{

  static int count = 0;

  while (count < order) {
    if (current->prev) {
      count++;
      current = current->prev;
    } else
      break;

  }
  count = 0;
  return (current);

}

LINK downInList(int order, LINK current)
{

  static int count = 0;
  while (count < order) {
    if (current->next) {
      count++;
      current = current->next;
    } else
      break;

  }
  count = 0;
  return (current);

}

/*******************************************************

                            tailCloser

                      returns 0 if head is closer
                     than tail to the new element.
                            (1 else)                        
                          returns 2 if list size < 2

********************************************************/

int tailCloser(const LIST * plist, u64 tNew)
{

  int returnValue = 2;
  u64 tTail, tHead;
  if (dlist_size(plist) > 1) {
    tTail = getTimeOfThisEVENT((EVENT_RECORD *) dlist_tail(plist)->data);
    tHead = getTimeOfThisEVENT((EVENT_RECORD *) dlist_head(plist)->data);

    if (tNew + tNew <= tHead + tTail)	/*  if (tNew - tTail < tHead -tNew) */
      returnValue = 1;
    else
      returnValue = 0;
  }
  return (returnValue);
}




/*******************************************************

                       getOrderOfListSize

           Returns          0 if list is empty
                     1  if  1 < list_size <= 10
                    10 if  10 < list_size <= 100
                   100 if 100 < list_size <= 1000 
                 1000 if 1000 < list_size <= 10 000
               10000 if 10000 < list_size <= 100 000
             100000 if 100000 < list_size <= 1 000 000
                     etc...
  
********************************************************/

int getOrderOfListSize(const LIST * plist)
{
  int order = 0, size = 0;

  size = dlist_size(plist);

  if (size) {
    order = 1;
    while (size > order)
      order = 10 * order;

    order = order / 10;
  }
  return (order);
}


/*******************************************************

                       insertOK

           insert an Event Record in dlist next to
          location. if location is null push on top
           WARNING : The inserted element has CWN = 0

********************************************************/
int insertOK(LIST * plist, ELEMENT * location, const EVENT_RECORD * pERin)
{
  if (location == NULL) {	/*  push on top if location == NULL */
    if (dlist_ins_prev(plist, dlist_head(plist), pERin) != 0) {	/* insert it in dlistP2  */
      printf("\n***  oneList_bonus.c : error in insertOK\n");
      return (1);
    }
    dlist_head(plist)->CWN = 0;

  } else {			/*  else insert in the stack */

    if (dlist_ins_next(plist, location, pERin) != 0) {	/* insert it in dlistP2 */
      printf("\n***  oneList_bonus.c : error in insertOK\n");
      return (1);
    }
    location->next->CWN = 0;
  }
  return (0);
}
