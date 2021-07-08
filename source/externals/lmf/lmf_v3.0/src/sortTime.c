/*-------------------------------------------------------

List Mode Format 
                        
--  sortTime.c  --                      
  
Magalie.Krieguer@iphe.unil.ch
     
Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of sortTime.c:


Sort chronologically a singles file. Called by processRecordCarrier.c
called itself by LMFReader.c. 
The event are sorted in a doubly linked list of constant size
("listsize" events) When list size is reached, the oldest element 
is written for each new element that is coming.
To finish to write at the end, you must use these lines (in LMFreader()):

while(1)
{
if(finishTimeSorting(pEncoH,pEH,pCRH,pGDH,pcC,pER,pCRR,pf,bis_ccsFileName) == 0)
break;
} 
	  
-------------------------------------------------------*/

#include <stdio.h>
#include "lmf.h"

static FILE *pf = NULL;

static LIST timeList;
static int listsize = 100;	/*  100 is default value :  */
/*                             can be change from anywhere  */
/*                             by setTimeListSize(); */
static ELEMENT *location;
void *data;
EVENT_RECORD *pERin, *pERout;


void setTimeListSize(int size)
{
  listsize = size;
}



void sortTime(const ENCODING_HEADER * pEncoH,
	      const EVENT_HEADER * pEH,
	      const COUNT_RATE_HEADER * pCRH,
	      const GATE_DIGI_HEADER * pGDH,
	      const CURRENT_CONTENT * pcC,
	      const EVENT_RECORD * pER,
	      const COUNT_RATE_RECORD * pCRR,
	      FILE * pfile, const i8 * nameOfFile)
{

  static int doneOnce = FALSE;
  static int thisOne = FALSE;	/*   = FALSE if we dont return an event */



  if (doneOnce == FALSE) {	/*  first call */

    thisOne = FALSE;
    setSearchMode(1);
    dlist_init(&timeList, (void *) freeER);
    pERin = newER(pEH);		/*   complete allocatation for the very first element       */
    pERout = newER(pEH);	/*   complete allocatation for the output element       */

    copyER(pER, pERin, pEH);	/*  *pERin = *pER but safe */
    dlist_ins_prev(&timeList, dlist_head(&timeList), pERin);	/* insert the first in timeList  */

    doneOnce = TRUE;
  } else {			/*  all calls except first one */



    if (timeList.size < listsize) {	/*   first "listsize"th calls  */
      pERin = newER(pEH);	/*   complete allocatation for an element         */
      copyER(pER, pERin, pEH);	/*   *pERin = *pER but safe         */




      if (timeList.size == 1) {
	if (getTimeOfThisEVENT(pERin) <
	    getTimeOfThisEVENT((EVENT_RECORD *) dlist_head(&timeList)->
			       data)) {
	  dlist_ins_next(&timeList, dlist_head(&timeList), pERin);
	} else {

	  dlist_ins_prev(&timeList, dlist_head(&timeList), pERin);
	}
      } else {
	location = locateInList(&timeList, pERin);	/*  find location in list */



	insertOK(&timeList, location, pERin);	/*  insert element at good position */

      }


      thisOne = FALSE;
    } else {			/*  other calls */


      thisOne = TRUE;

      /*   remove last element from list and return it */
      copyER((EVENT_RECORD *) dlist_tail(&timeList)->data, pERout, pEH);	/*  pERout = tail but safe */
      if ((dlist_remove(&timeList, dlist_tail(&timeList), (void **) &data))
	  == 0 && ((&timeList)->destroy != NULL)) {
	(&timeList)->destroy(data);
      }

      /*       insert mew element in list */
      pERin = newER(pEH);	/*   complete allocatation for an element */
      copyER(pER, pERin, pEH);	/*   *pERin = *pER but safe */
      location = locateInList(&timeList, pERin);	/*  find location in list */
      insertOK(&timeList, location, pERin);	/*   insert element at good position */

    }
  }
  if (thisOne == TRUE)
    LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pERout, pCRR, &pf,
	       bis_ccsFileName);

}




int finishTimeSorting(const ENCODING_HEADER * pEncoH,
		      const EVENT_HEADER * pEH,
		      const COUNT_RATE_HEADER * pCRH,
		      const GATE_DIGI_HEADER * pGDH,
		      const CURRENT_CONTENT * pcC,
		      const EVENT_RECORD * pER,
		      const COUNT_RATE_RECORD * pCRR,
		      FILE * pfile, const i8 * nameOfFile)
{
  if (timeList.size != 0) {
    copyER((EVENT_RECORD *) dlist_tail(&timeList)->data, pERout, pEH);	/*  pERout = tail but safe */
    if ((dlist_remove(&timeList, dlist_tail(&timeList), (void **) &data))
	== 0 && ((&timeList)->destroy != NULL)) {
      (&timeList)->destroy(data);
    }
    LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pERout, pCRR, &pf,
	       bis_ccsFileName);
    return (1);
  } else
    return (0);


}

void destroySortTime(void)
{


  if (pERin) {
    freeER(pERin);
  }


  dlist_destroy(&timeList);


}
