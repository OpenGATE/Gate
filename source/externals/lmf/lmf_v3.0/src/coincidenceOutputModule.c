/*-------------------------------------------------------

           List Mode Format 
                        
     --  coincidenceOutputModule.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of coincidenceOutputModule.c:
     This module manages the output of coincidence
     sorted by the coincidence sorter.
     coincidenceOutputMode = 0 means that we store coincidence in a file
     coincidenceOutputMode = 1 means that we store coincidence in a list
     to be treated before the writting.


-------------------------------------------------------*/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "lmf.h"

static i8 coincidenceOutputMode = 0;
/*
  mode 0 LMF file
  mode 1 list of coincidence 
*/

static FILE *pfC = NULL;	/* // file of coinci */
static LIST listC;		/* // list of coinci */
static int headDone = FALSE;
static EVENT_RECORD *pERCin;
void outputCoincidence(const ENCODING_HEADER * pEncoHC,
		       const EVENT_HEADER * pEHC,
		       const GATE_DIGI_HEADER * pGDHC,
		       const CURRENT_CONTENT * pcCC,
		       const EVENT_RECORD * pERC, i8 coincidenceOutputMode)
{

  if (coincidenceOutputMode == 0) {
    LMFCbuilder(pEncoHC, pEHC, pGDHC, pcCC, pERC, &pfC, coinci_ccsFileName);	/* ...write it */

  }
  if (coincidenceOutputMode == 1) {
    if (headDone == FALSE) {
      headDone = TRUE;

    }

    pERCin = newER(pEHC);	/* // complete allocatation for the very first element */
    copyER(pERC, pERCin, pEHC);	/* // *pERCin = *pERC but safe */

    if (dlist_ins_prev(&listC, dlist_head(&listC), pERCin) != 0) {	/* //insert the first in &listC */
      printf
	  ("*** WARNING : coincidenceOutputModule.c : impossible to insert a new event in list\n");
      printf("exit\n");
      exit(0);
    }


  }




}


void outputCoincidenceModuleDestructor()
{
  headDone = FALSE;
}

LIST(*getListOfCoincidenceOutput())
{
  return (&listC);
}

void initializeListOfCoincidence()
{
  dlist_init(&listC, (void *) freeER);	/*  // init the list of coinci  */
}

void setCoincidenceOutputMode(u8 value)
{
  coincidenceOutputMode = value;
}

u8 getCoincidenceOutputMode()
{
  return (coincidenceOutputMode);
}


void closeOutputCoincidenceFile()
{
  if (pfC != NULL) {

    fclose(pfC);
    pfC = NULL;
    printf("coincidences file closed\n");
  }
}
