/*-------------------------------------------------------

           List Mode Format 
                        
     --  treatEventRecord.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of keepOnlyTrue.c:

	 If you want to treat a data of the event record
	 (exemple, to "cut" energy) you can treat them here
	 before to copy it with the "treat and copy" option
	 of the LMF reader.

-------------------------------------------------------*/

#include <stdio.h>

#include "lmf.h"

// ,

EVENT_RECORD *keepOnlyTrue(const ENCODING_HEADER * pEncoH,
			   const EVENT_HEADER * pEH,
			   const GATE_DIGI_HEADER * pGDH,
			   EVENT_RECORD * pER)
{
  int keepIT = TRUE;		/*  = FALSE if we dont keep this event */
  static int doneOnce = FALSE;
  static int fileOK = TRUE;	/*  the file is a coincidence file ? */

  keepIT = TRUE;

  if (doneOnce == FALSE) {
    if (pEncoH->scanContent.eventRecordBool == FALSE) {
      printf("*** error : keepOnlyTrue.c : not an event record file\n");
      fileOK = FALSE;
      printf("<ENTER> to continue\n");
      getchar();

    }

    if (pEH->coincidenceBool == FALSE) {
      printf("*** error : keepOnlyTrue.c : not a coincidence file\n");
      fileOK = FALSE;
      printf("<ENTER> to continue\n");
      getchar();
    }
    if (pEncoH->scanContent.gateDigiRecordBool == FALSE) {
      printf("*** warning : keepOnlyTrue.c : no gate digi in this file\n");
      fileOK = FALSE;
      printf("<ENTER> to continue\n");
      getchar();
    }
    if (pGDH->comptonBool == FALSE) {
      printf
	  ("*** warning : keepOnlyTrue.c : no number of compton in this file\n");
      fileOK = FALSE;
      printf("<ENTER> to continue\n");
      getchar();
    }
    if (pGDH->eventIDBool == FALSE) {
      printf("*** warning : keepOnlyTrue.c : no event ID  in this file\n");
      fileOK = FALSE;
      printf("<ENTER> to continue\n");
      getchar();
    }
    doneOnce = TRUE;
  }


  if (pER->pGDR->eventID[0] != pER->pGDR->eventID[1]) {
    keepIT = FALSE;
    // printf("%d\n",pER->pGDR->eventID[0]);
    //getchar();

  } else if (pER->pGDR->numberCompton[0] || pER->pGDR->numberCompton[1]) {
    keepIT = FALSE;

  } else {
    keepIT = TRUE;


  }



  if (keepIT)
    return (pER);
  else
    return (NULL);


}
