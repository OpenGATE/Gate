/*-------------------------------------------------------

           List Mode Format 
                        
     --  LMFbuilder.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of LMFbuilder.c:
     Write a record in LMF .ccs binary file at each call.
     At the first call, manage the opening of the file
     and the writting of the head of file. 
     The lines with fflush can be uncomment, but 
     it increases the writting time.


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


/* -=-=-=-=-=-=  FUNCTION LMF BUILDER -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
void LMFbuilder(const ENCODING_HEADER * pEncoH,
		const EVENT_HEADER * pEH,
		const COUNT_RATE_HEADER * pCRH,
		const GATE_DIGI_HEADER * pGDH,
		const CURRENT_CONTENT * pcC,
		const EVENT_RECORD * pER,
		const COUNT_RATE_RECORD * pCRR,
		FILE ** ppfile, const i8 * nameOfFile)
{

/*- - - - - - - - - - -  -=-=-=-=-=-=-=--=-=-=-=- - - - - - - - - - - - - - - */
/*                       Write the .ccs file HEAD                             */
/*- - - - - - - - - - -  -=-=-=-=-=-=-=--=-=-=-=- - - - - - - - - - - - - - - */

  if (*ppfile == NULL) {
    *ppfile = fopen(nameOfFile, "w+b");	/*create and Opening  WRITE_FILE */


    buildHead(pEncoH, pEH, pGDH, pCRH, *ppfile);	/* build and write head of .ccs file */

    fflush(*ppfile);

    fclose(*ppfile);		/* close the file in read write mode and... */
    *ppfile = fopen(nameOfFile, "a+b");	/*...open it  in "apend  mode"   */
  }



  /* */
 /*- - - - - - - - - - -  -=-=-=-=-=-=-=--=-=-=-=- - - - - - - - - - - - - - - */
  /*                       Write the .ccs file BODY                             */
 /*- - - - - - - - - - -  -=-=-=-=-=-=-=--=-=-=-=- - - - - - - - - - - - - - - */
  if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {	/* */
    buildER(pEH, pEncoH->scanEncodingIDLength, pER, *ppfile);	/*build and write  an event record */
    /* if (fflush(*ppfile) != 0) *//*force the writting of the record */
    /*        printf("\n*** ERROR : LMFBuilder : problem with fflush\n"); */


    if (pEH->gateDigiBool) {
      buildGDR(pGDH, pER->pGDR, pEH, *ppfile);	/*build and write a gate digi record */
      /*   if (fflush(*ppfile) != 0) */
      /*force the writting of the record */
      /*     printf("\n*** ERROR : LMFBuilder : problem with fflush\n");  */
    }
  }
  if (pcC->typeOfCarrier == pEncoH->scanContent.countRateRecordTag) {
    buildCRR(pEncoH, pCRH, pCRR, *ppfile);	/*build and write a  count rate record */
    /* if (fflush(*ppfile) != 0) */
    /*force the writting of the record */
    /*        printf("\n*** ERROR : LMFBuilder : problem with fflush\n");       */
  }
}				/* */

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-==-=-=-=-=-=-=-=-=-=*/


void CloseLMFfile(FILE * pfile)
{
  fclose(pfile);		/* Close the file */
}


/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-==-=-=-=-=-=-=-=-=-=*/
