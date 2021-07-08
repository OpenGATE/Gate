/*-------------------------------------------------------

List Mode Format 
                        
--  LMFCbuilder.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of LMFCbuilder.c:
This file is a copy of LMFbuilder.c but designed,
and optimized for the coincidence output module.


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"



/* -=-=-=-=-=-=  FUNCTION LMF BUILDER -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

void LMFCbuilder(const ENCODING_HEADER * pEncoH,
		 const EVENT_HEADER * pEH,
		 const GATE_DIGI_HEADER * pGDH,
		 const CURRENT_CONTENT * pcC,
		 const EVENT_RECORD * pER,
		 FILE ** ppfile, const i8 * nameOfFile)
{
  /*- - - - - - - - - - -  -=-=-=-=-=-=-=--=-=-=-=- - - - - - - - - - - - - - - */
  /*                       Write the .ccs file HEAD                             */
  /*- - - - - - - - - - -  -=-=-=-=-=-=-=--=-=-=-=- - - - - - - - - - - - - - - */
  /* This if-block is just done at the 1st call */
  if (*ppfile == NULL) {
    *ppfile = fopen(nameOfFile, "w+b");	/*create and Opening  WRITE_FILE */


    buildHead(pEncoH, pEH, pGDH, NULL, *ppfile);	/* build and write head of .ccs file */

    fflush(*ppfile);
    /* the head is not to build anymore  */
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
    /*        printf("\n*** ERROR : LMFBuilder : problem with fflush\n");        
     */
    if (pEH->gateDigiBool) {
      buildGDR(pGDH, pER->pGDR, pEH, *ppfile);	/*build and write a gate digi record */
      /* if (fflush(*ppfile) != 0)  *//*force the writting of the record */
      /*    printf("\n*** ERROR : LMFBuilder : problem with fflush\n"); *//* */
    }
  }



  if (pcC->typeOfCarrier == pEncoH->scanContent.countRateRecordTag) {	/* */
    printf("*** ERROR : LMFCbuilder.c : find a count rate here !\n");
  }

  return;
}				/* */

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-==-=-=-=-=-=-=-=-=-=*/


void CloseLMFCfile(FILE * pfile)
{				/* Close the file */
  if (pfile)
    fclose(pfile);


}


/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-==-=-=-=-=-=-=-=-=-=*/


void FreeLMFCBuilderCarrier(ENCODING_HEADER * pEncoH,
			    EVENT_HEADER * pEH,
			    GATE_DIGI_HEADER * pGDH,
			    CURRENT_CONTENT * pcC, EVENT_RECORD * pER)
{
  if (pEncoH->scanContent.eventRecordBool == 1) {
    free(pER->crystalIDs);
    free(pER->energy);
    free(pEH);
    if (pEncoH->scanContent.gateDigiRecordBool == 1) {
      if (pGDH)
	free(pGDH);
      if (pER->pGDR)
	free(pER->pGDR);
    }

    free(pER);

  }


  if (pcC)
    free(pcC);
  if (pEncoH)
    free(pEncoH);
  printf("coincidence LMF file built succesfully by LMFCbuilder...ok\n");
}
