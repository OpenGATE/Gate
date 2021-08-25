/*-------------------------------------------------------

           List Mode Format 
                        
     --  exempleMain_01.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/
/*------------------------------------------------------------------------
 
			   
	 Description : 
	 Explain how to use the libLMF.a tools, for the binary part of LMF.
         This example builds  an artificial test_ex1.ccs file. 

---------------------------------------------------------------------------*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../includes/lmf.h"

int main()
{

  u64 i = 0;

  /* definition of a LMF RECORD CARRIER and corresponding pointers            */
  struct LMF_ccs_encodingHeader *pEncoH = NULL;
  struct LMF_ccs_eventHeader *pEH = NULL;
  struct LMF_ccs_countRateHeader *pCRH = NULL;
  struct LMF_ccs_gateDigiHeader *pGDH = NULL;
  struct LMF_ccs_currentContent *pcC = NULL;
  struct LMF_ccs_eventRecord *pER = NULL;
  struct LMF_ccs_countRateRecord *pCRR = NULL;


  i16 numberOfRecords = 1, nRecordsWritten = 0;
  FILE *pf1 = NULL;		/* //,*pf2=NULL; */
  const i8 nameOfFile1[] = "test1_ex1.ccs";

  //  intro(); /* introducing message */
  helpExample1();
  /* -=-=-= generation of an LMF RECORD CARRIER -=-=-=-=-=-=-=-=-=-=-==-=-=-= */
  pcC = generatecC();		/* just an allocaton for a current content structure */
  pEncoH = generateEncoH(askTheMode());	/* generation of an encoding Header */
  if (pEncoH->scanContent.eventRecordBool == 1)	/* if Event Bool = 1 : */
    pEH = generateEH(pEncoH);	/* generation of an event Header */
  if (pEncoH->scanContent.countRateRecordBool == 1)	/* if CR Bool = 1 : */
    pCRH = generateCRH();	/* generation of a count rate Header */
  if (pEncoH->scanContent.gateDigiRecordBool == 1)	/* if GDR Bool = 1 : */
    pGDH = generateGDH();	/* generation of a gate digi  Header */




  /* =-=-= BUILD THE  LMF                       -=-=-=-=-=-=-=-=-=-=-==-=-=-= */

  printf("How many loops do you want to generate ? \n");
  printf("(x 10) \n");

  numberOfRecords = hardgeti16(1, 9);

  for (i = 0; i < 10 * numberOfRecords; i++) {
    printf(".");
    if (pEncoH->scanContent.eventRecordBool == 1) {	/* if Event Bool = 1 : */

      pER = generateER(pEncoH, pEH);	/* generation of an event Record */
      /*Be careful pER can be  NULL randomly */
      if ((pEH->gateDigiBool) && (pER)) {
	/* generation of an event record extension : gate digi record */
	pER->pGDR = generateGDR(pEncoH, pGDH);
      }

      pcC->typeOfCarrier = pEncoH->scanContent.eventRecordTag;
      if (pER) {
	/* This line is the magic line that adds the current event to the output file */
	LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf1, nameOfFile1);	/* ...write it */
	nRecordsWritten++;
      }
/* 	  // FOR A SECOND LMF */
/* 	  //	  if((i%2)==0) // write only half datas on the 2nd file */
      /* 	  //  pf2 = LMFbuilder(pEncoH,pEH,pCRH,pcC,pER,pCRR,pf2,nameOfFile2); *//* ...write it */
    }

    if (pEncoH->scanContent.countRateRecordBool == 1) {
      pCRR = generateCRR(pEncoH, pCRH);	/* generation of a count rate Record */
      pcC->typeOfCarrier = pEncoH->scanContent.countRateRecordTag;
      LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, &pf1, nameOfFile1);	/* ...write it */
      nRecordsWritten++;
      /*  // FOR A SECOND LMF */
      /* 	  // pf2 = LMFbuilder(pEncoH,pEH,pCRH,pcC,pER,pCRR,pf2,nameOfFile2);   *//* ...write it */
    }
  }


  printf("\n%s built (%llu loops done, %d records written)\n", nameOfFile1,
	 i, nRecordsWritten);
  if (pEncoH->scanContent.nRecord != 0) {

    CloseLMFfile(pf1);
/*        // FOR A SECOND LMF */
/*        // CloseLMFfile(pf2); */

    generatecCDestructor();
    generateERDestructor();
    generateCRRDestructor();
    generateGDRDestructor();
    generateEHDestructor();
    generateCRHDestructor();
    generateGDHDestructor();
    generateEncoHDestructor();
  }

  printf("\nYou have run successfully exampleMain1.c");

  printf("\nTry now exampleMain4.c to read and process ");
  printf("\nthe built file :  test1_ex1.ccs\n");

  return (EXIT_SUCCESS);

}
