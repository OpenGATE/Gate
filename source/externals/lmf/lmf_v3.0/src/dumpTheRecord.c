/*-------------------------------------------------------

           List Mode Format 
                        
     --  dumpTheRecord.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dumpTheRecord.c:
     Called for each record read by LMFreader() in mode
     dump
     It displays on screen the containing of a .ccs file


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include "lmf.h"

void dumpTheRecord(const ENCODING_HEADER * pEncoH,
		   const EVENT_HEADER * pEH,
		   const COUNT_RATE_HEADER * pCRH,
		   const GATE_DIGI_HEADER * pGDH,
		   const CURRENT_CONTENT * pcC,
		   const EVENT_RECORD * pER,
		   const COUNT_RATE_RECORD * pCRR)
{

  static int headalreadyDump = FALSE;
  static u64 recordNumber = 1, eventRecordNumber =
      1, countRateRecordNumber = 1;

  if (headalreadyDump == FALSE) {
    printf("\n");
    printf("****************************\n");
    printf("* DUMP THE LMF BINARY FILE *\n");
    printf("****************************\n");
    printf("\n");
    printf("******     HEAD      *******\n");
    printf("\n");

    dumpHead(pEncoH, pEH, pGDH, pCRH);
    headalreadyDump = TRUE;
    printf("\n");
    printf("******    BODY      ********\n");
    printf("\n");
  }

  //if(pER->pGDR->eventID[1] == pER->pGDR->eventID[0])
  //{ 
  printf("--> RECORD %llu : ", recordNumber);
  /* -=-=-=-=-=-=-=-= DUMP THE BODY  */
  if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {
    recordNumber++;
    eventRecordNumber++;
    printf("\n*********** Event Record *********** \n");
    dumpEventRecord(pEncoH, pEH, pER);
    if (pEH->gateDigiBool)
      dumpGateDigiRecord(pEncoH, pEH, pGDH, pER->pGDR);


    getchar();
  }
  if (pcC->typeOfCarrier == pEncoH->scanContent.countRateRecordTag) {
    recordNumber++;
    countRateRecordNumber++;

    printf("\n*********** Count Rate Record *************\n");
    dumpCountRateRecord(pEncoH, pCRH, pCRR);
    getchar();
  }
  //}

}
