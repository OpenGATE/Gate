/*-------------------------------------------------------

           List Mode Format 
                        
     --  findXYZinLMFfile.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of findXYZinLMFfile.c:


	 This function is the sister of LMFreader(). But LMFreader()
	 read all the ccs file and process evry record.
	 Here we just read, record, by record, and return the 
	 xyz position. This function is dedicated to the
	 STIR reconstruction.
-------------------------------------------------------*/





#include <netinet/in.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../includes/lmf.h"
#include <time.h>



static EVENT_HEADER *pEH = NULL;
static COUNT_RATE_HEADER *pCRH = NULL;
static GATE_DIGI_HEADER *pGDH = NULL;
static CURRENT_CONTENT *pcC = NULL;
static EVENT_RECORD *pER = NULL;
static COUNT_RATE_RECORD *pCRR = NULL;

static int doneOnce = FALSE;
static u8 *pBufEvent, *pBufGateDigi, *pBufCountRate;	/* buffer for read */
static u16 Epattern = 0, CRpattern = 0, GDpattern = 0, pattern =
    0, sizeGD = 0, sizeE = 0, sizeCR = 0, testread = 0;
static u8 uni8 = 0;
static int verboseLevel = 7;

calculOfEventPosition resultOfCalculOfEventPosition;

int findXYZinLMFfile(FILE * pfCCS,
		     double *x1,
		     double *y1,
		     double *z1,
		     double *x2,
		     double *y2, double *z2, ENCODING_HEADER * pEncoH)
{
  int returnValue;


  if (doneOnce == FALSE) {	/* // just done at first call */
    doneOnce = TRUE;

    init_findXYZinLMFfile(pfCCS, pEncoH);
    pER = newER(pEH);
    printf("Init done once .....\n");
  }



  /* -=-=-=-=-=-=-=-= READ ONE RECORD  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

  if ((testread = (fread(&uni8, sizeof(u8), 1, pfCCS))) == 1) {	/* read 1 byte */
    if ((uni8 & BIT8) == 0) {	/* TAG = 0 ; it s an event */
      pcC->typeOfCarrier = pEncoH->scanContent.eventRecordTag;
      fseek(pfCCS, -1, 1);	/* one byte before , then read the whole event record */

      testread = (fread(&pBufEvent[0], sizeof(u8), sizeE, pfCCS));
      if (testread == sizeE)	/* read the event record in once and fill the */
	readOneEventRecord(&pBufEvent[0], Epattern, pEH, pEncoH->scanEncodingIDLength, pER);	/* carrier */
      else			/* is there a problem of reading ? */
	printf("\nReading problem in LMFReader for an event record...");

      if (pEH->gateDigiBool) {	/* extended event record for Gate Digi information */
	testread = (fread(&pBufGateDigi[0], sizeof(u8), sizeGD, pfCCS));
	if (testread == sizeGD)	/* read the event record in once and fill the */
	  readOneGateDigiRecord(&pBufGateDigi[0], GDpattern, pGDH, pEH, pER);	/* carrier */
	else			/* is there a problem of reading ? */
	  printf
	      ("\nReading problem in LMFReader for a gate digi record...");
      }

    } else if ((uni8 & (~(BIT1 + BIT2 + BIT3 + BIT4))) == BIT8) {	/* TAG = 1000  COUNTRATE */

      pcC->typeOfCarrier = pEncoH->scanContent.countRateRecordTag;
      fseek(pfCCS, -1, 1);	/* one byte before then  read the whole Count Rate record */
      testread = (fread(&pBufCountRate[0], sizeof(u8), sizeCR, pfCCS));
      if (testread == sizeCR)	/* read the count rate record in once and fill the  */
	pCRR = readOneCountRateRecord(pEncoH, pBufCountRate, CRpattern);	/*carrier */
      else			/* is there a problem of reading ? */
	printf
	    ("\nReading problem in LMFReader for a count rate record...");
    }



    if (pcC->typeOfCarrier == pEncoH->scanContent.countRateRecordTag) {	/*        This record is a count rate ... no treated here ... */
      if (verboseLevel > 5)
	printf
	    ("a count rate record have been read read but not treated\n");

    } else if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {	/*        This record is an event ... treated here ... */

      resultOfCalculOfEventPosition =
	  locateEventInLaboratory(pEncoH, pER, 0);

      *x1 =
	  resultOfCalculOfEventPosition.eventInLaboratory3DPosition.radial;
      *y1 =
	  resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	  tangential;
      *z1 =
	  resultOfCalculOfEventPosition.eventInLaboratory3DPosition.axial;

      returnValue = 1;

      if (pEH->coincidenceBool == 1) {

	resultOfCalculOfEventPosition =
	    locateEventInLaboratory(pEncoH, pER, 1);


	*x2 =
	    resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	    radial;
	*y2 =
	    resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	    tangential;
	*z2 =
	    resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	    axial;
	returnValue = 2;

      }

    } else {
      if (verboseLevel)
	printf("an unknown record have been read in .ccs file\n");
    }
    return (returnValue);
  } else
    return (0);			/* // end of file */

}



void init_findXYZinLMFfile(FILE * pfCCS, ENCODING_HEADER * pEncoH)
{


  pcC = (CURRENT_CONTENT *) malloc(sizeof(CURRENT_CONTENT));
  if (!pcC)
    printf("*** error  : read_LMF.c : malloc impossible for pcC");


  printf("pEncoH nrecord = %d\n", pEncoH->scanContent.nRecord);
  if (pEncoH->scanContent.nRecord != 0) {	/* if there are records in the file...  */
    fseek(pfCCS, -(2 * pEncoH->scanContent.nRecord), 1);	/* seek the encoding patterns  */
    if (pEncoH->scanContent.eventRecordBool == 1) {	/* is there an E pattern ?  */

      if (verboseLevel)
	printf("Event Record in this file\n");
      fread(&pattern, sizeof(u16), 1, pfCCS);	/*read a i16  */
      pattern = ntohs(pattern);
      Epattern = pattern;	/* Buffer the event pattern */
      pEH = extractEpat(Epattern);	/*analyse the event pattern  */
      if (verboseLevel) {
	if (!pEH->detectorIDBool)
	  printf("warning : no crystal ID in this file\n");
	else
	  printf("crystal ID in this file\n");
      }

      if (verboseLevel) {
	if (!pEH->coincidenceBool)
	  printf("warning : no coincidence in this file\n");
	else
	  printf("coincidences in this file\n");
      }


      sizeE = calculatesizeevent(Epattern);	/* size of 1 Event Record */
      sizeE += 2 * (pEncoH->scanEncodingIDLength);
      pBufEvent = malloc(sizeE * sizeof(u8));

      if (pEncoH->scanContent.gateDigiRecordBool == 1) {

	if (verboseLevel)
	  printf("GateDigi Record in this file ");
	fread(&pattern, sizeof(u16), 1, pfCCS);	/*read a i16  */
	pattern = ntohs(pattern);
	GDpattern = pattern;	/* Buffer the gate digi pattern */
	pGDH = extractGDpat(GDpattern);	/*analyse the gate digi pattern  */


	sizeGD = calculatesizegatedigi(GDpattern, Epattern);	/* size of 1 Event Record */
	printf("size GDR = %d\n", sizeGD);
	pBufGateDigi = malloc(sizeGD * sizeof(u8));
      }






    }
    if (pEncoH->scanContent.countRateRecordBool == 1) {	/* and a CR pattern ?  */

      if (verboseLevel)
	printf("Count rate Record in this file\n");

      fread(&pattern, sizeof(u16), 1, pfCCS);	/*        so read it */
      pattern = ntohs(pattern);	/* byte order  */
      CRpattern = pattern;
      pCRH = extractCRpat(CRpattern);	/*and analyse it */

      sizeCR = calculatesizecountrate(CRpattern, pEncoH);	/* size of 1 CR Record */
      pBufCountRate = malloc(sizeCR * sizeof(u8));

    }
  }
}


FILE *open_CCS_file2(const i8 * nameOfFileCCS)
{
  FILE *pfCCS;
  pfCCS = fopen(nameOfFileCCS, "rb");	/* // LMF binary file */

  if (!pfCCS) {
    printf("*** exit : findXYZinLMFfile.c : impossible to find %s\n",
	   nameOfFileCCS);
    exit(0);
  } else if (verboseLevel)
    printf(".ccs open\n");


  return (pfCCS);
}

void destroy_findXYZinLMFfile(ENCODING_HEADER * pEncoH)
{


  if (pcC) {
    if (verboseLevel)
      printf("free pcC\n");
    free(pcC);
  }
  if (pBufEvent) {
    if (verboseLevel)
      printf("free pBufEvent\n");
    free(pBufEvent);
  }
  if (pBufCountRate) {
    if (verboseLevel)
      printf("free pBufCountRate\n");
    free(pBufCountRate);
  }

  doneOnce = FALSE;

}
