/*-------------------------------------------------------

           List Mode Format 
                        
     --  exempleMain_06.c  --                      

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
         A more advanced exemple of read & treat LMF files. We read a LMF file of singles,
	 treat the singles (with cuts for exemple), send them 
	 to coincidence sorter, that returns a (pointer to) a list of coincidences, 
	 for each single (of course this list is empty most of time).
	 The outputCoincidence function accept 0 or 1 as its 
	 last parameter  : coincidence storage mode. 
	 0 means 
	 "write this coincidence in a file" like it is explicitely done here 
	 after the coincidence treatment. 
	 1 means "push them on 
	 a stack, like is done here in a transparent way in the coincidence sorter 
	 after  we have done 
	 setCoincidenceStorageMode(1);
	 This exemple uses the 2 modes of output coincidences :
	 - pushing on a stack and then if they are accepted
	 - writing in a file
	 
	 It details what the very transparent LMFreader()
	 function does. But if you want a to optimize the speed
	 of sorting, i recommand the LMFreader() (see exempleMain_04)
	 to sort the coincidences.
	 
	 The coincidence output file is always  : 
	 <name of singles file>_coinci.ccs
	 
	 I strongly recommand the reading of 
	 "Software Design" : 
	 http://www-iphe.unil.ch/~PET/CCC/memos/ClearPET_software_design.pdf

---------------------------------------------------------------------------*/
#include <netinet/in.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../includes/lmf.h"
#include <time.h>


int main()
{

/* definition of a LMF RECORD CARRIER and corresponding pointers  */
  static struct LMF_ccs_encodingHeader *pEncoH, *pEncoHC;	/* encoding header structure pointer */
  static struct LMF_ccs_eventHeader *pEH, *pEHC;	/* event header structure pointer */
  static struct LMF_ccs_countRateHeader *pCRH, *pCRHC;	/* count rate header structure pointer */
  static struct LMF_ccs_gateDigiHeader *pGDH, *pGDHC;	/* gate digi header structure pointer */
  static struct LMF_ccs_currentContent *pcC, *pcCC;	/* current content structure pointer */
  static struct LMF_ccs_eventRecord *pER, *pERC;	/* event record structure pointer */
  static struct LMF_ccs_countRateRecord *pCRR;	/* count rate record structure pointer */
  static u8 *pBufEvent, *pBufCountRate, *pBufGateDigi;	/* buffer for read blocks of bytes */
  FILE *pf = NULL;		/* singles record file pointer  */
  u16 Epattern = 0, CRpattern = 0, GDpattern, pattern = 0;	/* to read record headers */
  u16 sizeE = 0, sizeCR = 0, sizeGD = 0;	/* size of different records */
  u16 testread = 0;
  u16 *pcrist;			/*to extract crystal IDs */
  u8 uni8 = 0;
  void *data;			/* to destroy the coincidence list member correctly */
  i8 keepIT = FALSE, keepITC = FALSE;	/* to specify if we keep the singles 
					   (resp. the coincidences) */


/*******************************************************************/
/*                                                                 */
/*                    START                                        */
/*                                                                 */
/*******************************************************************/

  printf("\n****************************************\n");
  printf("                EXEMPLE 6                ");
  printf("\n****************************************\n\n\n");
  system("ls *.cch *.ccs");	/* displays on screen the LMF files in current dir */


 /***************************************************/
  /*                                                 */
  /*     1.  ASCII PART : .cch FILE                  */
  /*                                                 */
 /***************************************************/

  /* 
     This next line asks to user to enter the file of singles he wants to read 
     without extension. Then  read file.cch and fill in structures LMF_cch 
     It also creates the different file names we need (as extern variables)
   */
  if (LMFcchReader(""))
    exit(EXIT_FAILURE);

 /***************************************************/
  /*                                                 */
  /*     2.  BINARY PART : .ccs FILE                 */
  /*                                                 */
 /***************************************************/

 /***********    open the file .ccs     ****************************/
  pf = fopen(ccsFileName, "rb");	/* ccsFileName is an extern, open it in binary mode */
  if (pf == NULL) {		/* test if the file exists */
    printf("\nFile %s not found by the LMFreader...\n", ccsFileName);
    exit(0);
  }

  /* 
     local allocation for the current content structure. You must 
     free(pcC) at the end.
   */
  pcC = (struct LMF_ccs_currentContent *)
      malloc(sizeof(struct LMF_ccs_currentContent));
  if (!pcC) {
    printf
	("*** ERROR : exempleMain_06.c : impossible to malloc for pcC\n");
    exit(0);
  }


  /***************************************************/
  /*                                                 */
  /*       READ HEAD OF .ccs FILE                    */
  /*                                                 */
  /***************************************************/


 /***************   first part of head : the encoding header ***/

  fseek(pf, 0L, 0);		/* seek the begin of file */

  /*
     fill the encoding header structure. Be careful : the readhead function
     is doing the allocation itself for an encoding header structure.
     But it is your responsability to free it at the end with destroyReadHead().
     The function read the first bytes of the .ccs binary file, 
     extracts the information and fill pEncoH. 
   */
  pEncoH = readHead(pf);
  printf("\nhead read ok...\n");

  /***************   second part of head : the record headers ***/


  /* -=-=-== READ THE ENCODING PATTERNS AND FILL THE RECORD HEADERS  =-=-=-=-    */
  if (pEncoH->scanContent.nRecord != 0) {	/* if there are records in the file...     */
    fseek(pf, 10L, 0);		/* seek the encoding patterns. step forward 10 bytes */


    if (pEncoH->scanContent.eventRecordBool == 1) {	/* if there an event pattern */
      fread(&pattern, sizeof(u16), 1, pf);	/* read the pattern i16 */
      pattern = ntohs(pattern);	/* network to host conversion    */
      Epattern = pattern;	/* store the event pattern */
      /*
         After use extractEpat, that allocates and fill
         the event header structure (pEH)
         it is your responsability to destroyExtractEpat()
         at the end.
       */
      pEH = extractEpat(Epattern);	/* analyse the event pattern and fill pEH */

      if (pEH->coincidenceBool == 1) {	/*check if it is a singles file */
	printf("This exemple needs a singles file\n");
	printf("No coincidence file accepted !!!\n");
	exit(0);
      }
      /* 
         calculatesizeevent() calculate the size (bytes) of an event record  
         This size depends of the event header (pEH). If the size of 
         a record is 15 bytes we must read the .ccs file block by block
         (of 15 bytes)
       */
      sizeE = calculatesizeevent(Epattern);	/* size of 1 Event Record */
      pBufEvent = malloc(sizeE * sizeof(u8));	/* free it at end */
    }

    if (pEncoH->scanContent.gateDigiRecordBool == 1) {	/* if there an gateDigi pattern */
      fread(&pattern, sizeof(u16), 1, pf);	/* read the pattern i16 */
      pattern = ntohs(pattern);	/* network to host byte order conversion    */
      GDpattern = pattern;	/* store the gateDigi pattern */
      /*
         After use extractGDpat, that allocates and fill
         the gateDigi header structure (pGDH)
         it is your responsability to destroyExtractGDpat()
         at the end.
       */
      pGDH = extractGDpat(GDpattern);	/* analyse the gateDigi pattern and fill pGDH */
      /* 
         calculatesizegateDigi() calculate the size (bytes) of a gateDigi record  
         This size depends of the gateDigi header (pGDH). If the size of 
         a record is, for ex., 15 bytes we must read the .ccs file block by block
         (of 15 bytes)
       */
      sizeGD = calculatesizegatedigi(GDpattern, Epattern);	/* size of 1 GateDigi Record */
      pBufGateDigi = malloc(sizeGD * sizeof(u8));	/* free it at end */
    }

    if (pEncoH->scanContent.countRateRecordBool == 1) {	/* and a CR pattern ?  */
      fread(&pattern, sizeof(u16), 1, pf);	/*        so read it      */
      pattern = ntohs(pattern);	/* byte order : network to host  */
      CRpattern = pattern;
      /*
         After use extractEpat, that allocates and fill
         the count rate header structure (pCRH)
         It is your responsability to destroyExtractCRpat()
         at the end.
       */
      pCRH = extractCRpat(CRpattern);	/*and analyse it */
      sizeCR = calculatesizecountrate(CRpattern, pEncoH);	/* size of 1 CR Record */
      pBufCountRate = malloc(sizeCR * sizeof(u8));	/* free it at end */
    }
  } else {			/*  no records in file ---> exit  */

    printf("no records in file\n");
    exit(0);
  }


  /***************************************************/
  /*                                                 */
  /*       READ BODY OF .ccs FILE                    */
  /*                                                 */
  /***************************************************/


  /*
     here we allocate a coincidence record carrierjust for the coincidences
     Dont forget to free them !!!
   */
  pEncoHC = (struct LMF_ccs_encodingHeader *)
      malloc(sizeof(struct LMF_ccs_encodingHeader));
  pEHC = (struct LMF_ccs_eventHeader *)
      malloc(sizeof(struct LMF_ccs_eventHeader));
  pGDHC = (struct LMF_ccs_gateDigiHeader *)
      malloc(sizeof(struct LMF_ccs_gateDigiHeader));
  pCRHC = (struct LMF_ccs_countRateHeader *)
      malloc(sizeof(struct LMF_ccs_countRateHeader));
  pcCC = (struct LMF_ccs_currentContent *)
      malloc(sizeof(struct LMF_ccs_currentContent));
  if ((pEncoHC == NULL) || (pEHC == NULL) || (pGDHC == NULL)
      || (pCRHC == NULL))
    printf("\n *** error : exemple_06 : malloc\n");



  /*
     copy the singles record carrier in this new coincidence record
     carrier
   */
  if (pEncoH)
    *pEncoHC = *pEncoH;		/* // no pointer in this structure, it is safe */
  if (pEH)
    *pEHC = *pEH;		/* // no pointer in this structure, it is safe */
  if (pCRH)
    *pCRHC = *pCRH;		/* // no pointer in this structure, it is safe */
  if (pGDH)
    *pGDHC = *pGDH;		/*  // no pointer in this structure, it is safe */

  pEHC->coincidenceBool = 1;	/* // specify that it's a coincidence Record carrier */


  /*
     Set the coincidence storage mode to 1
     1 means that we do not store the coincidence in a file (it would be 0)
     but we return them as a list to be treated.
   */
  setCoincidenceOutputMode(1);

  initializeListOfCoincidence();	/*  // as you can understand .. */
  /*
     Create the name of output file for coincidence : xxx_coinci.ccs
   */
  get_extension_ccs_FileName("_coinci");
  /*  ********************************************
   *                MAIN LOOP                    *
   *  ********************************************/

  /*
     This while loop read the records one by one and
     continues to the end of file.
     It test (with the tag) what is the next event and then 
     read a corresponding block of bytes. The block size is 
     sizeE for an event record and sizeCR for a count rate
     The event records fill pER (event record structure pointer)
     The count rate records fill pCRR (count rate record structure pointer)
     It is your responsability to call the 2 destructors at the end :
     destroyGDRreader() for readOneGateDigiRecord,
     destroyERreader() for readOneEventRecord and
     destroyCRRreader() for readOneCountRateRecord.
   */

  pER = newER(pEH);
  while ((testread = (fread(&uni8, sizeof(u8), 1, pf))) == 1) {	/* read 1 byte */
    if ((uni8 & BIT8) == 0) {	/* TAG = 0 ; it s an event */
      /*
         we specify that this carrier contains now an event record
       */
      pcC->typeOfCarrier = pEncoH->scanContent.eventRecordTag;
      fseek(pf, -1, 1);		/* one byte before , then read the whole event record */
      testread = (fread(&pBufEvent[0], sizeof(u8), sizeE, pf));
      if (testread == sizeE)	/* read the event record in once and fill the */
	readOneEventRecord(&pBufEvent[0], Epattern, pEH, pEncoH->scanEncodingIDLength, pER);	/* carrier */
      else			/* is there a problem of reading ? */
	printf("\nReading problem in LMFReader for an event record...");

      if (pEH->gateDigiBool) {	/* extended event record for Gate Digi information */
	testread = (fread(&pBufGateDigi[0], sizeof(u8), sizeGD, pf));

	if (testread == sizeGD)	/* read the event record in once and fill the */
	  readOneGateDigiRecord(&pBufGateDigi[0], GDpattern, pGDH, pEH, pER);	/* carrier */
	else			/* is there a problem of reading ? */
	  printf
	      ("\nReading problem in LMFReader for a gate digi record...");
      }
    } else if ((uni8 & (~(BIT1 + BIT2 + BIT3 + BIT4))) == BIT8) {	/* TAG = 1000  COUNTRATE */
      pcC->typeOfCarrier = pEncoH->scanContent.countRateRecordTag;
      fseek(pf, -1, 1);		/* one byte before then  read the whole Count Rate record */
      testread = (fread(&pBufCountRate[0], sizeof(u8), sizeCR, pf));
      if (testread == sizeCR)	/* read the count rate record in once and fill the  */
	pCRR = readOneCountRateRecord(pEncoH, pBufCountRate, CRpattern);	/*carrier */
      else			/* is there a problem of reading ? */
	printf
	    ("\nReading problem in LMFReader for a count rate record...");
    }



	  /************************************************


                  T R E A T   T H E   S I N G L E S


	  ************************************************/


	  /**********************************
            THE RECORD IS AN EVENT RECORD ? 
	   *********************************/
    if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {

	      /************************************

                   Type here your treatment for 
                   the single event records.
                   If you want to ignore this 
	           record and get a new one
                   just type in your "if block" 
                   pER = NULL;
                   
                   This exemple shows how to cut 
                   the low energy singles.
	      **************************************/

      /* if energy  < 100 keV */
      if ((pER->energy[0] * GATE_LMF_ENERGY_STEP_KEV) < 100)
	keepIT = FALSE;
      else
	keepIT = TRUE;
    } else {
      /*
         else nothing, we just treat the event
         records in this exemple, not the 
         count rate records !!!
       */
    }


	  /************************************************


             T R E A T   T H E   C O I N C I D E N C E S


	  ************************************************/

    if (keepIT) {		/*  // if the singles is accepted */
      if (pcC->typeOfCarrier == pEncoH->scanContent.eventRecordTag) {
/* 		  // send it to the coincidence sorter */
/* 		  // in mode 1 : push on a list the coincidence */
	sortCoincidence(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR);
      }


      /*
         now we manage this list of coincidence
         and destroy all its members...
         Most of time, this list is empty. Because you don't 
         have a coincidence for each singles you treat. 
         But if you have one coincidence, you can also have two 
         or more, because of the design of the coincidence sorter.

       */
      while ((getListOfCoincidenceOutput())->size != 0) {

	pERC = dlist_head(getListOfCoincidenceOutput())->data;	/*  // take coincidence */
		  /************************************

                   Type here your treatment for 
                   the coincidence event records.
                   If you want to ignore this 
	           record and get a new one
                   just type in your "if block" 
                   keepITC = FALSE;
                   
                   This exemple shows how to cut 
                   the coincidence that happen
                   in layer 0 (internal)
		  **************************************/


	/*
	   This function "demakes" the crystal ID
	   and gives back an allocated pointer on 5 i16
	   It's your responsability to free(pcrist);
	   just after use.
	   Return values : 
	   pcrist[0] = layer ID
	   pcrist[1] = crystal ID
	   pcrist[2] = submodule ID
	   pcrist[3] = module ID
	   pcrist[4] = rsector ID
	 */
	pcrist = demakeid(pERC->crystalIDs[0], pEncoHC);
	if (pcrist[0] == 0) {

	  keepITC = FALSE;
	} else {

	  keepITC = TRUE;
	}
	free(pcrist);




		  /**************************************/
	if (keepITC)
	  outputCoincidence(pEncoHC, pEHC, pGDHC, pcCC, pERC, 0);
/* // 0 is the mode to write this coinci in a file */


	/* THIS CURIOUS LINES DESTROY SAFELY ONE MEMBER OF THE LIST */
	if (((dlist_remove(getListOfCoincidenceOutput(),
			   dlist_head(getListOfCoincidenceOutput()),
			   (void **) &data)) == 0)
	    && (getListOfCoincidenceOutput()->destroy != NULL)) {
	  getListOfCoincidenceOutput()->destroy(data);
	}
      }
    }

  }






  /***************************************************/
  /*                                                 */
  /*                 END                             */
  /*                                                 */
  /***************************************************/

  if (pEncoH->scanContent.eventRecordBool) {
    destroyList();		/*  // in _coincidenceSorter.c */
    destroyCoinciFile();	/* // in _cleanKit.c */
    outputCoincidenceModuleDestructor();
  } else
    printf("no event records to sort in this file");


  if (pEncoH->scanContent.gateDigiRecordBool == 1) {

    destroyExtractGDpat();	/* //      free pGDH */
    free(pBufGateDigi);
  }

  if (pEncoH->scanContent.eventRecordBool == 1) {

    destroyExtractEpat();	/* //      free pEH */
    free(pBufEvent);
  }
  if (pEncoH->scanContent.countRateRecordBool == 1) {
    /* this 2 lines cannot be inverted */
    destroyCRRreader(pCRH);
    destroyExtractCRpat();
    free(pBufCountRate);
  }

  free(pcC);
  free(pcCC);
  free(pEncoHC);
  free(pEHC);
  free(pGDHC);
  free(pCRHC);

  destroyReadHead();		/* destructor of pEncoH */
  fclose(pf);			/* close the binary file */

  /*-=-=-=--=-=-=--=-=-=-==----=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  printf("Main over\n");
  return (EXIT_SUCCESS);
}
