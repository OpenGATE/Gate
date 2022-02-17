/*-------------------------------------------------------

List Mode Format 
                        
--  readHead.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of readHead.c:
this function read the head of a .ccs file and fill with the results a
encoding header structure dynamically allocated (3 substructures) and
return a pointer on this sturcture. Warning : it doesn't read the
encoding patterns...

Ex :
pEncoH = readHead(pf); 

-------------------------------------------------------*/
#include <stdio.h>		/*for printf... */
#include <stdlib.h>		/*for malloc */
#include <netinet/in.h>		/*for byte order: ntohs & htons */
#include "lmf.h"


static ENCODING_HEADER *pEncoH;
static int allocEncoHdone = FALSE;
static u8 i = 0;
void destroyReadHead()
{
  if (pEncoH)
    free(pEncoH);
  allocEncoHdone = FALSE;
}


/* // this function returns 1 if pat started by Event Record Tag : 0000 xxxx xxxx xxxx */
i8 checkEventTag(u16 pat)
{
  i8 yesItIs;

  if ((pat & (BIT16 + BIT15 + BIT14 + BIT13)) != 0)	/* // if pat & 1111 0000 0000 0000 != 0 */
    yesItIs = FALSE;
  else
    yesItIs = TRUE;


  return (yesItIs);
}

/* // this function returns 1 if pat started by Count rate Record Tag : 1000 xxxx xxxx xxxx  */
i8 checkCountRateTag(u16 pat)
{
  i8 yesItIs;

  if ((pat & (BIT15 + BIT14 + BIT13)) != 0)
    yesItIs = FALSE;
  else
    yesItIs = TRUE;

  if ((pat & BIT16) != BIT16)
    yesItIs = FALSE;

  return (yesItIs);

}

/* // this function returns 1 if pat started by gate digi Record Tag : 1100 xxxx xxxx xxxx */
i8 checkGateDigiTag(u16 pat)
{

  i8 yesItIs = FALSE;

  if ((pat & (BIT14 + BIT13)) != 0)
    yesItIs = FALSE;
  else
    yesItIs = TRUE;

  if ((pat & (BIT16 + BIT15)) != (BIT16 + BIT15))
    yesItIs = FALSE;

  return (yesItIs);



}


ENCODING_HEADER *readHead(FILE * pf)
{
  ENCODING_HEADER *pEncoH;

  u16vsu8 buffID16;		/*buffer for an u16 */
  u32vsu8 buffID32;		/*buffer for an u32 */
  u64vsu8 buffID64;		/*buffer for an u64 */

  u8 *buffer;

  u16 checkPattern;

  u16 *topo;
  u64 generalID, tangentialID, axialID;

  checkPattern = putBits(14) << 2;

  if ((pEncoH = malloc(sizeof(ENCODING_HEADER))) == NULL)
    printf("\n***ERROR : in readHead.c : impossible to do : malloc()\n");

  if ((fread(buffID16.w8, sizeof(u8), 2, pf)) != 2)
    printf("\n***ERROR : in readHead.c : impossible to do : fread()\n");

  /* Check if it's a lmf file v.2 (if yes) or v.1 */
  if (((buffID16.w16 >> 2) << 2) == checkPattern) {
    switch (buffID16.w16 - ((buffID16.w16 >> 2) << 2)) {
    case 0:
      pEncoH->scanEncodingIDLength = 0;
      if ((fread(buffID16.w8, sizeof(u8), 2, pf)) != 2)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      demakeRule(pEncoH, buffID16.w16);

      /* read the general toppology */
      if ((fread(buffID16.w8, sizeof(u8), 2, pf)) != 2)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      generalID = buffID16.w16;
      /* read the tangential topology */
      if ((fread(buffID16.w8, sizeof(u8), 2, pf)) != 2)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      tangentialID = buffID16.w16;
      /* debuild the axial topology */
      if ((fread(buffID16.w8, sizeof(u8), 2, pf)) != 2)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      axialID = buffID16.w16;

      break;
    case 1:
      pEncoH->scanEncodingIDLength = 1;
      if ((fread(buffID32.w8, sizeof(u8), 4, pf)) != 4)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      demakeRule(pEncoH, buffID32.w32);

      /* read the general toppology */
      if ((fread(buffID32.w8, sizeof(u8), 4, pf)) != 4)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      generalID = buffID32.w32;
      /* read the tangential topology */
      if ((fread(buffID32.w8, sizeof(u8), 4, pf)) != 4)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      tangentialID = buffID32.w32;
      /* debuild the axial topology */
      if ((fread(buffID32.w8, sizeof(u8), 4, pf)) != 4)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      axialID = buffID32.w32;

      break;
    case 2:
      pEncoH->scanEncodingIDLength = 2;
      if ((fread(buffID64.w8, sizeof(u8), 8, pf)) != 8)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      printf("OK\n");
      demakeRule(pEncoH, buffID64.w64);
      printf("OK\n");

      /* read the general toppology */
      if ((fread(buffID64.w8, sizeof(u8), 8, pf)) != 8)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      generalID = buffID64.w64;
      /* read the tangential topology */
      if ((fread(buffID64.w8, sizeof(u8), 8, pf)) != 8)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      tangentialID = buffID64.w64;
      /* debuild the axial topology */
      if ((fread(buffID64.w8, sizeof(u8), 8, pf)) != 8)
	printf
	    ("\n***ERROR : in readHead.c : impossible to do : fread()\n");
      axialID = buffID64.w64;

      break;
    }
  } else {
    pEncoH->scanEncodingIDLength = 0;
    buffID16.w16 = ntohs(buffID16.w16);
    demakeRule(pEncoH, buffID16.w16);

    /* read the general toppology */
    if ((fread(buffID16.w8, sizeof(u8), 2, pf)) != 2)
      printf("\n***ERROR : in readHead.c : impossible to do : fread()\n");
    buffID16.w16 = ntohs(buffID16.w16);
    generalID = buffID16.w16;
    /* read the tangential topology */
    if ((fread(buffID16.w8, sizeof(u8), 2, pf)) != 2)
      printf("\n***ERROR : in readHead.c : impossible to do : fread()\n");
    buffID16.w16 = ntohs(buffID16.w16);
    tangentialID = buffID16.w16;
    /* debuild the axial topology */
    if ((fread(buffID16.w8, sizeof(u8), 2, pf)) != 2)
      printf("\n***ERROR : in readHead.c : impossible to do : fread()\n");
    buffID16.w16 = ntohs(buffID16.w16);
    axialID = buffID16.w16;
  }

  topo = demakeid(generalID, pEncoH);
  pEncoH->scannerTopology.totalNumberOfRsectors = topo[4] + 1;
  pEncoH->scannerTopology.totalNumberOfModules = topo[3] + 1;
  pEncoH->scannerTopology.totalNumberOfSubmodules = topo[2] + 1;
  pEncoH->scannerTopology.totalNumberOfCrystals = topo[1] + 1;
  pEncoH->scannerTopology.totalNumberOfLayers = topo[0] + 1;
  free(topo);

  topo = demakeid(tangentialID, pEncoH);
  pEncoH->scannerTopology.numberOfSectors = topo[4] + 1;
  pEncoH->scannerTopology.tangentialNumberOfModules = topo[3] + 1;
  pEncoH->scannerTopology.tangentialNumberOfSubmodules = topo[2] + 1;
  pEncoH->scannerTopology.tangentialNumberOfCrystals = topo[1] + 1;
  pEncoH->scannerTopology.radialNumberOfLayers = topo[0] + 1;
  free(topo);

  topo = demakeid(axialID, pEncoH);
  pEncoH->scannerTopology.numberOfRings = topo[4] + 1;
  pEncoH->scannerTopology.axialNumberOfModules = topo[3] + 1;
  pEncoH->scannerTopology.axialNumberOfSubmodules = topo[2] + 1;
  pEncoH->scannerTopology.axialNumberOfCrystals = topo[1] + 1;
  pEncoH->scannerTopology.axialNumberOfLayers = topo[0] + 1;
  free(topo);


  /* read the number of types of records */
  if ((fread(buffID16.w8, sizeof(u8), 2, pf)) != 2)
    printf("\n***ERROR : in readHead.c : impossible to do : fread()\n");
  /* and store it */
  buffID16.w16 = ntohs(buffID16.w16);
  pEncoH->scanContent.nRecord = buffID16.w16;

  buffer = malloc(buffID16.w16 * sizeof(u16));
  fread(buffer, sizeof(u8), 2 * (buffID16.w16), pf);	/*buffer the 2,4 or 6 next bytes */
  pEncoH->scanContent.eventRecordBool = 0;	/*fill the bool and the tags */
  pEncoH->scanContent.countRateRecordBool = 0;
  pEncoH->scanContent.gateDigiRecordBool = 0;
  pEncoH->scanContent.eventRecordTag = EVENT_RECORD_TAG;
  pEncoH->scanContent.countRateRecordTag = COUNT_RATE_RECORD_TAG;
  pEncoH->scanContent.gateDigiRecordTag = GATE_DIGI_RECORD_TAG;


  for (i = 0; i < pEncoH->scanContent.nRecord; i++) {	/* loop on different record types */
    buffID16.w8[0] = buffer[2 * i];	/* buffID16 = next 2 bytes */
    buffID16.w8[1] = buffer[2 * i + 1];	/* buffID16 = next 2 bytes */
    buffID16.w16 = ntohs(buffID16.w16);

    if ((checkEventTag(buffID16.w16)) == 1)	/*is it  an event pattern ? */
      pEncoH->scanContent.eventRecordBool = 1;

    if ((checkCountRateTag(buffID16.w16)) == 1)	/*is it a count rate pattern ? */
      pEncoH->scanContent.countRateRecordBool = 1;

    if ((checkGateDigiTag(buffID16.w16)) == 1)	/*is it  a gate digi pattern ? */
      pEncoH->scanContent.gateDigiRecordBool = 1;
  }
  free(buffer);

  return pEncoH;
}
