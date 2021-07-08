/*-------------------------------------------------------

           List Mode Format 
                        
     --  buildHead.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of buildHead.c:
     This function build the head of .ccs FILE 
     The head contains informations of encoding header
     but also record header.
     It needs :
     pointer to Encoding Header Struct.
     pointer to Event Header Struct.
     pointer to Count Rate Header Struct.
     pointer to a writting FILE


-------------------------------------------------------*/
#include <netinet/in.h>		/*for htons and ntohs */
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

void buildHead(const ENCODING_HEADER * pEncoH,
	       const EVENT_HEADER * pEH,
	       const GATE_DIGI_HEADER * pGDH,
	       const COUNT_RATE_HEADER * pCRH, FILE * pf)
{
  u16vsu8 buffID16;		/*buffer for an uns i16 */
  u32vsu8 buffID32;		/*buffer for an uns i16 */
  u64vsu8 buffID64;		/*buffer for an uns i16 */

  u64 generalEncoding, tangentialEncoding, axialEncoding;
  u16 errFlag;

  fseek(pf, 0L, 0);		/* Seek the begin of the file */

  /*BYTES 1 & 2 : ID rule ex: 111 0000 111 00000 1 = rrrssssmmmcccccl */

  buffID16.w16 = (u16) putBits(14);

  generalEncoding =
      makeid(pEncoH->scannerTopology.totalNumberOfRsectors - 1,
	     pEncoH->scannerTopology.totalNumberOfModules - 1,
	     pEncoH->scannerTopology.totalNumberOfSubmodules - 1,
	     pEncoH->scannerTopology.totalNumberOfCrystals - 1,
	     pEncoH->scannerTopology.totalNumberOfLayers - 1, pEncoH,
	     &errFlag);

  tangentialEncoding = makeid(pEncoH->scannerTopology.numberOfSectors - 1,
			      pEncoH->scannerTopology.
			      tangentialNumberOfModules - 1,
			      pEncoH->scannerTopology.
			      tangentialNumberOfSubmodules - 1,
			      pEncoH->scannerTopology.
			      tangentialNumberOfCrystals - 1,
			      pEncoH->scannerTopology.
			      radialNumberOfLayers - 1, pEncoH, &errFlag);

  axialEncoding = makeid(pEncoH->scannerTopology.numberOfRings - 1,
			 pEncoH->scannerTopology.axialNumberOfModules - 1,
			 pEncoH->scannerTopology.axialNumberOfSubmodules -
			 1,
			 pEncoH->scannerTopology.axialNumberOfCrystals - 1,
			 pEncoH->scannerTopology.axialNumberOfLayers - 1,
			 pEncoH, &errFlag);


  switch (pEncoH->scanEncodingIDLength) {
  case 0:
    buffID16.w16 = buffID16.w16 << 2;
    fwrite(buffID16.w8, sizeof(u8), 2, pf);	/*Write on 2 bytes these rule */
    buffID16.w16 = (u16) makeRule(pEncoH);	/* Build the rule ID */
    fwrite(buffID16.w8, sizeof(u8), 2, pf);	/*Write on 2 bytes this ID */

    /*BYTES 3 & 4 : general Encoding */
    buffID16.w16 = (u16) generalEncoding;
    fwrite(buffID16.w8, sizeof(u8), 2, pf);

    /*BYTES 5 & 6 :  Tangential Encoding */
    buffID16.w16 = (u16) tangentialEncoding;
    fwrite(buffID16.w8, sizeof(u8), 2, pf);

    /*BYTES 7 & 8 : Axial Encoding */
    buffID16.w16 = (u16) axialEncoding;
    fwrite(buffID16.w8, sizeof(u8), 2, pf);

    break;
  case 1:
    buffID16.w16 = (buffID16.w16 << 2) | 1;
    fwrite(buffID16.w8, sizeof(u8), 2, pf);	/*Write on 2 bytes these rule */
    buffID32.w32 = (u32) makeRule(pEncoH);	/* Build the rule ID */
    fwrite(buffID32.w8, sizeof(u8), 4, pf);	/*Write on 4 bytes this ID */

    /*BYTES 3 & 4 : general Encoding */
    buffID32.w32 = (u32) generalEncoding;
    fwrite(buffID32.w8, sizeof(u8), 4, pf);

    /*BYTES 5 & 6 :  Tangential Encoding */
    buffID32.w32 = (u32) tangentialEncoding;
    fwrite(buffID32.w8, sizeof(u8), 4, pf);

    /*BYTES 7 & 8 : Axial Encoding */
    buffID32.w32 = (u32) axialEncoding;
    fwrite(buffID32.w8, sizeof(u8), 4, pf);

    break;
  case 2:
    buffID16.w16 = (buffID16.w16 << 2) | 2;
    fwrite(buffID16.w8, sizeof(u8), 2, pf);	/*Write on 2 bytes these rule */
    buffID64.w64 = makeRule(pEncoH);	/* Build the rule ID */
    fwrite(buffID64.w8, sizeof(u8), 8, pf);	/*Write on 8 bytes these rule */
    /*BYTES 3 & 4 : general Encoding */
    buffID64.w64 = generalEncoding;
    fwrite(buffID64.w8, sizeof(u8), 8, pf);

    /*BYTES 5 & 6 :  Tangential Encoding */
    buffID64.w64 = tangentialEncoding;
    fwrite(buffID64.w8, sizeof(u8), 8, pf);

    /*BYTES 7 & 8 : Axial Encoding */
    buffID64.w64 = axialEncoding;
    fwrite(buffID64.w8, sizeof(u8), 8, pf);

    break;
  }


  /*BYTES 9 & 10 : Number of different types of records */
  buffID16.w16 = pEncoH->scanContent.nRecord;
  buffID16.w16 = htons(buffID16.w16);	/* swap the i16 */
  fwrite(buffID16.w8, sizeof(u8), 2, pf);

  /*next 2 BYTES : Encoding pattern for event Record */
  if (pEH) {
    if (pEncoH->scanContent.eventRecordBool == TRUE) {
      buffID16.w16 = makeEpattern(pEH);
      buffID16.w16 = htons(buffID16.w16);	/* swap the i16 */
      fwrite(buffID16.w8, sizeof(u8), 2, pf);
    }
    /*next 2 BYTES  : Encoding pattern for gate digi Record */



    if (pEncoH->scanContent.gateDigiRecordBool == TRUE) {
      buffID16.w16 = makeGDpattern(pGDH);
      buffID16.w16 = htons(buffID16.w16);	/* swap the i16 */
      fwrite(buffID16.w8, sizeof(u8), 2, pf);

    }

  }

  /*next 2 Bytes : Encoding pattern for count rate Record */
  if (pCRH)
    if (pEncoH->scanContent.countRateRecordBool == TRUE) {
      buffID16.w16 = makeCRpattern(pCRH);
      buffID16.w16 = htons(buffID16.w16);	/* swap the i16 */
      fwrite(buffID16.w8, sizeof(u8), 2, pf);
    }
}
