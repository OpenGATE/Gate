/*-------------------------------------------------------

List Mode Format 
                        
--  fillEncoHforGate.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of fillEncoHforGate.c:

This function is exactly the same that generateEncoH.c.          
The only differences are I/O and the malloc is not done here     
     
Here the cut/paste of the generateEncoH.c description :
this function fill artificially an encoding header structure.    
exemple1 : pEncoH = generateEncoH(0);                            
exemple2 : pEncoH = generateEncoH(askTheMode());                 
the last one is for asking to the user the type of records that 
he wants to store :                                              
0 : no records                                                   
1 : event records                                                
2 : count rate records                                           
3 : the both                                                      

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"
void fillEncoHforGate(int aR, int tR,	/* // axial / tangeantial rsector */
		      int aM, int tM,	/* // ...  submodule */
		      int aS, int tS,	/* // module */
		      int aC, int tC,	/* // crystal */
		      int aL, int rL,	/* // axial / radial layer */
		      ENCODING_HEADER * pEncoH, u16 chosenMode)
{
  i16 choice = 0;		/* for the user's choice */


  printf("\n\n\nEncoding Header maker : \n");
  printf("rsector  = %d x %d\n ", aR, tR);
  printf("module  = %d x %d\n ", aM, tM);
  printf("submodule  = %d x %d\n ", aS, tS);
  printf("crystal  = %d x %d\n ", aC, tC);
  printf("layer  = %d x %d\n ", aL, rL);

  /* An example of encodingHeader */

  pEncoH->scannerTopology.numberOfRings = aR;	/* number of rings  */
  pEncoH->scannerTopology.numberOfSectors = tR;	/* number of sectors per ring */
  pEncoH->scannerTopology.totalNumberOfRsectors = aR * tR;	/*number total of sectors */

  pEncoH->scannerTopology.axialNumberOfModules = aM;	/* number of modules axially */
  pEncoH->scannerTopology.tangentialNumberOfModules = tM;	/* number of modules tangentially */
  pEncoH->scannerTopology.totalNumberOfModules = aM * tM;	/* number of modules per sector */

  pEncoH->scannerTopology.axialNumberOfSubmodules = aS;	/* number of submodules axially */
  pEncoH->scannerTopology.tangentialNumberOfSubmodules = tS;	/* number of submodules tangentially */
  pEncoH->scannerTopology.totalNumberOfSubmodules = aS * tS;	/* number of submodules per modules */




  pEncoH->scannerTopology.axialNumberOfCrystals = aC;	/* number of crystals axially */
  pEncoH->scannerTopology.tangentialNumberOfCrystals = tC;	/* number of crystals tangentially */
  pEncoH->scannerTopology.totalNumberOfCrystals = aC * tC;	/* number of crystals per submodule */

  pEncoH->scannerTopology.axialNumberOfLayers = aL;	/* number of layers axially (always 1) */
  if (rL > 1) {
    pEncoH->scannerTopology.radialNumberOfLayers = rL;	/* number of layers radially */
    pEncoH->scannerTopology.totalNumberOfLayers = aL * rL;	/* number of layers = nLayerTangential */
    inteligentRuleMaker(aR * tR, aM * tM, aS * tS, aC * tC, aL * rL,
			pEncoH);
    /* // find the reserved number of bytes for each level */

  } else {
    pEncoH->scannerTopology.radialNumberOfLayers = 0;	/* number of layers radially */
    pEncoH->scannerTopology.totalNumberOfLayers = 0;	/* number of layers = nLayerTangential */
    inteligentRuleMaker(aR * tR, aM * tM, aS * tS, aC * tC, 0, pEncoH);
    /* // find the reserved number of bytes for each level */

  }




  choice = chosenMode;


  pEncoH->scanContent.nRecord = 0;
  pEncoH->scanContent.eventRecordBool = 0;	/* event not recorded */
  pEncoH->scanContent.countRateRecordBool = 0;	/* countrate not recorded  */


  if (choice & BIT1) {
    pEncoH->scanContent.nRecord++;
    pEncoH->scanContent.eventRecordBool = 1;	/* event recorded */
  }

  if (choice & BIT2) {
    pEncoH->scanContent.nRecord++;
    pEncoH->scanContent.countRateRecordBool = 1;	/* countrate recorded */
  }

  if (choice & BIT3) {
    pEncoH->scanContent.nRecord++;
    pEncoH->scanContent.gateDigiRecordBool = 1;	/* gate digi recorded */
    if (!(choice & BIT1)) {
      printf("*** WARNING : fillEncohforGate.c : Anormal chosen mode :\n");
      printf
	  ("impossible to have gate digi record without event record \n");
      exit(0);
    }
  }


  pEncoH->scanContent.eventRecordTag = EVENT_RECORD_TAG;	/* event tag=0(1st bit of encoding event) */
  pEncoH->scanContent.countRateRecordTag = COUNT_RATE_RECORD_TAG;	/* countrate tag = 1000 (4 bits) */
  pEncoH->scanContent.gateDigiRecordTag = GATE_DIGI_RECORD_TAG;	/* GDR tag = 1100 (4 bits) */
}


void inteligentRuleMaker(int rsector,
			 int module,
			 int submodule,
			 int crystal, 
			 int layer, 
			 ENCODING_HEADER * pEncoH)
{
  u8 bitNumberL, bitNumberC, bitNumberS, bitNumberM, bitNumberR;
  u8 bitTotalNumber = 0;


  /***                  LAYER                      ***/
  if (layer)
    bitNumberL = (u8) findNumberOfBitsNeededFor(layer);
  else
    bitNumberL = 0;		// case of no layer

  pEncoH->scanEncodingID.bitForLayers = bitNumberL;	/*  bits reserved for layers in ID */

  if (bitNumberL)
    pEncoH->scanEncodingID.maximumLayers = poweru32(2, bitNumberL);	/* number max of layers */
  else
    pEncoH->scanEncodingID.maximumLayers = 0;	/* number max of layers */

  bitTotalNumber = bitTotalNumber + bitNumberL;

  /***                  CRYSTAL                      ***/
  bitNumberC = (u8) findNumberOfBitsNeededFor(crystal);
  pEncoH->scanEncodingID.bitForCrystals = bitNumberC;	/*  bits reserved for crystals in ID */
  pEncoH->scanEncodingID.maximumCrystals = poweru32(2, bitNumberC);	/* number max of crystals */
  bitTotalNumber = bitTotalNumber + bitNumberC;

  /***                  SUBMODULE                    ***/
  bitNumberS = (u8) findNumberOfBitsNeededFor(submodule);
  pEncoH->scanEncodingID.bitForSubmodules = bitNumberS;	/*  bits reserved for submodules in ID */
  pEncoH->scanEncodingID.maximumSubmodules = poweru32(2, bitNumberS);	/* number max of submodules */
  bitTotalNumber = bitTotalNumber + bitNumberS;

  /***                  MODULE                      ***/
  bitNumberM = (u8) findNumberOfBitsNeededFor(module);
  pEncoH->scanEncodingID.bitForModules = bitNumberM;	/*  bits reserved for modules in ID */
  pEncoH->scanEncodingID.maximumModules = poweru32(2, bitNumberM);	/* number max of modules */
  bitTotalNumber = bitTotalNumber + bitNumberM;


  /***                  RSECTOR                     ***/

  bitNumberR = (u8) findNumberOfBitsNeededFor(rsector);

  if (bitNumberR + bitTotalNumber > 64) {
    printf
	("*** ERROR : fillEncoHforGate.c : too many elements in scanner : impossible to make a rule");
    exit(0);
  } else if (bitNumberR + bitTotalNumber > 32) {
    bitNumberR = 64 - bitTotalNumber;
    pEncoH->scanEncodingIDLength = 2;
  } else if (bitNumberR + bitTotalNumber > 16) {
    bitNumberR = 32 - bitTotalNumber;
    pEncoH->scanEncodingIDLength = 1;
  } else {
    bitNumberR = 16 - bitTotalNumber;
    pEncoH->scanEncodingIDLength = 0;
  }


  pEncoH->scanEncodingID.bitForRsectors = bitNumberR;	/* bits reserved for rings and sectors in ID */
  pEncoH->scanEncodingID.maximumRsectors = poweru32(2, bitNumberR);	/*  max of sectors */


  printf("\n\nIntelligent LMF rule maker : \n");
  printf("scanEncodingIDLength = %d\n", pEncoH->scanEncodingIDLength);
  printf("%d rsectors -> %d bits\n", rsector, bitNumberR);
  printf("%d modules -> %d bits\n", module, bitNumberM);
  printf("%d submodules -> %d bits\n", submodule, bitNumberS);
  printf("%d crystals -> %d bits\n", crystal, bitNumberC);
  printf("%d layers -> %d bits\n", layer, bitNumberL);
}

/* 
   gives the minimum number of bits needed to contains a int 
   Exemple : for 18, you need at least 5 bits, for 16 just 4 bits ...
*/
int findNumberOfBitsNeededFor(int value)
{
  int n = 0, M = 1;

  while (M + 1 <= value) {
    n++;
    M = M * 2;
  }
  if (value == 1)
    return (1);
  else
    return (n);
}
