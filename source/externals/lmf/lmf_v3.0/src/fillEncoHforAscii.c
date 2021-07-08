/*-------------------------------------------------------

           List Mode Format 
                        
     --  fillEncoHforAscii.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of fillEncoHforAscii.c:


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



ENCODING_HEADER *fillEncoHforAscii(int aR, int tR,	/* // axial / tangeantial rsector */
				   int aM, int tM,	/* // ...  submodule */
				   int aS, int tS,	/* // module */
				   int aC, int tC,	/* // crystal */
				   int aL, int rL,	/* // axial / radial layer */
				   ENCODING_HEADER * pEncoH,
				   u16 chosenMode)
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
  pEncoH->scannerTopology.radialNumberOfLayers = rL;	/* number of layers radially */
  pEncoH->scannerTopology.totalNumberOfLayers = aL * rL;	/* number of layers = nLayerTangential */

  inteligentRuleMaker(aR * tR, aM * tM, aS * tS, aC * tC, aL * rL, pEncoH);
  /*  // find the reserved number of bytes for each level */



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


  return (pEncoH);

}
