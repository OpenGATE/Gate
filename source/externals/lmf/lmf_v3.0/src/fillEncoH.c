/*-------------------------------------------------------

           List Mode Format 
                        
     --  fillEncoH.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of fillEncoH.c:



     This function is exactly the same that generateEncoH.c.          
     The only differences are I/O and the malloc is not done here     
     
     Here the cut/paste of the generateEncoH.c description :
     this function fill artificially an encoding header structure.    
     exemple1 : pEncoH = generateEncoH(1);                            
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

ENCODING_HEADER *fillEncoH(ENCODING_HEADER * pEncoH, u16 chosenMode)
{


  i16 choice = 0;		/* for the user's choice */



  /* An example of encodingHeader */
  pEncoH->scanEncodingID.bitForRsectors = BITS_FOR_RSECTORS;	/* bits reserved for rings and sectors in ID */
  pEncoH->scanEncodingID.maximumRsectors = poweri8(2, BITS_FOR_RSECTORS);	/*  max of sectors */
  pEncoH->scanEncodingID.bitForModules = BITS_FOR_MODULES;	/*  bits reserved for modules in ID */
  pEncoH->scanEncodingID.maximumModules = poweri8(2, BITS_FOR_MODULES);	/* number max of modules */
  pEncoH->scanEncodingID.bitForSubmodules = BITS_FOR_SUBMODULES;	/*  bits reserved for submodules in ID */
  pEncoH->scanEncodingID.maximumSubmodules = poweri8(2, BITS_FOR_SUBMODULES);	/* number max of submodules */
  pEncoH->scanEncodingID.bitForCrystals = BITS_FOR_CRYSTALS;	/*  bits reserved for crystals in ID */
  pEncoH->scanEncodingID.maximumCrystals = poweri8(2, BITS_FOR_CRYSTALS);	/* number max of crystals */
  pEncoH->scanEncodingID.bitForLayers = BITS_FOR_LAYERS;	/*  bits reserved for layers in ID */
  pEncoH->scanEncodingID.maximumLayers = poweri8(2, BITS_FOR_LAYERS);	/* number max of layers */

  pEncoH->scannerTopology.numberOfRings = NUMBER_OF_RINGS;	/* number of rings  */
  pEncoH->scannerTopology.numberOfSectors = NUMBER_OF_SECTORS;	/* number of sectors per ring */
  pEncoH->scannerTopology.totalNumberOfRsectors = NUMBER_OF_RINGS * NUMBER_OF_SECTORS;	/*number total of sectors */

  pEncoH->scannerTopology.axialNumberOfModules = AXIAL_NUMBER_OF_MODULES;	/* number of modules axially */
  pEncoH->scannerTopology.tangentialNumberOfModules = TANGENTIAL_NUMBER_OF_MODULES;	/* number of modules tangentially */
  pEncoH->scannerTopology.totalNumberOfModules = AXIAL_NUMBER_OF_MODULES * TANGENTIAL_NUMBER_OF_MODULES;	/* number of modules per sector */

  pEncoH->scannerTopology.axialNumberOfSubmodules = AXIAL_NUMBER_OF_SUBMODULES;	/* number of submodules axially */
  pEncoH->scannerTopology.tangentialNumberOfSubmodules = TANGENTIAL_NUMBER_OF_SUBMODULES;	/* number of submodules tangentially */
  pEncoH->scannerTopology.totalNumberOfSubmodules = AXIAL_NUMBER_OF_SUBMODULES * TANGENTIAL_NUMBER_OF_SUBMODULES;	/* number of submodules per modules */

  pEncoH->scannerTopology.axialNumberOfCrystals = AXIAL_NUMBER_OF_CRYSTALS;	/* number of crystals axially */
  pEncoH->scannerTopology.tangentialNumberOfCrystals = TANGENTIAL_NUMBER_OF_CRYSTALS;	/* number of crystals tangentially */
  pEncoH->scannerTopology.totalNumberOfCrystals = AXIAL_NUMBER_OF_CRYSTALS * TANGENTIAL_NUMBER_OF_CRYSTALS;	/* number of crystals per submodule */

  pEncoH->scannerTopology.axialNumberOfLayers = AXIAL_NUMBER_OF_LAYERS;	/* number of layers axially (always 1) */
  pEncoH->scannerTopology.radialNumberOfLayers = RADIAL_NUMBER_OF_LAYERS;	/* number of layers radially */
  pEncoH->scannerTopology.totalNumberOfLayers = AXIAL_NUMBER_OF_LAYERS * RADIAL_NUMBER_OF_LAYERS;	/* number of layers = nLayerTangential */


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
      printf("*** WARNING : fillEncoh.c : Anormal chosen mode :\n");
      printf
	  ("impossible to have gate digi record without event record \n");
      exit(0);
    }
  }



  pEncoH->scanContent.eventRecordTag = EVENT_RECORD_TAG;	/* event tag=0(1st bit of encoding event) */
  pEncoH->scanContent.countRateRecordTag = COUNT_RATE_RECORD_TAG;	/* countrate tag = 1000 (4 bits) */
  pEncoH->scanContent.gateDigiRecordTag = GATE_DIGI_RECORD_TAG;	/* gate digi tag = 1100 (4 bits) */

  return (pEncoH);

}
