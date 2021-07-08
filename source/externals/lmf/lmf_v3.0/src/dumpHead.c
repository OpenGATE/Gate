/*-------------------------------------------------------

           List Mode Format 
                        
     --  dumpHead.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dumpHead.c:
     This function called by dumpTheRecord()
     dispays on screen the containing of a
     encoding header structure.



-------------------------------------------------------*/
#include <stdio.h>		/*for printf... */
#include <stdlib.h>		/*for malloc */
#include <netinet/in.h>		/*for ntohs & htons */
#include "lmf.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
this function read the head of a .ccs file 

Ex :
dumpHead(pfread); 
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

void dumpHead(const ENCODING_HEADER * pEncoH,
	      const EVENT_HEADER * pEH,
	      const GATE_DIGI_HEADER * pGDH,
	      const COUNT_RATE_HEADER * pCRH)
{


  printf("****  Crystal ID RULE    ***\n");
  printf("* RSECTOR : %d bits. Max = %d\n",
	 pEncoH->scanEncodingID.bitForRsectors,
	 pEncoH->scanEncodingID.maximumRsectors);
  printf("* MODULE : %d bits. Max = %d\n",
	 pEncoH->scanEncodingID.bitForModules,
	 pEncoH->scanEncodingID.maximumModules);
  printf("* SUBMODULE : %d bits. Max = %d\n",
	 pEncoH->scanEncodingID.bitForSubmodules,
	 pEncoH->scanEncodingID.maximumSubmodules);
  printf("* CRYSTAL : %d bits. Max = %d\n",
	 pEncoH->scanEncodingID.bitForCrystals,
	 pEncoH->scanEncodingID.maximumCrystals);
  printf("* LAYER : %d bits. Max = %d\n",
	 pEncoH->scanEncodingID.bitForLayers,
	 pEncoH->scanEncodingID.maximumLayers);

  printf("\n");
  printf("****  GENERAL TOPOLOGY   ***\n");
  printf("* Number of RSECTOR : %d\n",
	 pEncoH->scannerTopology.totalNumberOfRsectors);
  printf("* Number of MODULE : %d\n",
	 pEncoH->scannerTopology.totalNumberOfModules);
  printf("* Number of SUBMODULE : %d\n",
	 pEncoH->scannerTopology.totalNumberOfSubmodules);
  printf("* Number of CRYSTAL : %d\n",
	 pEncoH->scannerTopology.totalNumberOfCrystals);
  printf("* Number of LAYER : %d\n",
	 pEncoH->scannerTopology.totalNumberOfLayers);

  printf("\n");
  printf("***  TANGENTIAL TOPOLOGY  **\n");
  printf("* Number of RSECTOR : %d\n",
	 pEncoH->scannerTopology.numberOfSectors);
  printf("* Number of MODULE : %d\n",
	 pEncoH->scannerTopology.tangentialNumberOfModules);
  printf("* Number of SUBMODULE : %d\n",
	 pEncoH->scannerTopology.tangentialNumberOfSubmodules);
  printf("* Number of CRYSTAL : %d\n",
	 pEncoH->scannerTopology.tangentialNumberOfCrystals);
  printf("* Number of LAYER (radial) : %d\n",
	 pEncoH->scannerTopology.radialNumberOfLayers);

  printf("\n");
  printf("****   AXIAL TOPOLOGY   ***\n");
  printf("* Number of RSECTOR : %d\n",
	 pEncoH->scannerTopology.numberOfRings);
  printf("* Number of MODULE : %d\n",
	 pEncoH->scannerTopology.axialNumberOfModules);
  printf("* Number of SUBMODULE : %d\n",
	 pEncoH->scannerTopology.axialNumberOfSubmodules);
  printf("* Number of CRYSTAL : %d\n",
	 pEncoH->scannerTopology.axialNumberOfCrystals);
  printf("* Number of LAYER : %d\n",
	 pEncoH->scannerTopology.axialNumberOfLayers);

  printf("\n");
  printf("*** NUMBER OF DIFFERENT  ***\n");
  printf("***    RECORD TYPES      ***\n");
  printf("* %d record types :\n", pEncoH->scanContent.nRecord);

  if (pEncoH->scanContent.eventRecordBool == 1) {
    printf("\n- EVENT RECORD");
    dumpEventHeader(pEH);
  }
  if (pEncoH->scanContent.gateDigiRecordBool == 1) {
    printf("\n- GATE DIGI RECORD");
    dumpGateDigiHeader(pGDH);
  }



  if (pEncoH->scanContent.countRateRecordBool == 1) {
    printf("\n- COUNT RATE RECORD");
    dumpCountRateHeader(pCRH);
  }


}
