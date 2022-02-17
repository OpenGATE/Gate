/*-------------------------------------------------------

           List Mode Format 
                        
     --  exempleMain_05.c  --                      

     Magalie.krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/
/*------------------------------------------------------------------------
			   
	 Description : 
	 
         Scan all detector IDs in scanner and calculate the XYZ position.
	 Output result.data

---------------------------------------------------------------------------*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../includes/lmf.h"
#include <time.h>


int cch_index = 0;
int row_index = 0;

int main()
{
  struct LMF_ccs_eventRecord *pER = NULL;
  struct LMF_ccs_encodingHeader *pEncoH = NULL;
  FILE *pToCcsFile = NULL;
  int index = 0;
  int numericalValue[7] = { 0 };
  calculOfEventPosition resultOfCalculOfEventPosition;

  FILE *result = NULL;

  u16 errFlag;
  /*  
     i8 description[7][charNum]={{"rsector ID"},
     {"module ID"},
     {"submodule ID"},
     {"crystal ID"},
     {"layer ID"},
     {"gantry's angular position"},
     {"gantry's axial position"}};
     u16 limits[7]={0};
     contentLMFdata structRingDiameter = {0};
     i8 input[charNum];
   */

  if (LMFcchReader(""))
    exit(EXIT_FAILURE);		/* read file.cch and fill in structures LMF_cch */

  /* fill in the LMF_ccs_encodingHeader structure */
  pToCcsFile = fopen(ccsFileName, "rb");
  if (openFile(pToCcsFile, ccsFileName))
    return (EXIT_FAILURE);
  fseek(pToCcsFile, 0L, 0);
  pEncoH = readHead(pToCcsFile);

  if ((pER =
       (struct LMF_ccs_eventRecord *)
       malloc(sizeof(struct LMF_ccs_eventRecord))) == NULL)
    printf("\n***ERROR : in generateER.c : impossible to do : malloc()\n");
  if ((pER->crystalIDs = malloc(sizeof(u16))) == NULL)
    printf("\n***ERROR : in generateER.c : impossible to do : malloc()\n");

  /* -> the rsector ID,
     -> the module ID,
     -> the submodule ID,
     -> the crystal ID,
     -> the layer ID,
     -> the gantry's angular position,
     -> the gantry's axial position.
   */
  index = 0;
  index = getLMF_cchInfo("angular gantry position");
  pER->gantryAngularPos = (u16) plist_cch[index].def_unit_value.vNum;
  index = 0;
  index = getLMF_cchInfo("axial gantry position");
  pER->gantryAxialPos = (u16) plist_cch[index].def_unit_value.vNum;

  result = fopen("result.data", "w+");
  if ((openFile(result, "result.data")) == EXIT_FAILURE)
    exit(EXIT_FAILURE);

  for (numericalValue[0] = 0;
       numericalValue[0] <
       (u16) pEncoH->scannerTopology.totalNumberOfRsectors;
       numericalValue[0]++) {
    for (numericalValue[1] = 0;
	 numericalValue[1] <
	 (u16) pEncoH->scannerTopology.totalNumberOfModules;
	 numericalValue[1]++) {
      for (numericalValue[2] = 0;
	   numericalValue[2] <
	   (u16) pEncoH->scannerTopology.totalNumberOfSubmodules;
	   numericalValue[2]++) {
	for (numericalValue[3] = 0;
	     numericalValue[3] <
	     (u16) pEncoH->scannerTopology.totalNumberOfCrystals;
	     numericalValue[3]++) {
	  for (numericalValue[4] = 0;
	       numericalValue[4] <
	       (u16) pEncoH->scannerTopology.totalNumberOfLayers;
	       numericalValue[4]++) {
	    /* with the rsector ID, the module ID, the submodule ID, the crystal ID and the layer ID,
	       the makeid function find the event ID */
	    pER->crystalIDs[0] =
		makeid(numericalValue[0], numericalValue[1],
		       numericalValue[2], numericalValue[3],
		       numericalValue[4], pEncoH, &errFlag);

	    /* calculation of the event position in the 3D laboratory (x,y,z) system */
	    resultOfCalculOfEventPosition =
		locateEventInLaboratory(pEncoH, pER, 0);

	    fprintf(result, "%d %d %d %d %d %d %d %f %f %f\n",
		    numericalValue[0], numericalValue[1],
		    numericalValue[2], numericalValue[3],
		    numericalValue[4], numericalValue[5],
		    numericalValue[6],
		    resultOfCalculOfEventPosition.
		    eventInLaboratory3DPosition.radial,
		    resultOfCalculOfEventPosition.
		    eventInLaboratory3DPosition.tangential,
		    resultOfCalculOfEventPosition.
		    eventInLaboratory3DPosition.axial);
	  }
	}
      }
    }
  }

  printf("End of result file\n");
  fclose(result);

  /* frees the allocated memory by a previous call to malloc(), calloc() or realloc() */
  LMFcchReaderDestructor();
  free(pER->crystalIDs);
  free(pER);
  free(pEncoH);

  return (EXIT_SUCCESS);


}
