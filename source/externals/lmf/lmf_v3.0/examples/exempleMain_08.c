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
	 Example to use the computation of events position in the 3D laboratory (x,y,z) system.
	 The LMF_ccs_eventRecord structure and the LMF_ccs_encodingHeader structure 
	 are initialized directly in the main. If the user wants to modify the scanner topology,
	 he must open and change the file "../includes/constantsLMF_ccs.h ".
	 The user must choose:
	 -> the rsector ID,
	 -> the module ID,
	 -> the submodule ID,
	 -> the crystal ID,
	 -> the layer ID,
	 -> the gantry's angular position,
	 -> the gantry's axial position.
	 The program prints the event position in the laboratory coordinates (x,y,z) system.  
         
---------------------------------------------------------------------------*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../includes/lmf.h"
#include <time.h>

int main()
{
  struct LMF_ccs_eventRecord *pER = NULL;
  struct LMF_ccs_encodingHeader *pEncoH = NULL;
  int index = 0, inputData = 0;
  i8 description[7][charNum] = { {"rsector ID"},
  {"module ID"},
  {"submodule ID"},
  {"crystal ID"},
  {"layer ID"},
  {"gantry's angular position"},
  {"gantry's axial position"}
  };
  u16 limits[7] = { 0 };
  int numericalValue[7] = { 0 };
  i8 input[charNum];
  calculOfEventPosition resultOfCalculOfEventPosition;
  u16 errFlag;

  /* Initialization of the LMF_ccs_eventRecord structure and the LMF_ccs_encodingHeader structure
     We have chose an "one ring scanner" with 8 rsectors.
     Each rsector has 3 rows of modules tangentially and only one column of modules axially
     Each module is divided in 4 columns of submodules and 1 row tangentially
     Finally each submodule is divided in a matrix of 8 rows of 8 columns of crystals
     We have defined 2 layers radially of each crystal */

  /* fill in the LMF_ccs_encodingHeader structure */
  if ((pEncoH =
       (struct LMF_ccs_encodingHeader *)
       malloc(sizeof(struct LMF_ccs_encodingHeader))) == NULL)
    printf
	("\n*** ERROR : in generateEncoH.c : impossible to do : malloc()\n");
  pEncoH = generateEncoH(1);

  if ((pER =
       (struct LMF_ccs_eventRecord *)
       malloc(sizeof(struct LMF_ccs_eventRecord))) == NULL)
    printf("\n***ERROR : in generateER.c : impossible to do : malloc()\n");
  if ((pER->crystalIDs = malloc(sizeof(u16))) == NULL)
    printf("\n***ERROR : in generateER.c : impossible to do : malloc()\n");

  limits[0] = (u16) pEncoH->scannerTopology.totalNumberOfRsectors;
  limits[1] = (u16) pEncoH->scannerTopology.totalNumberOfModules;
  limits[2] = (u16) pEncoH->scannerTopology.totalNumberOfSubmodules;
  limits[3] = (u16) pEncoH->scannerTopology.totalNumberOfCrystals;
  limits[4] = (u16) pEncoH->scannerTopology.totalNumberOfLayers;
  limits[5] = 256;
  limits[6] = 256;

  /* read file.cch and fill in structures LMF_cch */
  if (LMFcchReader(""))
    exit(EXIT_FAILURE);

  /* the user must choose:
     -> the rsector ID,
     -> the module ID,
     -> the submodule ID,
     -> the crystal ID,
     -> the layer ID,
     -> the gantry's angular position,
     -> the gantry's axial position.
   */

  for (index = 0; index < 7; index++) {
    initialize(input);
    printf("Choose a %s (number between 0 and %d):", description[index],
	   (limits[index] - 1));
    if (*gets(input) == '\0')
      numericalValue[index] = 0;
    else {
      inputData = 0;
      inputData = atoi(input);
      if ((inputData >= 0 && inputData < limits[index]) != 1)
	numericalValue[index] = 0;
      else
	numericalValue[index] = inputData;
    }
  }
  pER->gantryAngularPos = (u16) numericalValue[5];
  pER->gantryAxialPos = (u16) numericalValue[6];

  /* with the rsector ID, the module ID, the submodule ID, the crystal ID and the layer ID,
     the makeid function find the event ID */
  pER->crystalIDs[0] =
      makeid(numericalValue[0], numericalValue[1], numericalValue[2],
	     numericalValue[3], numericalValue[4], pEncoH, &errFlag);

  /* calculation of the event position in the 3D laboratory (x,y,z) system */
  resultOfCalculOfEventPosition = locateEventInLaboratory(pEncoH, pER, 0);

  printf("%s: %d\t%s: %d\t%s: %d\t%s: %d\t%s: %d\t%s: %d\t%s: %d\n",
	 description[0], numericalValue[0],
	 description[1], numericalValue[1],
	 description[2], numericalValue[2],
	 description[3], numericalValue[3],
	 description[4], numericalValue[4],
	 description[5], numericalValue[5],
	 description[6], numericalValue[6]);
  printf("x: %f\ty: %f\tz: %f\n",
	 resultOfCalculOfEventPosition.eventInLaboratory3DPosition.radial,
	 resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	 tangential,
	 resultOfCalculOfEventPosition.eventInLaboratory3DPosition.axial);

  /* frees the allocated memory by a previous call to malloc(), calloc() or realloc() */
  LMFcchReaderDestructor();
  free(pER->crystalIDs);
  free(pER);
  free(pEncoH);

  return (EXIT_SUCCESS);
}
