/*-------------------------------------------------------

List Mode Format 
                        
--  LMFcchReader.c  --                      

Magalie.Krieguer@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of LMFcchReader.c:

Fill in the LMF record carrier with the data contained in the LMF ASCII header file
Functions used for the ascii part of LMF:
->LMFcchReader - Read the scan file (.cch) and fill in
structures LMF_cch in the LMF Record Carrier
->LMFcchReaderDestructor - Destroy structures LMF_cch
in the LMF Record Carrier

---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

/**** Function LMFcchReader ****/

int LMFcchReader(i8 inputFile[charNum])
{
  int cch_index = 0;
  int row_index = 0;
  int last_lmf_header = 0;

  static ENCODING_HEADER *pEncoHforGeometry;
  FILE *pToCcsFile = NULL;

  last_lmf_header = readTheLMFcchDataBase();

  /** set the input file name **/
  if (setFileName(inputFile) != EXIT_SUCCESS)
    return (EXIT_FAILURE);
  if ((last_cch_list = fillInFieldAndData()) == 0)
    return (EXIT_FAILURE);


  /* create the space for the list of Shift values dynamically */
  pToCcsFile = fopen(ccsFileName, "rb");
  if (openFile(pToCcsFile, ccsFileName))
    return (EXIT_FAILURE);
  fseek(pToCcsFile, 0L, 0);
  pEncoHforGeometry = readHead(pToCcsFile);

  if ((ppShiftValuesList =
       calloc((int) pEncoHforGeometry->scannerTopology.
	      totalNumberOfRsectors, sizeof(double))) == NULL) {
    printf(ERROR36);
    return (EXIT_FAILURE);
  }
  for (row_index = 0;
       row_index <
       (int) pEncoHforGeometry->scannerTopology.totalNumberOfRsectors;
       row_index++) {
    if ((ppShiftValuesList[row_index] = calloc(3, sizeof(double))) == NULL) {
      printf(ERROR36);
      return (EXIT_FAILURE);
    }
  }
  lastShiftValuesList_index = row_index;

  /** Convert and store cch data in the structures LMF_cch **/

  for (cch_index = 0; cch_index <= last_cch_list; cch_index++) {
    if ((last_lmf_header = testField(last_lmf_header, cch_index)) < 0)
      return (EXIT_FAILURE);
    if (fillInCchDefaultUnitValue
	(last_lmf_header, cch_index, pEncoHforGeometry) != 0)
      return (EXIT_FAILURE);
  }

  /* if(editLMF_cchData(last_lmf_header)!=0) return(EXIT_FAILURE); */

  /* printf("Total nb of Rsectors:\t%d\n",(int)pEncoHforGeometry->scannerTopology.totalNumberOfRsectors); */

  /** Test the scan file name give by the user (in keyboard) and the scan file name store in the cch file **/
  cch_index = getLMF_cchInfo("scan file name");

  if ((strcasecmp(fileName, plist_cch[cch_index].def_unit_value.vChar)) !=
      0) {
    printf(ERROR32, fileName, plist_cch[cch_index].def_unit_value.vChar);
    return (EXIT_FAILURE);
  }
  /* Make available to the system the space previously allocated by calloc: destroy pEncoHforGeometry */
  destroyReadHead();


  getTimeStepFromCCH();

  return (EXIT_SUCCESS);
}


/**** Function LMFcchReaderDestructor ****/

void LMFcchReaderDestructor()
{
  int row_index = 0;

  if (plist_lmf_header != NULL)
    free(plist_lmf_header);
  if (plist_cch != NULL)
    free(plist_cch);
  plist_lmf_header = NULL;
  plist_cch = NULL;
  for (row_index = 0; row_index < (lastShiftValuesList_index + 1);
       row_index++) {
    if (ppShiftValuesList[row_index])
      free(ppShiftValuesList[row_index]);

  }
  free(ppShiftValuesList);
}
