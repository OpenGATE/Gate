/*-------------------------------------------------------

List Mode Format 
                        
--  5rings.c  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2004 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------
Description:
New function which allows to pass from ClearPET 1ring 
8 crystal rings shifted to a non-shifted 5 crystal rings

-------------------------------------------------------*/


#include <stdio.h>
#include "lmf.h"

static LMF_cch_scannerGeometry myScannerGeometry = { 0 };
static LMF_cch_scannerGeometry *pScanGeo = &myScannerGeometry;
static double *first_pSubstructuresNumericalValues = NULL;
static double *pSubstructuresNumericalValues = NULL;


void init5rings(const ENCODING_HEADER * pEncoH,
		ENCODING_HEADER ** ppEncoHC)
{
  int cch_index = 0;

  if (fillInStructScannerGeometry(cch_index, pScanGeo) == 1) {
    printf(ERROR_GEOMETRY1);
    printf(ERROR5, cchFileName);
    exit(EXIT_FAILURE);
  }

  first_pSubstructuresNumericalValues =
      setSubstructuresValues(pScanGeo, pEncoH);
  pSubstructuresNumericalValues = first_pSubstructuresNumericalValues + 15;

  *ppEncoHC = (ENCODING_HEADER *) malloc(sizeof(ENCODING_HEADER));

  if (pEncoH)
    **ppEncoHC = *pEncoH;	/* no pointer in this structure, it is safe */

  (*ppEncoHC)->scannerTopology.axialNumberOfCrystals = 5;
}

void make5rings(const ENCODING_HEADER * pEncoH,
		const EVENT_HEADER * pEH,
		EVENT_RECORD ** ppER, ENCODING_HEADER ** ppEncoHC)
{
  generalSubstructureID localSubstructureID = { 0 };
  u16 *pcrist;
  const int substructureOrder = 1;
  u16 sct;
  int axialNb, tangNb;
  u16 errFlag = 0;
  static u8 doneonce = 0;


  if (!doneonce) {
    init5rings(pEncoH, ppEncoHC);
    doneonce++;
  }

  pcrist = demakeid((*ppER)->crystalIDs[0], pEncoH);
  sct = pcrist[4];
  localSubstructureID =
      locateID(pcrist, substructureOrder,
	       first_pSubstructuresNumericalValues);

  axialNb = localSubstructureID.axial;
  tangNb = localSubstructureID.tangential;

  if (sct % 2) {
    if (axialNb > 4)
      *ppER = NULL;
  } else {
    if (axialNb < 3)
      *ppER = NULL;
    else {
      axialNb -= 3;
      pcrist[1] =
	  (int) pSubstructuresNumericalValues[substructureOrder] *
	  axialNb + tangNb;
      (*ppER)->crystalIDs[0] =
	  makeid(pcrist[4], pcrist[3], pcrist[2], pcrist[1], pcrist[0],
		 pEncoH, &errFlag);
    }
  }

  free(pcrist);

  if (*ppER)
    if (pEH->coincidenceBool == TRUE) {
      pcrist = demakeid((*ppER)->crystalIDs[1], pEncoH);
      sct = pcrist[4];
      localSubstructureID =
	  locateID(pcrist, substructureOrder,
		   first_pSubstructuresNumericalValues);

      axialNb = localSubstructureID.axial;
      tangNb = localSubstructureID.tangential;

      if (sct % 2) {
	if (axialNb > 4)
	  *ppER = NULL;
      } else {
	if (axialNb < 3)
	  *ppER = NULL;
	else {
	  axialNb -= 3;
	  pcrist[1] =
	      (int) pSubstructuresNumericalValues[substructureOrder] *
	      axialNb + tangNb;
	  (*ppER)->crystalIDs[1] =
	      makeid(pcrist[4], pcrist[3], pcrist[2], pcrist[1], pcrist[0],
		     pEncoH, &errFlag);
	}
      }

      free(pcrist);
    }
}
