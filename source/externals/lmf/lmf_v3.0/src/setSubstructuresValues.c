/*-------------------------------------------------------

           List Mode Format 
                        
     --  setSubstructuresValues.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of setSubstructuresValues.c:


	Find the 3D coordinates of an event in the scanner coordinates (x,y,z) system
	->setSubstructuresValues - Create an array (called "substructureNumericalValues"),
	  which contains the substructures numerical values:
	  ------------------------------------------------------------------------------------
	  | layerAxPitch - crystalAxPitch - submoduleAxPitch - moduleAxPitch - rsectorAxPitch |
	  ------------------------------------------------------------------------------------
	  | layerRdPitch - crystalTgPitch - submoduleTgPitch - moduleTgPitch - rsectorTgPitch |
	  ------------------------------------------------------------------------------------
	  | axNbOfLayers - axNbOfCrystals - axNbOfSubmodules - axNbOfModules - axNbOfRsectors |
	  ------------------------------------------------------------------------------------
	  | rdNbOfLayers - tgNbOfCrystals - tgNbOfSubmodules - tgNbOfModules - tgNbOfRsectors |
	  ------------------------------------------------------------------------------------
-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>		/*EXIT_SUCCESS & EXIT_FAILURE */
#include "lmf.h"

/*  setSubstructuresValues - Create an array (called "substructureNumericalValues") */
/*--------------------------------------------------------------------------------------------------------------
| layerAxialPitch/crystalAxialPitch/submoduleAxialPitch/moduleAxialPitch/rsectorAxialPitch                      | 
----------------------------------------------------------------------------------------------------------------
| layerRadialPitch/crystalTangentialPitch/submoduleTangentialPitch/moduleTangentialPitch/rsectorTangentialPitch |
----------------------------------------------------------------------------------------------------------------
| axialNumberOfLayers/axialNumberOfCrystals/axialNumberOfSubmodules/axialNumberOfModules/axialNumberOfRsectors  |
---------------------------------------------------------------------------------------------------------------
| radialNbOfLayers/tangentialNbOfCrystals/tangentialNbOfSubmodules/tangentialNbOfModules/tangentialNbOfRsectors |
---------------------------------------------------------------------------------------------------------------*/

double *setSubstructuresValues(LMF_cch_scannerGeometry * pScanGeo,
			       const ENCODING_HEADER * pEncoH)
{
  int list_index = 0, substructuresNumericalValues_index = 0;
  double *pStructDataGeometry = (double *) pScanGeo;
  double *first_pStructDataGeometry = pStructDataGeometry;
  double *pSubstructuresNumericalValues = NULL;
  i8 generalScannerGeometryInfo[21][charNum] =
      { {"geometrical design type"},
  {"ring diameter"},
  {"azimuthal step"},
  {"axial step"},
  /* RSECTOR */
  {"rsector axial pitch"},
  {"rsector axial size"},
  {"rsector tangential size"},
  {"rsector azimuthal pitch"},
  /* MODULE */
  {"module axial pitch"},
  {"module tangential pitch"},
  {"module axial size"},
  {"module tangential size"},
  /* SUBMODULE */
  {"submodule axial pitch"},
  {"submodule tangential pitch"},
  {"submodule axial size"},
  {"submodule tangential size"},
  /* CRYSTAL */
  {"crystal axial pitch"},
  {"crystal tangential pitch"},
  {"crystal axial size"},
  {"crystal tangential size"},
  {"crystal radial size"}
  };

  i8 axialSubstructuresDescription[4][charNum] = { {"crystal axial pitch"},
  {"submodule axial pitch"},
  {"module axial pitch"},
  {"rsector axial pitch"}
  };


  i8 tangentialSubstructuresDescription[3][charNum] =
      { {"crystal tangential pitch"},
  {"submodule tangential pitch"},
  {"module tangential pitch"}
  };

  static double substructuresNumericalValues[4][5] = { {0} };

  /* Find the substructure axial pitch and store this info 
     in the first row of the array called substructuresNumericalValues */
  for (substructuresNumericalValues_index = 0;
       substructuresNumericalValues_index < 4;
       substructuresNumericalValues_index++) {
    for (list_index = 0; list_index < 21; list_index++) {
      if (strcasecmp
	  (axialSubstructuresDescription
	   [substructuresNumericalValues_index],
	   generalScannerGeometryInfo[list_index]) == 0) {
	pStructDataGeometry = first_pStructDataGeometry + list_index;
	substructuresNumericalValues[0][substructuresNumericalValues_index
					+ 1] = *pStructDataGeometry;
	break;
      }
    }
  }

  /* Find the substructure tangential pitch and store this info
     in the second row of the array called substructuresNumericalValues */
  for (substructuresNumericalValues_index = 0;
       substructuresNumericalValues_index < 3;
       substructuresNumericalValues_index++) {
    for (list_index = 0; list_index < 21; list_index++) {
      if (strcasecmp
	  (tangentialSubstructuresDescription
	   [substructuresNumericalValues_index],
	   generalScannerGeometryInfo[list_index]) == 0) {
	pStructDataGeometry = first_pStructDataGeometry + list_index;
	substructuresNumericalValues[1][substructuresNumericalValues_index
					+ 1] = *pStructDataGeometry;
	break;
      }
    }
  }

  /* Fill in the third row of the array called substructuresNumericalValues with:
     axialNumberOfLayers, 
     axialNumberOfCrystals, 
     axialNumberOfSubmodules, 
     axialNumberOfModules, 
     axialNumberOfRsectors */

  substructuresNumericalValues[2][0] =
      (double) pEncoH->scannerTopology.axialNumberOfLayers;
  substructuresNumericalValues[2][1] =
      (double) pEncoH->scannerTopology.axialNumberOfCrystals;
  substructuresNumericalValues[2][2] =
      (double) pEncoH->scannerTopology.axialNumberOfSubmodules;
  substructuresNumericalValues[2][3] =
      (double) pEncoH->scannerTopology.axialNumberOfModules;
  substructuresNumericalValues[2][4] =
      (double) pEncoH->scannerTopology.numberOfRings;

  /* Fill in the fourth row of the array called substructuresNumericalValues with:
     radialNumberOfLayers, 
     tangentialNumberOfCrystals, 
     tangentialNumberOfSubmodules,
     tangentialNumberOfModules, 
     tangentialNumberOfRsectors */

  substructuresNumericalValues[3][0] =
      (double) pEncoH->scannerTopology.radialNumberOfLayers;
  substructuresNumericalValues[3][1] =
      (double) pEncoH->scannerTopology.tangentialNumberOfCrystals;
  substructuresNumericalValues[3][2] =
      (double) pEncoH->scannerTopology.tangentialNumberOfSubmodules;
  substructuresNumericalValues[3][3] =
      (double) pEncoH->scannerTopology.tangentialNumberOfModules;
  substructuresNumericalValues[3][4] =
      (double) pEncoH->scannerTopology.numberOfSectors;

  pSubstructuresNumericalValues = &substructuresNumericalValues[0][0];

  return (pSubstructuresNumericalValues);
}
