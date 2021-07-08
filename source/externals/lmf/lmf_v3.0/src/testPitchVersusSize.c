/*-------------------------------------------------------

           List Mode Format 
                        
     --  testPitchVersusSize.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of testPitchVersusSize.c:


	Find the 3D coordinates of an event in the scanner coordinates (x,y,z) system
	->testPitchVersusSize - Control the values of substructurePitches 
	  in comparison with structureSizes
		 
-
-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>		/*EXIT_SUCCESS & EXIT_FAILURE */
#include "lmf.h"

/* testPitchVersusSize - Control the values of substructurePitches in comparison with structureSizes */

int testPitchVersusSize(double *first_pSubstructuresNumericalValues,
			LMF_cch_scannerGeometry * pScanGeo,
			double *first_pRdSizeLayers)
{

  int list_index = 0, structuresSizes_index = 0;
  double totalLayersRadialSize = 0;
  double *pStructDataGeometry = (double *) pScanGeo;
  double *first_pStructDataGeometry = pStructDataGeometry;
  double *pRdSizeLayers = NULL,
      *pTangentialPitch = NULL,
      *pAxialPitch = NULL,
      *pAxialNumberOfSubstructures = NULL,
      *pTangentialNumberOfSubstructures = NULL;

  i8 generalScannerGeometryInfo[21][charNum] =
      { {"geometrical design type"},
  {"ring diameter"},
  {"azimuthal step"},
  {"axial step"},
  {"rsector axial pitch"},
  {"rsector axial size"},
  {"rsector tangential size"},
  {"rsector azimuthal pitch"},
  {"module axial pitch"},
  {"module tangential pitch"},
  {"module axial size"},
  {"module tangential size"},
  {"submodule axial pitch"},
  {"submodule tangential pitch"},
  {"submodule axial size"},
  {"submodule tangential size"},
  {"crystal axial pitch"},
  {"crystal tangential pitch"},
  {"crystal axial size"},
  {"crystal tangential size"},
  {"crystal radial size"}
  };

  i8 axialStructuresSizeDescription[4][charNum] = { {"crystal axial size"},
  {"submodule axial size"},
  {"module axial size"},
  {"rsector axial size"}
  };


  i8 tangentialStructuresSizeDescription[4][charNum] =
      { {"crystal tangential size"},
  {"submodule tangential size"},
  {"module tangential size"},
  {"rsector tangential size"}
  };
  i8 substructuresStructuresName[5][charNum] = { {"layer"},
  {"crystal"},
  {"submodule"},
  {"module"},
  {"rsector"}
  };

  double structuresSizes[2][5] = { {0} };

  /* Find the axial size structures values and store these info 
     in the first row of the table called structuresSizes */
  for (structuresSizes_index = 0; structuresSizes_index < 4;
       structuresSizes_index++) {
    for (list_index = 0; list_index < 20; list_index++) {
      if (strcasecmp(axialStructuresSizeDescription[structuresSizes_index],
		     generalScannerGeometryInfo[list_index]) == 0) {
	pStructDataGeometry = first_pStructDataGeometry + list_index;
	structuresSizes[0][structuresSizes_index + 1] =
	    *pStructDataGeometry;
	break;
      }
    }
  }

  /* Find the tangential size structures values and store these info 
     in the second row of the table called structuresSizes */
  for (structuresSizes_index = 0; structuresSizes_index < 4;
       structuresSizes_index++) {
    for (list_index = 0; list_index < 20; list_index++) {
      if (strcasecmp
	  (tangentialStructuresSizeDescription[structuresSizes_index],
	   generalScannerGeometryInfo[list_index]) == 0) {
	pStructDataGeometry = first_pStructDataGeometry + list_index;
	structuresSizes[1][structuresSizes_index + 1] =
	    *pStructDataGeometry;
	break;
      }
    }
  }

  /* Control the values of substructurePitches in comparison with structureSizes */
  pAxialPitch = first_pSubstructuresNumericalValues;
  pTangentialPitch = first_pSubstructuresNumericalValues + 5;
  pAxialNumberOfSubstructures = first_pSubstructuresNumericalValues + 10;
  pTangentialNumberOfSubstructures =
      first_pSubstructuresNumericalValues + 15;

  for (list_index = 1; list_index < 4; list_index++) {
    if ((pTangentialPitch[list_index] *
	 (pTangentialNumberOfSubstructures[list_index] - 1)) >
	structuresSizes[1][list_index + 1]) {
      printf(ERROR_GEOMETRY6, substructuresStructuresName[list_index],
	     substructuresStructuresName[list_index + 1]);
      printf(ERROR5, cchFileName);
      return (EXIT_FAILURE);
    }
    if ((pAxialPitch[list_index] *
	 (pAxialNumberOfSubstructures[list_index] - 1)) >
	structuresSizes[0][list_index + 1]) {
      printf(ERROR_GEOMETRY7, substructuresStructuresName[list_index],
	     substructuresStructuresName[list_index + 1]);
      printf(ERROR5, cchFileName);
      return (EXIT_FAILURE);
    }
  }
  /* Control the values of the sum of the radial size of each layer and the crystal radial size */
  for (list_index = 0; list_index < (int) pAxialNumberOfSubstructures[0];
       list_index++) {
    pRdSizeLayers = first_pRdSizeLayers + list_index;
    totalLayersRadialSize = totalLayersRadialSize + (*pRdSizeLayers);
  }
  if (totalLayersRadialSize > pScanGeo->crystalRadialSize) {
    printf(ERROR_GEOMETRY11);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }

  return (EXIT_SUCCESS);
}
