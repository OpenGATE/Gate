/*-------------------------------------------------------

           List Mode Format 
                        
     --  fillInStructScannerGeometry.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of fillInStructScannerGeometry.c:

	Find the 3D coordinates of an event in the scanner coordinates (x,y,z) system
	->fillInStructScannerGeometry - Initialize the members of 
	                                the structure LMF_cch_scannerGeometry 
-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>		/*EXIT_SUCCESS & EXIT_FAILURE */
#include "lmf.h"

/** fillInStructScannerGeometry - Initialize the members of the structure 
    LMF_cch_scannerGeometry **/

int fillInStructScannerGeometry(int cch_index,
				LMF_cch_scannerGeometry * pScanGeo)
{

  double *pStructDataGeometry = (double *) pScanGeo;
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

  int list_index = 0, TESTADD = 0;
  double *first_pStructDataGeometry = pStructDataGeometry;

  for (list_index = 0; list_index < 21; list_index++) {
    for (cch_index = 0; cch_index <= last_cch_list; cch_index++) {
      if (strcasecmp
	  (generalScannerGeometryInfo[list_index],
	   plist_cch[cch_index].field) == 0) {
	pStructDataGeometry = first_pStructDataGeometry + list_index;
	*pStructDataGeometry = plist_cch[cch_index].def_unit_value.vNum;
	TESTADD = 1;
	break;
      }
    }
    if (TESTADD == 0) {
      printf(ERROR_GEOMETRY4, generalScannerGeometryInfo[list_index]);
      return (EXIT_FAILURE);
    }
    TESTADD = 0;
  }
  return (EXIT_SUCCESS);
}
