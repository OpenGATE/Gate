/*-------------------------------------------------------

           List Mode Format 
                        
     --  setLayersInfo.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of setLayersInfo.c:

	Find the 3D coordinates of an event in the laboratory 
	coordinates (x,y,z) system
	->setLayersInfo - Initialize parameters layer radial size 
	                  and interaction length of each layer
-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>		/*EXIT_SUCCESS & EXIT_FAILURE */
#include "lmf.h"

/* setLayersInfo - Initialize parameters layers radial sizeand interaction length of each layer */

int setLayersInfo(int cch_index,
		  int rdNbOfLayers,
		  double *first_pIntLengthLayers,
		  double *first_pRdSizeLayers)
{

  double *pIntLengthLayers = NULL, *pRdSizeLayers = NULL;
  i8 **pIntLengthDescriptionList, **pRdSizeDescriptionList;
  int list_index = 0, sizeOfString = (int) charNum, TESTADD = 0;


  pIntLengthDescriptionList = calloc(sizeOfString, sizeof(i8 *));
  pRdSizeDescriptionList = calloc(sizeOfString, sizeof(i8 *));

  for (list_index = 0; list_index < rdNbOfLayers; list_index++) {
    if ((pIntLengthDescriptionList[list_index] =
	 calloc(sizeOfString, sizeof(i8 *))) == NULL) {
      printf(ERROR_GEOMETRY9);
      return (EXIT_FAILURE);
    }

    if ((pRdSizeDescriptionList[list_index] =
	 calloc(sizeOfString, sizeof(i8 *))) == NULL) {
      printf(ERROR_GEOMETRY9);
      exit(EXIT_FAILURE);
    }
  }

  for (list_index = 0; list_index < rdNbOfLayers; list_index++) {
    sprintf(pIntLengthDescriptionList[list_index],
	    "in layer%d interaction length", list_index);
    sprintf(pRdSizeDescriptionList[list_index], "layer%d radial size",
	    list_index);
  }
  pIntLengthLayers = first_pIntLengthLayers;

  for (list_index = 0; list_index < rdNbOfLayers; list_index++) {
    for (cch_index = 0; cch_index <= last_cch_list; cch_index++) {
      if (strcasecmp
	  (pIntLengthDescriptionList[list_index],
	   plist_cch[cch_index].field) == 0) {
	pIntLengthLayers = first_pIntLengthLayers + list_index;
	*pIntLengthLayers = plist_cch[cch_index].def_unit_value.vNum;
	TESTADD = 1;
	break;
      }
    }

    if (TESTADD == 0) {
      printf(ERROR_GEOMETRY4, pIntLengthDescriptionList[list_index]);
      printf(ERROR5, cchFileName);
      return (EXIT_FAILURE);
    }
    TESTADD = 0;
  }
  pRdSizeLayers = first_pRdSizeLayers;

  for (list_index = 0; list_index < rdNbOfLayers; list_index++) {
    for (cch_index = 0; cch_index <= last_cch_list; cch_index++) {
      if (strcasecmp
	  (pRdSizeDescriptionList[list_index],
	   plist_cch[cch_index].field) == 0) {
	pRdSizeLayers = first_pRdSizeLayers + list_index;
	*pRdSizeLayers = plist_cch[cch_index].def_unit_value.vNum;
	TESTADD = 1;
	break;
      }
    }

    if (TESTADD == 0) {
      printf(ERROR_GEOMETRY4, pRdSizeDescriptionList[list_index]);
      printf(ERROR5, cchFileName);
      return (EXIT_FAILURE);
    }
    TESTADD = 0;
  }

  free(pIntLengthDescriptionList);
  free(pRdSizeDescriptionList);

  return (EXIT_SUCCESS);
}
