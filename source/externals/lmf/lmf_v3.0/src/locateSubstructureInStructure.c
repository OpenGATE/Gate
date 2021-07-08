/*-------------------------------------------------------

           List Mode Format 
                        
     --  locateSubstructureInStructure.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of locateSubstructureInStructure.c:


	Find the 3D coordinates of an event in the 3D laboratory (x,y,z) system
	->locateSubstructureInStructure - Calculate the 3D coordinates of a substructure 
	  in the reference frame attached to the rsector:
	    3D coordinates of a crystal in the tangential reference frame attached to the submodule
	  + 3D coordinates of a submodule in the tangential reference frame attached to the module
	  + 3D coordinates of a module in the tangential reference frame attached to the rsector

-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

/* locateSubstructureInStructure - Process to calculate the 3D coordinates of a substructure 
   in the referance frame attached to the rsector = */
/*   3D coordinates of a crystal in the tangential reference frame attached to the submodule */
/* + 3D coordinates of a submodule in the tangential reference frame attached to the module */
/* + 3D coordinates of a module in the tangential reference frame attached to the rsector */


coordinates locateSubstructureInStructure(u16 * pcrist,
					  int substructureOrder,
					  double
					  *first_pSubstructuresNumericalValues)
{

  generalSubstructureID substructureID = { 0 };
  coordinates structureFrame = { 0 };
  double *pTangentialPitch = NULL, *pAxialPitch =
      NULL, *pTangentialNumberOfSubstructures =
      NULL, *pAxialNumberOfSubstructures = NULL;

  pAxialPitch = first_pSubstructuresNumericalValues;
  pTangentialPitch = first_pSubstructuresNumericalValues + 5;
  pAxialNumberOfSubstructures = first_pSubstructuresNumericalValues + 10;
  pTangentialNumberOfSubstructures =
      first_pSubstructuresNumericalValues + 15;

  while (substructureOrder < 4) {

    /* Calculation of the axial and the tangential ID (id_t,id_z) */
    substructureID = locateID(pcrist,
			      substructureOrder,
			      first_pSubstructuresNumericalValues);

    /* Calculation of the substructure position in the reference frame attached to the structure */

    structureFrame.tangential +=
	(double) (pTangentialPitch[substructureOrder] *
		  (substructureID.tangential -
		   (0.5) *
		   (pTangentialNumberOfSubstructures[substructureOrder] -
		    1)));

    structureFrame.axial +=
	(double) (pAxialPitch[substructureOrder] *
		  (substructureID.axial -
		   (0.5) *
		   (pAxialNumberOfSubstructures[substructureOrder] - 1)));

    substructureOrder++;

    locateSubstructureInStructure(pcrist,
				  substructureOrder,
				  first_pSubstructuresNumericalValues);

  }
  return (structureFrame);
}
