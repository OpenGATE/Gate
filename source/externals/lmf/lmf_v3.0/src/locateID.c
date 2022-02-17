/*-------------------------------------------------------

           List Mode Format 
                        
     --  locateID.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of locateID.c:

	Find the 3D coordinates of an event in the scanner coordinates (x,y,z) system
	->locateID - Define the crystal or the submodule or the module 
	             or the rsector tangential and axial ID
	->remakeID - Define crystalID or submoduleID or moduleID 
	             or rsectorID with tangential and axial ID
-------------------------------------------------------*/



#include <stdio.h>
#include "lmf.h"

/**** locateID - Define the crystal or the submodule or the module 
      or the rsector tangential and axial ID ****/

generalSubstructureID locateID(u16 * pcrist,
			       int substructureOrder,
			       double *first_pSubstructuresNumericalValues)
{

  generalSubstructureID localSubstructureID = { 0 };
  double *pSubstructuresNumericalValues = NULL;

  pSubstructuresNumericalValues = first_pSubstructuresNumericalValues + 15;

  localSubstructureID.tangential = (int) pcrist[substructureOrder]
      - (((int) pcrist[substructureOrder]
	  / (int) pSubstructuresNumericalValues[substructureOrder])
	 * (int) pSubstructuresNumericalValues[substructureOrder]);

  localSubstructureID.axial = (int) pcrist[substructureOrder] /
      (int) pSubstructuresNumericalValues[substructureOrder];

  return (localSubstructureID);
}


/** remakeID - Define crystalID or submoduleID or moduleID or rsectorID with tangential and axial ID **/

u16 remakeID(generalSubstructureID substructureID,
	     int substructureOrder,
	     double *first_pSubstructuresNumericalValues)
{
  u16 newID;
  double *pSubstructuresNumericalValues = NULL;

  pSubstructuresNumericalValues = first_pSubstructuresNumericalValues + 15;

  newID = substructureID.tangential
      +
      (substructureID.axial *
       (int) pSubstructuresNumericalValues[substructureOrder]);

  return (newID);
}
