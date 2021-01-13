/*-------------------------------------------------------

List Mode Format 
                        
--  getEnergyStepFromCCH.c --

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2005 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of getEnergyStepFromCCH

Allows to retrieve the enegry step from CCH file


-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>

#include "lmf.h"

int getEnergyStepFromCCH()
{
  static u8 doneOnce = 0;
  static int energyStep = 0;

  i8 field[charNum];
  int cch_index = 0;

  if (!doneOnce) {
    strcpy(field, "energy step");

    for (cch_index = 0; cch_index <= last_cch_list; cch_index++) {
      if (strcasecmp(plist_cch[cch_index].field, field) == 0)
	break;
    }
    
    if (strcasecmp(plist_cch[cch_index].field, field) != 0)
      energyStep = GATE_LMF_ENERGY_STEP_KEV;
    else
      energyStep = (int) (plist_cch[cch_index].def_unit_value.vNum);
    
    doneOnce++;
  }

  return energyStep;
}
