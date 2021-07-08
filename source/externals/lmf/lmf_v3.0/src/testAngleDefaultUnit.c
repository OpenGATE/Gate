/*-------------------------------------------------------

           List Mode Format 
                        
     --  testAngleDefaultUnit.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of testAngleDefaultUnit.c:


	Find the 3D coordinates of an event in the scanner coordinates (x,y,z) system
	-> testAngleDefaultUnit - Test if the angle default unit is the radian
-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>		/*EXIT_SUCCESS & EXIT_FAILURE */
#include "lmf.h"

/**** testAngleDefaultUnit - Test if the angle default unit is the radian ****/

double testAngleDefaultUnit()
{

  i8 angleDefaultUnitList[4][charNum] = { "rad", "deg", "grad", "rp" };
  double angleDefaultUnitToRadConversionFactor[4] = { 1,
    1.745329252E-2,
    1.570796327E-2,
    6.283185307
  };
  int list_index = 0;
  i8 angleDefaultUnit[charNum];

  initialize(angleDefaultUnit);
  strcpy(angleDefaultUnit, ANGLE_UNIT);
  for (list_index = 0; list_index < 4; list_index++) {
    if (strncasecmp(angleDefaultUnit, angleDefaultUnitList[list_index], 2)
	== 0)
      return (angleDefaultUnitToRadConversionFactor[list_index]);
  }
  return (0);
}
