/*-------------------------------------------------------

List Mode Format 
                        
--  changeAngularPos.c --

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2005 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of changeAngularPos

-------------------------------------------------------*/

#include <stdio.h>

#include "lmf.h"

#define TWO_PI 6.283185307

static i32 angularPos = 14;
static u16 maxAngle = 0;

void setNewAngularPos()
{
  char response[10];
  int index;
  float dist;
  float angular_step;
  float max;

  index = getLMF_cchInfo("azimuthal step");
  angular_step = plist_cch[index].def_unit_value.vNum * 360 / TWO_PI;
  dist = angularPos * angular_step;

  printf("angular_step = %2f dist =  %2f\n",angular_step, dist);


  printf("New fixed angular position (default: %2f degrees):",dist);
  if (fgets(response,sizeof(response),stdin))
    sscanf(response,"%f",&dist);

  angularPos = (i32) (dist / angular_step);
  max = 360 / angular_step;
  maxAngle = (max > 0) ? (u16) max : (u16) (-max); 

  printf("New angular pos: %.2f degrees -> %ld\n",dist,angularPos);
  printf("maxAngle = %d\n",maxAngle);
  //  getchar();

  return;
}

void changeAngularPos(EVENT_RECORD *pER)
{
  //  printf("angularPos = %d, maxAngle = %d: pER->gantryAngularPos = %d -> ",angularPos, maxAngle,  pER->gantryAngularPos);
  pER->gantryAngularPos = (u16)(((i32) pER->gantryAngularPos + angularPos)) % maxAngle;
  //  printf("%d\n",pER->gantryAngularPos);
  //  getchar();
  return;
}
