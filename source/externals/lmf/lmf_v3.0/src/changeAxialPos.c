/*-------------------------------------------------------

List Mode Format 
                        
--  changeAxialPos.c --

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2005 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of changeAxialPos

-------------------------------------------------------*/

#include <stdio.h>

#include "lmf.h"

static u16 axialPos = 1;

void setNewAxialPos()
{
  char response[10];
  int index;
  float dist;
  float axial_step;

  index = getLMF_cchInfo("axial step");
  axial_step = plist_cch[index].def_unit_value.vNum;
  dist = axialPos * axial_step;

  printf("New fixed axial position (default: %.2f mm):",dist);
  if (fgets(response,sizeof(response),stdin))
    sscanf(response,"%f",&dist);

  axialPos = (u16) (dist / axial_step);

  printf("New axial pos: %.2f mm -> %d\n",dist,axialPos);

  return;
}

void changeAxialPos(EVENT_RECORD *pER)
{
  pER->gantryAxialPos = axialPos;

  return;
}
