/*-------------------------------------------------------

List Mode Format 
                        
--  DOImisIdentification.c  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2006 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of DOImisIdentification


-------------------------------------------------------*/


#include <stdio.h>
#include "lmf.h"

static double misIDfract = 0;
static u32 lso = 0, luyap = 0, tot = 0;

void setInitialParamForDOImisID(double misID_fraction)
{
  misIDfract = misID_fraction;

  if((misIDfract > 1) || (misIDfract < 0)) {
    printf("Fraction must be a number between 0 and 1\n\t-> EXIT\n");
    exit(EXIT_FAILURE);
  }
  printf("The DOI misidentification fraction set to %.0f %%\n",misIDfract*100);

  return;
}

void DOImisID(const ENCODING_HEADER *pEncoH, EVENT_RECORD *pER)
{
  double random;
  u16 *pcrist;
  u16 sct, mod, smd, cry, lay;
  u16 errFlag = 0;

  random = randd();

  if(random < misIDfract) {
    pcrist = demakeid(pER->crystalIDs[0], pEncoH);
    lay = pcrist[0];
    cry = pcrist[1];
    smd = pcrist[2];
    mod = pcrist[3];
    sct = pcrist[4];
    free(pcrist);
    if(lay)
      luyap++;
    else
      lso++;
    lay = (lay) ? 0 : 1;
    pER->crystalIDs[0] = makeid(sct,mod,smd,cry,lay,pEncoH, &errFlag);

    if(errFlag)
      exit(EXIT_FAILURE);
  }

  tot++;
  return;
}

void DOImisIDdestructor()
{
  printf("%.2f %% of LSO events were changed in LuYAP events\n",
	 (double)(lso)/tot*100);
  printf("%.2f %% of LuYAP events were changed in LSO events\n",
	 (double)(luyap)/tot*100);
}
