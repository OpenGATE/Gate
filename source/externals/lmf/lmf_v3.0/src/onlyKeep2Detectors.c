/*-------------------------------------------------------

List Mode Format 
                        
--  onlyKeep2Detectors.c  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2005 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of onlyKeep2Detectors

Just a small function to keep only to detector head.
Used in the aim to do normalisation data for the
ClearPET of Lausanne.


-------------------------------------------------------*/


#include <stdio.h>
#include "lmf.h"

static u16 sector[2];
static u16 module[2];

void setDetector1(u16 sct, u16 mod)
{
  sector[0] = sct;
  module[0] = mod;

  return;
}

void setDetector2(u16 sct, u16 mod)
{
  sector[1] = sct;
  module[1] = mod;

  return;
}

void onlyKeep2Detectors(const ENCODING_HEADER * pEncoH,
			EVENT_RECORD ** ppER)
{
  u16 sct, mod;
  u16 *pcrist;

  pcrist = demakeid((*ppER)->crystalIDs[0], pEncoH);
  sct = pcrist[4];
  mod = pcrist[3];
  free(pcrist);

  if (!(((sct == sector[0]) && (mod == module[0])) ||
	((sct == sector[1]) && (mod == module[1]))))
    *ppER = NULL;
}
