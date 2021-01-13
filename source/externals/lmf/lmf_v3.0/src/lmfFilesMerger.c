/*-------------------------------------------------------

List Mode Format 
                        
--  lmfFilesMerger.c  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2004 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of lmfFilesMerger

set a new name for the lmf file which will contain
several lmf files.


-------------------------------------------------------*/


#include <stdio.h>
#include "lmf.h"
#include <string.h>

static i8 newFileName[charNum];

void setNewLMFfileName(i8 newName[charNum])
{
  initialize(newFileName);
  strcpy(newFileName, newName);
  copyNewCCHfile(newFileName);
  strcat(newFileName, ".ccs");
}

void mergeLMFfiles(const ENCODING_HEADER * pEncoH,
		   const EVENT_HEADER * pEH,
		   const COUNT_RATE_HEADER * pCRH,
		   const GATE_DIGI_HEADER * pGDH,
		   const CURRENT_CONTENT * pcC,
		   const EVENT_RECORD * pER,
		   const COUNT_RATE_RECORD * pCRR, FILE ** ppfile)
{
  LMFbuilder(pEncoH, pEH, pCRH, pGDH, pcC, pER, pCRR, ppfile, newFileName);
}
