/*-------------------------------------------------------

           List Mode Format 
                        
     --  makeRule.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of makeRule.c:
     Knowing the scanner topology, we build here a rule 
     to encode the detector ID (16 bits)
     The rule gives the number of bits to encode:
     rsectors, modules, submodules, crystals, layers
     
     Exemple of rule for a scanner:
     1111 0010 0000 0001
     
     that means 

     4 bits for rsector (so this scanner has 16 rsectors maximum)
     2 bits for modules
     1 bit for submodules
     8 bits for crystals
     1 bit for layer.
     
     The sum must always be 16.
     

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


u64 makeRule(const ENCODING_HEADER * pEncoH)
{
  u64 ruleID = 0;
  /* Number of bits for each part */
  u16 nbRS = 0, nbM = 0, nbSM = 0, nbC = 0, nbL = 0;
  u8 i;				/*counters */

  u8 scanEncodingIDLength;

  scanEncodingIDLength = poweru32(2, 4 + pEncoH->scanEncodingIDLength);

  nbRS = pEncoH->scanEncodingID.bitForRsectors;
  nbM = pEncoH->scanEncodingID.bitForModules;
  nbSM = pEncoH->scanEncodingID.bitForSubmodules;
  nbC = pEncoH->scanEncodingID.bitForCrystals;
  nbL = pEncoH->scanEncodingID.bitForLayers;

  if ((nbRS + nbM + nbSM + nbC + nbL) != scanEncodingIDLength)
    printf
	("\n\nWarning = The total bytes number is different than %d in makeRule function\n\n",
	 scanEncodingIDLength / 8);

  for (i = 0; i < nbRS; i++)
    ruleID |= (u64) (1) << (scanEncodingIDLength - 1 - i);	/*set 1, nbRS times */

  for (i = 0; i < nbSM; i++)
    ruleID |= (u64) (1) << (scanEncodingIDLength - 1 - nbRS - nbM - i);	/*set 1, nbSM times */

  for (i = 0; i < nbL; i++)
    ruleID |= (u64) (1) << (scanEncodingIDLength - 1 - nbRS - nbM - nbSM - nbC - i);	/*set 1, nbL times */

  return (ruleID);
}
