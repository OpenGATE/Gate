/*-------------------------------------------------------

           List Mode Format 
                        
     --  demakeRule.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of demakeRule.c:
     Finds the rule for detector ID  (u16) :
     Ex if rule is 1110 0001 1100 0001 <=> 57793 
     it gives 3 4 3 5 1 in an array of u8
     (counts the successives 1 and 0)
     That means that the detector ID rule is:
     3 bits for rsector ID
     4 bits for module ID
     3 bits for submodule 
     5 bits for crystal ID
     1 bits for layer ID

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>


#include "lmf.h"

void demakeRule(ENCODING_HEADER * pEncoH, u64 codedRule)
{
  u8 *rule;
  int i = 0;
  u8 firstBit;

  if ((rule = malloc(5 * sizeof(u8))) == NULL)
    printf("\n***ERROR : in demakeRule.c : impossible to do : malloc()\n");

  rule[0] = rule[1] = rule[2] = rule[3] = rule[4] = 0;

  i = poweru32(2, 4 + pEncoH->scanEncodingIDLength) - 1;

  firstBit = codedRule >> i;
  while (codedRule >> i == firstBit) {
    if (i < 0)
      break;
    rule[4]++;
    if(firstBit)
      codedRule -= (codedRule >> i) << i;
    i--;
  }

  firstBit = codedRule >> i;
  while (codedRule >> i == firstBit) {
    if (i < 0)
      break;
    rule[3]++;
    if(firstBit)
      codedRule -= (codedRule >> i) << i;
    i--;
  }

  firstBit = codedRule >> i;
  while (codedRule >> i == firstBit) {
    if (i < 0)
      break;
    rule[2]++;
    if(firstBit)
      codedRule -= (codedRule >> i) << i;
    i--;
  }

  firstBit = codedRule >> i;
  while (codedRule >> i == firstBit) {
    if (i < 0)
      break;
    rule[1]++;
    if(firstBit)
      codedRule -= (codedRule >> i) << i;
    i--;
  }

  firstBit = codedRule >> i;
  while (codedRule >> i == firstBit) {
    if (i < 0)
      break;
    rule[0]++;
    if(firstBit)
      codedRule -= (codedRule >> i) << i;
    i--;
  }

  pEncoH->scanEncodingID.bitForRsectors = rule[4];	/* and fill the concerning structures */
  pEncoH->scanEncodingID.maximumRsectors = poweru32(2, rule[4]);	/* for rings and sectors */
  pEncoH->scanEncodingID.bitForModules = rule[3];
  pEncoH->scanEncodingID.maximumModules = poweru32(2, rule[3]);	/* for modules */
  pEncoH->scanEncodingID.bitForSubmodules = rule[2];
  pEncoH->scanEncodingID.maximumSubmodules = poweru32(2, rule[2]);	/*for submodules */
  pEncoH->scanEncodingID.bitForCrystals = rule[1];
  pEncoH->scanEncodingID.maximumCrystals = poweru32(2, rule[1]);	/* for crystals  */
  pEncoH->scanEncodingID.bitForLayers = rule[0];

  if (rule[0])
    pEncoH->scanEncodingID.maximumLayers = poweru32(2, rule[0]);	/* for layers */
  else
    pEncoH->scanEncodingID.maximumLayers = 0;

  free(rule);
}
