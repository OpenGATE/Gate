/*-------------------------------------------------------

           List Mode Format 
                        
     --  makeEpattern.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of makeEpattern.c:
  This function makes the encoding pattern for event record with the model :
  TTTT cdEn NNgb sRRR
  It needs a pointer to an eventHeader Structure.
  It gives back an u16 (2 bytes) equal to the corresponding value of 
  TTTT cdEn NNgb sGZR
  
  TTTT = 0000
  R = 0
 -------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"



u16 makeEpattern(const EVENT_HEADER * pEH)
{

  u16 epE = 0, testNN = 0;

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   TTTT                 */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  epE &= 4095;			/*TTTT = 0000 xxxx xx...     */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   c                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pEH->coincidenceBool) {
  case (0):
    epE &= (~BIT12);		/* mask AND 1111 0111 1111 1111 */
    break;
  case (1):
    epE |= BIT12;		/* mask OR 0000 1000 0000 0000 */
    break;
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   d                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pEH->detectorIDBool) {
  case (1):
    epE |= BIT11;		/* OR 0000 0100 0000 0000 */
    break;
  case (0):
    epE &= (~BIT11);		/* AND 1111 1011 1111 1111 */
    break;
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   E                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pEH->energyBool) {
  case (1):
    epE |= BIT10;		/* OR 0000 0010 0000 0000 */
    break;
  case (0):
    epE &= (~BIT10);		/* AND 1111 1101 1111 1111 */
    break;
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   n                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pEH->neighbourBool) {
  case (1):
    epE |= BIT9;		/* OR 0000 0001 0000 0000 */
    break;
  case (0):
    epE &= (~BIT9);		/* AND 1111 1110 1111 1111 */
    epE &= (~BIT7ET8);		/* AND 1111 1111 0011 1111     */
    testNN = 1;
    break;
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   NN                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  if (testNN != 1) {
    switch (pEH->neighbourhoodOrder) {
    case (0):
      epE &= (~(BIT7 + BIT8));	/* AND 1111 1111 0011 1111 */
      break;
    case (1):
      epE |= BIT7;		/* OR 0000 0000 0100 0000 */
      break;
    case (2):
      epE |= BIT8;		/* OR 0000 0000 1000 0000 */
      break;
    case (3):
      epE |= (BIT7 + BIT8);	/* OR 0000 0000 1100 0000 */
      break;
    }
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   g                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pEH->gantryAngularPosBool) {
  case (1):
    epE |= BIT6;		/* OR 0000 0000 0010 0000 */
    break;
  case (0):
    epE &= (~BIT6);		/* AND 1111 1111 1101 1111 */
    break;
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   b                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pEH->gantryAxialPosBool) {
  case (1):
    epE |= BIT5;		/* OR 0000 0000 0001 0000 */
    break;
  case (0):
    epE &= (~BIT5);		/* AND 1111 1111 1110 1111 */
    break;
  }


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   s                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pEH->sourcePosBool) {
  case (1):
    epE |= BIT4;		/* OR 0000 0000 0000 1000 */
    break;
  case (0):
    epE &= (~(BIT4));		/* AND 1111 1111 1111 0111 */
    break;
  }


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   G                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pEH->gateDigiBool) {
  case (1):
    epE |= BIT3;		/* OR 0000 0000 0000 0100 */
    break;
  case (0):
    epE &= (~(BIT3));		/* AND 1111 1111 1111 1011 */
    break;
  }


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   Z                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  switch (pEH->fpgaNeighBool) {
  case (1):
    epE |= BIT2;		/* OR 0000 0000 0000 0010 */
    break;
  case (0):
    epE &= (~(BIT2));		/* AND 1111 1111 1111 1011 */
    break;
  }



  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   R                   */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  epE &= (~(BIT1));		/* AND 1111 1111 1111 1000 */





  return (epE);

}
