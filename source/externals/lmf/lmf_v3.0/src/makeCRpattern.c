/*-------------------------------------------------------

           List Mode Format 
                        
     --  makeCRpattern.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of makeCRpattern.c:

  This function makes the encoding pattern for Count Rate record with the model :
  TTTT sSSc FrbR RRRR
  It needs a pointer to a countRateHeader Structure.
  It gives back an u16 (2 bytes) equal to the corresponding value of 
  TTTT sSSc FrbR RRRR
  
  TTTT = 1000
  RRR = 000
 
-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"




u16 makeCRpattern(const COUNT_RATE_HEADER * pCRH)
{

  u16 epCR = 0;

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   TTTT                 */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  epCR = 0;
  epCR |= BIT16;		/*TTTT = 1000 xxxx xx...     */


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   s                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pCRH->singleRateBool) {
  case (0):
    epCR &= (~BIT12);		/* mask AND 1111 0111 1111 1111 */
    break;
  case (1):
    epCR |= BIT12;		/* mask OR 0000 1000 0000 0000 */
    break;
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   SS                   */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pCRH->singleRatePart) {
  case (0):
    epCR &= (~(BIT10 + BIT11));	/* AND 1111 1001 1111 1111 */
    break;
  case (1):
    epCR |= BIT10;		/* OR 0000 0010 0000 0000 */
    break;
  case (2):
    epCR |= BIT11;		/* OR 0000 0100 0000 0000 */
    break;
  case (3):
    epCR |= (BIT10 + BIT11);	/* OR 0000 0110 0000 000 */
    break;
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   c                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pCRH->totalCoincidenceBool) {
  case (1):
    epCR |= BIT9;		/* OR 0000 0010 0000 0000 */
    break;
  case (0):
    epCR &= (~BIT9);		/* AND 1111 1101 1111 1111 */
    break;
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   F                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pCRH->totalRandomBool) {
  case (1):
    epCR |= BIT8;		/* OR 0000 0000 1000 0000 */
    break;
  case (0):
    epCR &= (~BIT8);		/* AND 1111 1111 0111 1111 */
    break;
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   r                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pCRH->angularSpeedBool) {
  case (1):
    epCR |= BIT7;		/* OR 0000 0000 0100 0000 */
    break;
  case (0):
    epCR &= (~BIT7);		/* AND 1111 1111 1011 1111 */
    break;
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   b                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pCRH->axialSpeedBool) {
  case (1):
    epCR |= BIT6;		/* OR 0000 0000 0010 0000 */
    break;
  case (0):
    epCR &= (~BIT6);		/* AND 1111 1111 1101 1111 */
    break;
  }


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   RRRRR                */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  epCR &= (~(BIT1 + BIT2 + BIT3 + BIT4 + BIT5));	/* AND 1111 1111 1110 0000 */

  /*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  return (epCR);

}
