/*-------------------------------------------------------

           List Mode Format 
                        
     --  makeGDpattern.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of makeGDpattern.c:

  This function makes the encoding pattern for gate digi record with the model :
  TTTT CSpe rGMD RRRR
  It needs a pointer to  a gateDigiHeader Structure.
  It gives back an u16 (2 bytes) equal to the corresponding value of 
  
  TTTT = 1100
  RRR = 00 0000
  

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"



u16 makeGDpattern(const GATE_DIGI_HEADER * pGDH)
{

  u16 epGD = 0;

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*             TTTT = 1100                */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  epGD |= BIT16;		/* // OR  1000 0... */
  epGD |= BIT15;		/* // OR  0100 0... */
  epGD &= ~BIT14;		/* // AND 1101 1... */
  epGD &= ~BIT13;		/* // AND 1110 1... */


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   C                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pGDH->comptonBool) {
  case (1):
    epGD |= BIT12;
    break;
  case (0):
    epGD &= (~BIT12);
    break;
  }
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   S                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pGDH->sourceIDBool) {
  case (1):
    epGD |= BIT11;
    break;
  case (0):
    epGD &= (~BIT11);
    break;
  }


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   p                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pGDH->sourceXYZPosBool) {
  case (1):
    epGD |= BIT10;
    break;
  case (0):
    epGD &= (~BIT10);
    break;
  }




  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   e                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pGDH->eventIDBool) {
  case (1):
    epGD |= BIT9;
    break;
  case (0):
    epGD &= (~BIT9);
    break;
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   r                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pGDH->runIDBool) {
  case (0):
    epGD &= (~BIT8);
    break;
  case (1):
    epGD |= BIT8;
    break;
  }

  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   G                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pGDH->globalXYZPosBool) {
  case (1):
    epGD |= BIT7;
    break;
  case (0):
    epGD &= (~(BIT7));
    break;
  }


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   M                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pGDH->multipleIDBool) {
  case (1):
    epGD |= BIT6;
    break;
  case (0):
    epGD &= (~(BIT6));
    break;
  }


  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*                   D                    */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  switch (pGDH->comptonDetectorBool) {
  case (1):
    epGD |= BIT5;

    break;
  case (0):
    epGD &= (~(BIT5));
    break;
  }






  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
  /*       RRRR =  0000                   */
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


  return (epGD);

}
