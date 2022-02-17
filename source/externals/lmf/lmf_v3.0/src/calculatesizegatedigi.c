/*-------------------------------------------------------

           List Mode Format 
                        
     --  calculatesizegatedigi.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of calculatesizegatedigi.c:
     This function calculates (in bytes) the size 
     of 1 gate digi record. It needs the encoding gatedigi pattern.
     Exemple a = calculatesizegatedigi(gdp2);


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


u16 calculatesizegatedigi(u16 gdp2, u16 ep2)
{

  u16 sizegatedigi = 0, vnn;


  vnn = findvnn(ep2);

  /*run id */
  if ((gdp2 & BIT8) == BIT8)
    sizegatedigi = sizegatedigi + 4;
  /*event id */
  if ((gdp2 & BIT9) == BIT9) {
    sizegatedigi = sizegatedigi + 4;
    if ((ep2 & BIT12) == BIT12)	/* //if c=1 */
      sizegatedigi = sizegatedigi + 4;
  }

  if (((gdp2 & BIT6) == BIT6) && ((ep2 & BIT12) == BIT12))	// if M && C
  {
    sizegatedigi = sizegatedigi + 4;

  }



  /*source id */
  if ((gdp2 & BIT11) == BIT11) {
    sizegatedigi = sizegatedigi + 2;
    if ((ep2 & BIT12) == BIT12)	/* //if c=1 */
      sizegatedigi = sizegatedigi + 2;
  }

  /* source pos. */
  if ((gdp2 & BIT10) == BIT10) {
    sizegatedigi = sizegatedigi + 6;

    if ((ep2 & BIT12) == BIT12)	/*  //if c=1 */
      sizegatedigi = sizegatedigi + 6;
  }
  /* number of compton  */
  if ((gdp2 & BIT12) == BIT12) {
    sizegatedigi++;
  }
  /* number of compton  in detector */
  if ((gdp2 & BIT5) == BIT5) {
    sizegatedigi++;
  }


  /* global pos. */
  if ((gdp2 & BIT7) == BIT7) {
    sizegatedigi = sizegatedigi + 6;

    if ((ep2 & BIT12) == BIT12)	/* //if c=1 */
      sizegatedigi = sizegatedigi + 6;

    if ((ep2 & BIT9) == BIT9) {	/* // if n=1 */
      sizegatedigi = sizegatedigi + (6 * vnn);
      if ((ep2 & BIT12) == BIT12)	/* //if c=1 and n = 1 */
	sizegatedigi = sizegatedigi + (6 * vnn);
    }
  }



  return (sizegatedigi);

}
