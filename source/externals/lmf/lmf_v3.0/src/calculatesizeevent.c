/*-------------------------------------------------------

           List Mode Format 
                        
     --  calculatesizeevent.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of calculatesizeevent.c:
   This function calculate (in bytes) the size 
   of 1 event. It needs the encoding event pattern.
    Exemple a = calculatesizeevent(ep2);


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


u16 calculatesizeevent(u16 ep2)
{

  u16 sizeevent = 0, vnn;


  vnn = findvnn(ep2);

  /* TAG + TIME STAMP +  TOF */
  if ((ep2 & BIT12) == 0)	/* check bit 12 in TTTT cdEn NNgb sRRRR */
    sizeevent = sizeevent + 8;	/* if bit 12 = 0 64 bits for tag (1) & time (63) */
  else
    sizeevent = sizeevent + 4;	/* if bit 12 = 1 32 bits for tag (1) & time (23) & tof (8) */
  /* id of 1st */
  if ((ep2 & BIT11) == BIT11)	/* check bit 11 in TTTT cdEn NNgb sRRRR */
    sizeevent = sizeevent + 2;
  /* id of 2nd */
  if (((ep2 & BIT11) == BIT11) && ((ep2 & BIT12) == BIT12))
    sizeevent = sizeevent + 2;
  /* gantry ang pos. */
  if ((ep2 & BIT6) == BIT6)
    sizeevent = sizeevent + 2;
  /* gantry axi pos. */
  if ((ep2 & BIT5) == BIT5)
    sizeevent = sizeevent + 2;
  /* source pos. */
  if ((ep2 & BIT4) == BIT4)
    sizeevent = sizeevent + 4;
  /* Energy of 1 */
  if ((ep2 & BIT10) == BIT10)
    sizeevent = sizeevent + 1;
  /* Energy of neigh of 1 */
  if ((ep2 & BIT9) == BIT9)
    sizeevent = sizeevent + vnn;
  /* Energy of 2nd */
  if (((ep2 & BIT10) == BIT10) && ((ep2 & BIT12) == BIT12))
    sizeevent = sizeevent + 1;
  /* Energy of neigh of 2nd */
  if (((ep2 & BIT9) == BIT9) && ((ep2 & BIT12) == BIT12))
    sizeevent = sizeevent + vnn;
  /* fpga neigh pos  */
  if (((ep2 & BIT2) == BIT2))
    sizeevent = sizeevent + 1;
  /*fpga neigh pos of 2nd */
  if (((ep2 & BIT2) == BIT2) && ((ep2 & BIT12) == BIT12))
    sizeevent = sizeevent + 1;

  return (sizeevent);

}
