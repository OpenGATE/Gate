/*-------------------------------------------------------

List Mode Format 
                        
--  timeOrder.hh  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2005 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of timeOrder class

class which allows to choose the EVENT_RECORD one of two
with an earlier time.


-------------------------------------------------------*/

#ifndef TIMEORDER_HH
#define TIMEORDER_HH

#include <iostream>
#include <stdio.h>
#include "lmf.h"

class timeOrder {
public:
  int operator() (EVENT_RECORD * x, EVENT_RECORD * y) const {
    return (u8ToU64(x->timeStamp) < u8ToU64(y->timeStamp));
  };
};

#endif
