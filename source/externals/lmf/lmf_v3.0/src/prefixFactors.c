/*-------------------------------------------------------

           List Mode Format 
                        
     --  prefixFactors.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of prefixFactors.c:
	 Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->prefixFactors - Define an array which contains all the factors
	                   used to convert the prefix unit in the default prefix unit

-------------------------------------------------------*/


#include <stdio.h>
#include "lmf.h"



#include "lmf.h"

i8 standard_prefix_list[16][3] = { "a", "f", "p", "n", "mu", "m", "c", "d",
  "", "da", "h", "k", "M", "G", "T", "P"
};

double standard_prefix_factors_list[16][16] =
    { {1, 1E-3, 1E-6, 1E-9, 1E-12, 1E-15, 1E-16, 1E-17,
       1E-18, 1E-19, 1E-20, 1E-21, 1E-24, 1E-27, 1E-30, 1E-33},
{1E3, 1, 1E-3, 1E-6, 1E-9, 1E-12, 1E-13, 1E-14, 1E-15,
 1E-16, 1E-17, 1E-18, 1E-21, 1E-24, 1E-27, 1E-30},
{1E6, 1E3, 1, 1E-3, 1E-6, 1E-9, 1E-10, 1E-11, 1E-12, 1E-13,
 1E-14, 1E-15, 1E-18, 1E-21, 1E24, 1E-27},
{1E9, 1E6, 1E3, 1, 1E-3, 1E-6, 1E-7, 1E-8, 1E-9, 1E-10, 1E-11,
 1E-12, 1E-15, 1E-18, 1E-21, 1E24},
{1E12, 1E9, 1E6, 1E3, 1, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7, 1E-8,
 1E-9, 1E-12, 1E-15, 1E-18, 1E-21},
{1E15, 1E12, 1E9, 1E6, 1E3, 1, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5,
 1E-6, 1E-9, 1E-12, 1E-15, 1E-18},
{1E16, 1E13, 1E10, 1E7, 1E4, 1E1, 1, 1E-1, 1E-2, 1E-3, 1E-4,
 1E-5, 1E-8, 1E-11, 1E-14, 1E-17},
{1E17, 1E14, 1E11, 1E8, 1E5, 1E2, 1E1, 1, 1E-1, 1E-2, 1E-3,
 1E-4, 1E-7, 1E-10, 1E-13, 1E-16},
{1E18, 1E15, 1E12, 1E9, 1E6, 1E3, 1E2, 1E1, 1, 1E-1, 1E-2,
 1E-3, 1E-6, 1E-9, 1E-12, 1E-15},
{1E19, 1E16, 1E13, 1E10, 1E7, 1E4, 1E3, 1E2, 1E1, 1, 1E-1,
 1E-2, 1E-5, 1E-8, 1E-11, 1E-14},
{1E20, 1E17, 1E14, 1E11, 1E8, 1E5, 1E4, 1E3, 1E2, 1E1, 1,
 1E-1, 1E-4, 1E-7, 1E-10, 1E-13},
{1E21, 1E18, 1E15, 1E12, 1E9, 1E6, 1E5, 1E4, 1E3, 1E2, 1E1, 1,
 1E-3, 1E-6, 1E-9, 1E-12},
{1E24, 1E21, 1E18, 1E15, 1E12, 1E9, 1E8, 1E7, 1E6, 1E5, 1E4,
 1E3, 1, 1E-3, 1E-6, 1E-9},
{1E27, 1E24, 1E21, 1E18, 1E15, 1E12, 1E11, 1E10, 1E9, 1E8,
 1E7, 1E6, 1E3, 1, 1E-3, 1E-6},
{1E30, 1E27, 1E24, 1E21, 1E18, 1E15, 1E14, 1E13, 1E12, 1E11,
 1E10, 1E9, 1E6, 1E3, 1, 1E-3},
{1E33, 1E30, 1E27, 1E24, 1E21, 1E18, 1E17, 1E16, 1E15, 1E14,
 1E13, 1E12, 1E9, 1E6, 1E3, 1}
};
