/*-------------------------------------------------------

           List Mode Format 
                        
     --  units.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of units.c:

	 Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->prefixFactors - Define an array which contains all the units
	                   used to convert a value in an default value
-------------------------------------------------------*/


#include <stdio.h>
#include "lmf.h"

i8 symbol_units_list[20][5][7] = { {"", "", "", "", ""}
,				/*data isn't defined */
{"", "", "", "", ""}
,				/*data = a sting */
{"", "", "", "", ""}
,				/*data = value without unit */
{"", "", "", "", ""}
,				/*data = a date */
{"h", "min", "s", "", ""}
,				/*data = a duration */
{"h", "min", "s", "", ""}
,				/*data = a time */
{"", "", "", "", ""}
,				/*data = a surface */
{"", "", "", "", ""}
,				/*data = a volume */
{"", "", "", "", ""}
,				/*data = a speed */
{"", "", "", "", ""}
,				/*data = a rotation speed */
{"eV", "J", "Wh", "cal", "erg"}
,				/*data = an energy */
{"Bq", "Ci", "", "", ""}
,				/*data = an activity */
{"g", "oz", "lb", "", ""}
,				/*data = a weight */
{"m", "in", "ft", "", ""}
,				/*data = a distance but not a shift value */
{"degree", "rad", "grad", "rp", ""}
,				/*data = an angle */
{"C", "F", "K", "", ""}
,				/*data = a temperature */
{"V", "", "", "", ""}
,				/*data = an electric field */
{"gauss", "T", "", "", ""}
,				/*data = a magnetic field */
{"Pa", "atm", "bar", "mmHg", ""}
,				/*data = a pression */
{"m", "in", "ft", "", ""}
};				/*data = a shift value */
int nb_units[20] =
    { 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 5, 2, 3, 3, 4, 3, 1, 2, 4, 3 };
