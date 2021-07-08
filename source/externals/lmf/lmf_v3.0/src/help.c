/*-------------------------------------------------------

           List Mode Format 
                        
     --  help.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of help.c:

     Displays online help

-------------------------------------------------------*/
#include <stdio.h>

void helpForStackCutTime()
{
  printf("\n\n\n");
  printf("* * * * * * * *  L M F    H E L P * * * * * * * * *\n");
  printf("*                                                 *\n");
  printf("* Stack cut time :                                *\n");
  printf("* The coincidence sorter stores singles in a list *\n");
  printf("* For each new single event, the list becomes     *\n");
  printf("* bigger and bigger. So we have chosen to remove  *\n");
  printf("* the oldest singles as soon as the difference    *\n");
  printf("* between time stamp of these singles and time    *\n");
  printf("* stamp of latest singles is higher than a value  *\n");
  printf("* called Stack Cut Time. If you choose a high     *\n");
  printf("* SCT value, the computation time increases,      *\n");
  printf("* but if you choose a too low value, you take the *\n");
  printf("* risk to miss some coincidences.                 *\n");
  printf("*                                                 *\n");
  printf("* * * * * * * * * * * * * * * * * * * * * * * * * *\n");
  printf("\n");

}

void helpExample1(void)
{

  printf("\n");
  printf("****************************************\n");
  printf("        \n");
  printf("             Example 1   \n");
  printf("    Creation of your first LMF file\n");
  printf(" \n");
  printf(" This example allows you to create a complete\n");
  printf(" binary LMF file, with .ccs extension. The values\n");
  printf(" of generated records are not pertinent, but\n");
  printf(" it is just a pedagogic example. The output of\n");
  printf(" this example is called test1_ex1.ccs\n");
  printf(" You can find in this directory test1_ex1.cch,\n");
  printf(" the associated ascii file. You can read and/or process \n");
  printf(" this couple of files with Example 4.\n");

  printf("\n\n");

}
