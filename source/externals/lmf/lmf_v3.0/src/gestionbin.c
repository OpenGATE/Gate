/*-------------------------------------------------------

           List Mode Format 
                        
     --  gestionbin.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of gestionbin.c:
     binary to decimal convertor and 
     decimal to binary convertor

-------------------------------------------------------*/
#include <stdio.h>
#include "lmf.h"

int bindec(i16 * a)
{
  int abis = 0;
  i16 tab[16];			/* To store a chain of 1 and 0 : 0111 1111... */

  int compt;
  int j, p2[16];		/* array of 2**0, 2**1, 2**2 ... */

  p2[0] = 1;
  tab[0] = 0;
  for (j = 1; j < 16; j++) {
    p2[(j)] = 2 * p2[(j - 1)];
    tab[0] = 0;
  }
  tab[8] = 1;

  for (compt = 0; compt < 16; compt++) {
    if (tab[compt] == 1) {

      abis = abis + p2[compt];
    }
  }
  return (abis);
}





void decbin(int a)
{
  int tam, cc;

  i16 tab[16];


  int j, p2[16];
  p2[0] = 1;
  tab[0] = 0;
  for (j = 1; j < 16; j++) {
    p2[(j)] = 2 * p2[(j - 1)];
    tab[0] = 0;
  }



  tam = a;



  for (cc = 0; cc < 16; cc++) {
    if (tam >= (p2[15 - cc])) {
      tab[15 - cc] = 1;
      tam = tam - p2[15 - cc];
    } else {

      tab[15 - cc] = 0;
    }
  }
  printf("\n");
  for (cc = 15; cc >= 0; cc--) {
    if (((cc + 1) % 4) == 0)
      printf("\t");

    printf("%d", tab[cc]);
  }

}


/* uncomment this main for example*/
/* main() */
/* { */
/*   decbin(4); */


/* } */
