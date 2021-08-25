/*-------------------------------------------------------

           List Mode Format 
                        
     --  monteCarloEngine.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of monteCarloEngine.c:
     generate a random float number between 0 & 1


-------------------------------------------------------*/

#include <stdio.h>
#include "lmf.h"


double randd()
{
  double aleac;

  aleac = (double)rand();
  aleac = aleac / RAND_MAX;
  return aleac;
}

/*
generate an u16 between a and b (include)
*/
u16 monteCarloInt(u16 a, u16 b)
{
  u16 value;
  static u16 A, B;
  A = a;
  B = b + 1;
  value = ((u16) (randd() * (B - A)) + A);


  return (value);

}

/* main() */
/* { */
/*   u16 mc; */
/*   int i; */
/*   int a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10; */
/*   a0=a1=a2=a3=a4=a5=a6=a7=a8=a9=a10 = 0; */

/*   for (i=0;i<100000;i++) */
/*     { */
/*       //   printf("mc = %d\n",mc=monteCarloInt(0,10)); */
/*       mc=monteCarloInt(0,10); */
/*       if(mc==0) a0++; */
/*       if(mc==1) a1++; */
/*       if(mc==2) a2++; */
/*       if(mc==3) a3++; */
/*       if(mc==4) a4++; */
/*       if(mc==5) a5++; */
/*       if(mc==6) a6++; */
/*       if(mc==7) a7++; */
/*       if(mc==8) a8++; */
/*       if(mc==9) a9++; */
/*       if(mc==10) a10++; */


/*     }     */

/*   printf("a0 = %d\n",a0); */
/*   printf("a1 = %d\n",a1); */
/*   printf("a2 = %d\n",a2); */
/*   printf("a3 = %d\n",a3); */
/*   printf("a4 = %d\n",a4); */
/*   printf("a5 = %d\n",a5); */
/*   printf("a6 = %d\n",a6); */
/*   printf("a7 = %d\n",a7); */
/*   printf("a8 = %d\n",a8); */
/*   printf("a9 = %d\n",a9); */
/*   printf("a10 = %d\n",a10); */

/* } */
