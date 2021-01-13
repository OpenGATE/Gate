/*-------------------------------------------------------

List Mode Format 
                        
--  timeValueManager.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of timeValueManager.c:

This function takes a double and convert it in integer
the result is given as a u8 pointer to 8 
u8 from 0 to 7
uc[0] is the lowest value
warning : you need a compiler that accepts i64 !!!!

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lmf.h"
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
static u64 timeStepFromCCH = 0;

u64 getTimeStepFromCCH(void)	// returns time step in femtoseconds (u64)
{
  /* 
     These next lines are used to extract time step from .cch
     The time is supposed to be in picosecond.

   */
  static int doneOnce = FALSE;
  static i8 field[charNum];
  static int cch_index = 0;
  if (doneOnce == FALSE) {

    strcpy(field, "clock time step");
    cch_index = getLMF_cchInfo(field);
    timeStepFromCCH =
	(u64) (plist_cch[cch_index].def_unit_value.vNum * 1000);

    //  printf("in time manager value=%e %s\n", plist_cch[cch_index].def_unit_value.vNum,plist_cch[cch_index].def_unit);
    printf("in time manager value=%.3f %s\n",
	   (double) timeStepFromCCH / 1000, plist_cch[cch_index].def_unit);
    doneOnce = TRUE;
    if (timeStepFromCCH == 0) {
      printf
	  ("***error : timeValueManager.c : time Step is null or less than 1 picosecond\n");
      exit(0);
    }
  }
  return (timeStepFromCCH);
}

/*******************************************************

                     getTimeOfThisEVENT

                      Returns the time
                   of a single EVENT_RECORD
               This time is stores on 8 bytes
                   in LMF. We have to use here
                 the time step stored in .CCH


********************************************************/
u64 getTimeOfThisEVENT(const EVENT_RECORD * pER) /* return time in pico-seconds */
{
  u64 eventTime;
  if (pER == NULL)
    eventTime = 0;
  else
    eventTime = (u64)((double)(timeStepFromCCH * u8ToU64(pER->timeStamp)) / 1000. + 0.5);
  return eventTime;
}


u64 getTimeOfThisCOINCI(const EVENT_RECORD * pER) /* return time in milli-seconds */
{
  u64 timeInMillis;
  if (pER == NULL)
    timeInMillis = 0;
  else
    timeInMillis = (pER->timeStamp[2] +
		    256 * pER->timeStamp[1] +
		    256 * 256 * pER->timeStamp[0]);

  return timeInMillis;
}


float getTimeOfFlightOfThisCOINCI(const EVENT_RECORD * pER) /* return time in nano-seconds */
{
  float tof;
  if (pER == NULL)
    tof = 0;
  else
    tof = (float)(pER->timeOfFlight) * CONVERT_TOF / 1E6;

  return tof;
}

u64 u8ToU64(const u8 * pc)
{
  u64vsu8 dc;

  memcpy(dc.w8, pc, 8);

  return (dc.w64);
}


double u8ToDouble(u8 * pc)
{
  double dd;
  u64vsu8 dc;
  memcpy(dc.w8, pc, 8);
  dd = (double) dc.w64;
  return (dd);
}

u8 *u64ToU8(const u64 ulli)
{
  static u64vsu8 dc;
  dc.w64 = ulli;
  return (dc.w8);
}

u8(*doubleToU8(double x))
{
  static u64vsu8 dc;
  dc.w64 = (u64) (x + 0.5);
  return (dc.w8);
}

u64 absullidiff(u64 a, u64 b)
{
  static i64 diff;
  diff = ((i64) a) - ((i64) b);
  if (diff >= 0)
    return (diff);
  else
    return (-diff);
}

void time38bitShifter(u8 * tmp, u32 shift)
{
  t32 value;

  memcpy(value.u8Tab, tmp, 8);
  value.u32Tab[1] += (shift << 6);
  memcpy(tmp, value.u8Tab, 8);
}


u32 getStrongBit(u8 * stamps)
{
  t32 tmp;

  memcpy(tmp.u8Tab, stamps, 8);

  return tmp.u32Tab[1];
}

/*   H O W  T O  U S E  I T !!!! */
/*   Uncomment this main !!! */
/* main() */
/* { */
/*   int i; */
/*   u32 *uli; */
/*   double x = 1024.23,y = 0; */
/*   u8 *ac; */
/*   u64 z; */
/*   ac = doubleToChar(x); */
/*   for (i=7;i>=0;i--) */
/*     printf("\t%d",ac[i]); */
/*   printf("\n"); */
/*   ac[7] = 1; */
/*   ac[6] = 1; */
/*   ac[5] = 0; */
/*   ac[4] = 1; */
/*   ac[0] = 1; */
/*   ac[1] = 1; */
/*   ac[2] = 0; */
/*   ac[3] = 0; */
/*   uli =(u32*) ac; */
/*  for (i=7;i>=0;i--) */
/*     printf("\t%d",ac[i]); */

/*   y = i8ToDouble(ac); */
/*   printf("\ny = %f\n",y); */
/*   z = i8ToLongLong(ac); */
/*   printf("\nz  = %d %d \n",z); */


/*   printf("\nuli = %d \t %d\n",uli[0],uli[1]); */

/*   printulli(z); */

/* }	 */
