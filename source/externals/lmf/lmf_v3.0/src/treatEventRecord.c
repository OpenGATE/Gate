/*-------------------------------------------------------

List Mode Format 
                        
--  treatEventRecord.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of treatEventRecord.c:

If you want to treat a data of the event record
(exemple, to "cut" energy) you can treat them here
before to copy it with the "treat and copy" option
of the LMF reader.

-------------------------------------------------------*/

#include <stdio.h>

#include "lmf.h"



void treatEventRecord(const ENCODING_HEADER * pEncoH,
		      const EVENT_HEADER * pEH,
		      const GATE_DIGI_HEADER * pGDH, EVENT_RECORD ** ppER)
{
  int keepIT = TRUE;		/*  = FALSE if we dont keep this event */

  /*  u16 *pcrist;  to demake id
     int verboseLevel = 0;
     int timeCoinci;
     u64 timeSingle;  */


  /*                Time :                
     Be careful the first of 8 bit of timeStamp[0]
     must stay to 0 for an event record.

     For a single, the time step is on 8 bytes,
     and can be send in an u64 (64bits)
     with the getTimeOfThisEVENT function
     For a coincidence, time is on 3 bytes.
     It can be send to an integer
     by 256*256*tS[0]+256*tS[1]+tS[2]
     where


   */
  keepIT = TRUE;

  if (deadTimeMgr(pEncoH, pEH, pGDH, *ppER))
    keepIT = TRUE;
  else
    keepIT = FALSE;

  /*     if(pEH->coincidenceBool == FALSE)    time of a SINGLE EVENT */
  /*     { */
  /*       timeSingle = getTimeOfThisEVENT(pER); */
  /*       if(verboseLevel) */
  /*    printf("old time = %llu\t",timeSingle); */

  /*
     put here the treatement for time of singles...
   */


  /*       timeSingle = timeSingle + 1; */
  /*       timeSingle = timeSingle - 1; */

  /*       if(verboseLevel) */
  /*    printf("new time = %llu\n",timeSingle); */

  /*    } */
  /*    else                  time of a COINCIDENCE EVENT */
  /*     { */

  /*       timeCoinci =256*256*pER->timeStamp[0]+256*pER->timeStamp[1]+pER->timeStamp[2]; */
  /*      if(verboseLevel) */
  /*    printf("old time = %d\t",timeCoinci); */
  /*           */
  /*       */
  /*    put here the treatement for time of coincidences... */
  /*       */


  /*         timeCoinci = timeCoinci + 1; */
  /*      timeCoinci = timeCoinci - 1; */

  /*      if(verboseLevel) */
  /*    printf("new time = %d\n",timeCoinci); */
  /*    */
  /*      if(verboseLevel) */
  /*    printf("old time of flight= %d\t",pER->timeOfFlight); */

  /*        */
  /*    put here the treatement for time of flight (coincidence event only) */
  /*       */

  /*      pER->timeOfFlight = pER->timeOfFlight + 1 -1; */
  /*        */
  /*         if(verboseLevel) */
  /*    printf("new time of flight= %d\n",pER->timeOfFlight); */

  /*   } */


  /*******************************************************************
   *                                                                  *
   *                                                                  *
   *                ID TREATMENT                                      *
   *                                                                  *
   *                                                                  *
   *******************************************************************/

  /*
     put here the treatement crystal ID
   */

  /*     if(pEH->detectorIDBool) */
  /*     { */
  /*       if(pEH->coincidenceBool == FALSE) */
  /*    { */
  /*      pcrist =  demakeid(pER->crystalIDs[0],pEncoH); */
  /*      if(pcrist[0] == 1)    just keep the internal layer */
  /*        keepIT = FALSE; */
  /*      free(pcrist); */

  /*    } */
  /*      else */
  /*    pER->crystalIDs[pEH->numberOfNeighbours+1] =  pER->crystalIDs[pEH->numberOfNeighbours+1]; */

  /*      } */

  /*******************************************************************
   *                                                                  *
   *                                                                  *
   *                ENERGY TREATMENT                                  *
   *                                                                  *
   *                                                                  *
   *******************************************************************/



  /*     if(pEH->energyBool == FALSE) */
  /*    { */
  /*      pER->energy[0] = pER->energy[0]; */

  /*    } */



  /*******************************************************************
   *                                                                  *
   *                                                                  *
   *                POSITIONS TREATMENT                               *
   *                                                                  *
   *                                                                  *
   ********************************************************************/
  /*   if(pEH->gantryAxialPosBool) */
  /*     pER->gantryAxialPos ;    that that here */

  /*   if(pEH->gantryAngularPosBool) */
  /*     pER->gantryAngularPos ;    that that here */

  /*    if(pEH->sourcePosBool) */
  /*     { */
  /*     pER->sourceAxialPos; */
  /*     pER->sourceAngularPos; */

  /*     } */

  /*

     If tou want to delete this current event
     return(NULL);
   */



  if (!keepIT)
    *ppER = NULL;
}
