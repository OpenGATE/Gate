/*-------------------------------------------------------

List Mode Format 
                        
--  setGantryPosition.c  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2004 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of setGantryPosition

Allows to associate a Gantry positions (angular & axial) to events (Singles or Coinci)
The user has to specify the name of a binary file which contains
the clock time of the Gantry (u32) vs the angular (u16) and axial (u16) position.
He can also choose the finding mode (linear interpolation or clostest values) 
and sets the shifts in ms between the the clock time and the positions.

-------------------------------------------------------*/

#include <stdio.h>

#include "lmf.h"

#define TWO_PI 6.283185307
#define MILLI2PICO 1.E9

static LIST listG;
static u8 setMode = 0;
static u32 clkRatio = 0X40000;	/* Ratio between the FPGA clock (40MHz) * 64
				   (cause the event time is stored in 25ns/64 in the ccs)
				   and the Gantry clock (40MHz/4096) for the ClearPET of Lausanne */
static u32 angularShift = 0;
static u32 axialShift = 0;

static u16 fixedGantryAngularPos = 0;
static u16 fixedGantryAxialPos = 0;

/* 
   Allows to set the mode for returning the positions of the Gantry
   setMode == 0 will return the positions for the clock time the nearest of the event clock time 
   setMode == 1 will do a linear interpolation between the positions (angular and axial)
   of two clock times arround the event clock time
   setMode == 2 will do a quadratic interpolation NOT IMPLEMENTED YET
   setMode == 255 will put an unique gantry position for all events
*/
void setGantryPosMode(u8 inlineSetMode)
{
  setMode = inlineSetMode;
}

/* Allows to change the Ratio between Events clock and Gantry clock */
void setClkRatio(u32 inlineClkRatio)
{
  clkRatio = inlineClkRatio;
}

void setFixedGantryAngularPos(u16 inlineFixedGantryAngularPos)
{
  fixedGantryAngularPos = inlineFixedGantryAngularPos;
}

void setFixedGantryAxialPos(u16 inlineFixedGantryAxialPos)
{
  fixedGantryAxialPos = inlineFixedGantryAxialPos;
}

u16 roundM(float nb)
{
  u16 value;
  if (nb < 0)
    value = 0;
  else
    value = (u16) (nb + 0.5);

  return value;
}

u32 roundL(double nb)
{
  u32 value;
  if (nb < 0)
    value = 0;
  else
    value = (u32) (nb + 0.5);

  return value;
}

u32 roundPosL(double nb)
{
  u32 value;
  if (nb < 0)
    nb = -nb;
  value = (u32) (nb + 0.5);
  return value;
}

/* 
   Allows to set shifts between time clock vs angular and axial position 
   The shifts must be in ms 
*/
void setPositionsShifts(u32 inlineAngularShift, u32 inlineAxialShift)
{
  angularShift =
      roundM((float) (inlineAngularShift) * MILLI2PICO * 1000 /
	     getTimeStepFromCCH() / clkRatio);
  axialShift =
      roundM((float) (inlineAxialShift) * MILLI2PICO * 1000 /
	     getTimeStepFromCCH() / clkRatio);
}

void freeEV(gantryEVENT * pEV)
{
  free(pEV);
}

/* 
   Allows to read the positions (angular & axial) vs time clocks in the file named posFileName.
   It must be a binary file where an event is coded by an u32 (time clocks) and 2 u16 (angular & axial pos)
*/
int readPosFromFile(char *posFileName)
{
  FILE *pfile = NULL;
  u32 counts;
  u16 *buffer;
  int doneOnce = 0;
  gantryEVENT *pEin;

  if (pfile)
    return -1;
  else
    pfile = fopen(posFileName, "rb");

  if (!pfile) {
    printf("Binary file %s does not exist. EXIT\n", posFileName);
    exit(0);
  }

  if ((buffer = (u16 *) malloc(2 * sizeof(u16))) == NULL)
    printf
	("\n***ERROR : in setGantryPosition.c : impossible to do : malloc()\n");

  while ((fread(&counts, sizeof(u32), 1, pfile)) > 0) {
    fread(buffer, sizeof(u16), 2, pfile);

    if ((pEin = (gantryEVENT *) malloc(sizeof(gantryEVENT))) == NULL)
      printf
	  ("\n***ERROR : in setGantryPosition.c : impossible to do : malloc()\n");
    pEin->counts = counts;
    pEin->angularPos = buffer[0];
    pEin->axialPos = buffer[1];
    if (!doneOnce) {
      /* inits the list and put the first element */
      dlist_init(&listG, (void *) freeEV);
      if (dlist_ins_prev(&listG, dlist_head(&listG), pEin) != 0)
	return 1;
      doneOnce++;
    } else
      /* put elements in the list */
    if (dlist_ins_next(&listG, dlist_tail(&listG), pEin))
      return 1;
  }

  fclose(pfile);
  free(buffer);
  return 0;
}

/*
  Allows to find the angular position to which the time clocks
  are closest to evCnts
*/
gantryEVENT *findAngularElement(u32 evCnts)
{
  static ELEMENT *element;
  static int doneOnce = 0;
  gantryEVENT *currentEv;
  gantryEVENT *previousEv;

  if (!doneOnce) {
    element = dlist_head(&listG);
    doneOnce++;
  }
  while (1) {
    currentEv = (gantryEVENT *) (element->data);
    if ((currentEv->counts + angularShift) >= evCnts) {
      if (dlist_prev(element)) {
	previousEv = (gantryEVENT *) (dlist_prev(element)->data);
	if ((currentEv->counts + angularShift) - evCnts >
	    evCnts - (previousEv->counts + angularShift)) {
	  currentEv = previousEv;
	  element = dlist_prev(element);
	}
      }
      break;
    }
    if (dlist_is_tail(element))
      break;
    else
      element = dlist_next(element);
  }

  return currentEv;
}

/*
  Allows to find the axial position to which the time clocks
  are closest to evCnts
*/
gantryEVENT *findAxialElement(u32 evCnts)
{
  static ELEMENT *element;
  static int doneOnce = 0;
  gantryEVENT *currentEv;
  gantryEVENT *previousEv;

  if (!doneOnce) {
    element = dlist_head(&listG);
    doneOnce++;
  }
  while (1) {
    currentEv = (gantryEVENT *) (element->data);
    if ((currentEv->counts + axialShift) >= evCnts) {
      if (dlist_prev(element)) {
	previousEv = (gantryEVENT *) (dlist_prev(element)->data);
	if ((currentEv->counts + axialShift) - evCnts >
	    evCnts - (previousEv->counts + axialShift)) {
	  currentEv = previousEv;
	  element = dlist_prev(element);
	}
      }
      break;
    }
    if (dlist_is_tail(element))
      break;
    else
      element = dlist_next(element);
  }

  return currentEv;
}

/*
  resolve the quadratic system
  |x1² x1 1| |a|   |f1|
  |x2² x2 1| |b| = |f2|
  |x3² x3 1| |c|   |f3|
  for a, b, c
*/
void GetQuadraticParams(float *a, float *b, float *c,
			float x1, float f1, float x2, float f2, float x3,
			float f3)
{
  *a = (1 / ((x1 - x2) * (x1 - x3))) * f1 +
      (1 / ((-x1 + x2) * (x2 - x3))) * f2 +
      (1 / ((x1 - x3) * (x2 - x3))) * f3;
  *b = -((x2 + x3) / ((x1 - x2) * (x1 - x3))) * f1 +
      ((x1 + x3) / ((x1 - x2) * (x2 - x3))) * f2 -
      ((x1 + x2) / ((x1 - x3) * (x2 - x3))) * f3;
  *c = ((x2 * x3) / ((x1 - x2) * (x1 - x3))) * f1 +
      ((x1 * x3) / (-(x1 * x2) + x2 * x2 + x1 * x3 - x2 * x3)) * f2 +
      ((x1 * x2) / ((x1 - x3) * (x2 - x3))) * f3;
}


/* 
   Finds the positions before and after that one corresponding to the evCnts
   and do a linear interpolation between them.
   For angular position, holds account of discontinuity after a complete revolution 
   around the circle
*/
void findPositions(u32 evCnts, EVENT_RECORD ** ppER)
{
  static LMF_cch_scannerGeometry myScannerGeometry = { 0 };
  static LMF_cch_scannerGeometry *pScanGeo = &myScannerGeometry;
  static double angleDefaultUnitToRadConversionFactor = 0;

  static u32 ringCirc = 0;
  static ELEMENT *element;
  static int doneOnce = 0;
  gantryEVENT *nextEv;
  gantryEVENT *previousEv;

  u32 t1, t2;
  float ang1, ang2, ax1, ax2;

  if (!doneOnce) {
    if (fillInStructScannerGeometry(0, pScanGeo) == 1)
      printf("ERROR GEoME\n");
    angleDefaultUnitToRadConversionFactor = testAngleDefaultUnit();

    /* Calculates the ring circumference in angular position unit */
    ringCirc =
	roundPosL(TWO_PI /
		  (pScanGeo->azimuthalStep *
		   angleDefaultUnitToRadConversionFactor));

    element = dlist_head(&listG);
    doneOnce++;
  }
  while (1) {
    nextEv = (gantryEVENT *) (element->data);
    t2 = nextEv->counts + angularShift;
    if (t2 >= evCnts) {
      ang2 = (float) nextEv->angularPos;
      ax2 = (float) nextEv->axialPos;
      /*      printf ("%lu => %lu %.0f %.0f\n",evCnts, t2, ang2, ax2); */
      if (dlist_prev(element)) {
	previousEv = (gantryEVENT *) (dlist_prev(element)->data);
	t1 = previousEv->counts + angularShift;
	ang1 = (float) previousEv->angularPos;
	ax1 = (float) previousEv->axialPos;
	if (ang2 - ang1 < -(float) (ringCirc) * 5 / 6)
	  (*ppER)->gantryAngularPos =
	      roundM(ang1 +
		     (ang2 - ang1 + ringCirc) * (evCnts - t1) / (t2 -
								 t1)) %
	      ringCirc;
	else
	  (*ppER)->gantryAngularPos =
	      roundM(ang1 + (ang2 - ang1) * (evCnts - t1) / (t2 - t1));
	(*ppER)->gantryAxialPos =
	    roundM(ax1 + (ax2 - ax1) * (evCnts - t1) / (t2 - t1));
	/*              printf ("%lu => %lu %.0f %.0f\n",evCnts, t1, ang1, ax1); */
	/*              printf("angularPos = %hu axialPos = %hu\n************************************\n", */
	/*                     (*ppER)->gantryAngularPos, (*ppER)->gantryAxialPos);   */
      } else {
	(*ppER) = 0;
	/*              (*ppER)->gantryAngularPos = ang2; */
	/*              (*ppER)->gantryAxialPos = ax2; */
      }
      break;
    }
    if (dlist_is_tail(element)) {
      (*ppER) = 0;
      /*      previousEv = (gantryEVENT *) (element->data); */
      /*      (*ppER)->gantryAngularPos = previousEv->angularPos; */
      /*      (*ppER)->gantryAxialPos = previousEv->axialPos; */
      break;
    } else
      element = dlist_next(element);
  }
}

/* 
   Same fonction than findPositions but only for finding the angular position
*/
void findAngularPosition(u32 evCnts, EVENT_RECORD ** ppER)
{
  static LMF_cch_scannerGeometry myScannerGeometry = { 0 };
  static LMF_cch_scannerGeometry *pScanGeo = &myScannerGeometry;
  static double angleDefaultUnitToRadConversionFactor = 0;

  static u32 ringCirc = 0;
  static ELEMENT *element;
  static int doneOnce = 0;
  gantryEVENT *nextEv;
  gantryEVENT *previousEv;

  u32 t1, t2;
  float ang1, ang2;

  if (!doneOnce) {
    if (fillInStructScannerGeometry(0, pScanGeo) == 1)
      printf("ERROR GEoME\n");
    angleDefaultUnitToRadConversionFactor = testAngleDefaultUnit();
    ringCirc =
	roundPosL(TWO_PI /
		  (pScanGeo->azimuthalStep *
		   angleDefaultUnitToRadConversionFactor));

    element = dlist_head(&listG);
    doneOnce++;
  }
  while (1) {
    nextEv = (gantryEVENT *) (element->data);
    t2 = nextEv->counts + angularShift;
    if (t2 >= evCnts) {
      ang2 = (float) nextEv->angularPos;
      /*      printf ("%lu => %lu %.0f\n",evCnts, t2, ang2); */
      if (dlist_prev(element)) {
	previousEv = (gantryEVENT *) (dlist_prev(element)->data);
	t1 = previousEv->counts + angularShift;
	ang1 = (float) previousEv->angularPos;
	if (t2 - t1) {
	  if (ang2 - ang1 < -(float) (ringCirc) * 5. / 6.)
	    (*ppER)->gantryAngularPos =
		roundM(ang1 +
		       (ang2 - ang1 + ringCirc) * (evCnts - t1) / (t2 -
								   t1)) %
		ringCirc;
	  else
	    (*ppER)->gantryAngularPos =
		roundM(ang1 + (ang2 - ang1) * (evCnts - t1) / (t2 - t1));
	} else
	  (*ppER) = 0;
	/*              printf ("%lu => %lu %.0f\n",evCnts, t1, ang1); */
	/*              printf("angularPos = %hu\n************************************\n", */
	/*                     (*ppER)->gantryAngularPos);   */
      } else {
	(*ppER) = 0;
	/*              (*ppER)->gantryAngularPos = ang2; */
      }
      break;
    }
    if (dlist_is_tail(element)) {
      (*ppER) = 0;
      /*      previousEv = (gantryEVENT *) (element->data); */
      /*      (*ppER)->gantryAngularPos = previousEv->angularPos; */
      break;
    } else
      element = dlist_next(element);
  }
}

/*
  Same fonction than findPositions but only for finding the axial position
*/
void findAxialPosition(u32 evCnts, EVENT_RECORD ** ppER)
{
  static ELEMENT *element;
  static int doneOnce = 0;
  static u16 maxGantryAxialPos = 0;

  gantryEVENT *nextEv;
  gantryEVENT *previousEv;

  int index;

  char response[10];
  float maxAxialPos;
  u16 rsectorAxialSize = 0;
  u16 gantryAxialPos = -1;

  u32 t1, t2;
  float ax1, ax2;

  if (!doneOnce) {
    index = getLMF_cchInfo("rsector axial size");
    rsectorAxialSize = (u16) plist_cch[index].def_unit_value.vNum;

    index = getLMF_cchInfo("axial step");

    maxAxialPos = rsectorAxialSize;	//(u16)(-1) * plist_cch[index].def_unit_value.vNum;
    printf
	("Max Axial Size in mm (default rsector axial length = %.2f mm): ",
	 maxAxialPos);
    if (fgets(response, sizeof(response), stdin))
      sscanf(response, "%f", &maxAxialPos);

    printf("maxAxialPos = %f\n", maxAxialPos);

    maxGantryAxialPos =
	maxAxialPos / plist_cch[index].def_unit_value.vNum + 0.5;

    printf("maxAxialPos = %.2f -> maxGantryAxialPos = %hd\n",
	   maxAxialPos, maxGantryAxialPos);

    element = dlist_head(&listG);
    doneOnce++;
  }

  while (1) {
    nextEv = (gantryEVENT *) (element->data);
    t2 = nextEv->counts + axialShift;
    if (t2 >= evCnts) {
      ax2 = (float) nextEv->axialPos;
      /*      printf ("%lu => %lu %.0f\n",evCnts, t2, ax2); */
      if (dlist_prev(element)) {
	previousEv = (gantryEVENT *) (dlist_prev(element)->data);
	t1 = previousEv->counts + axialShift;
	ax1 = (float) previousEv->axialPos;
	gantryAxialPos =
	    roundM(ax1 + (ax2 - ax1) * (evCnts - t1) / (t2 - t1));
	if (gantryAxialPos > maxGantryAxialPos)
	  *ppER = NULL;
	else
	  (*ppER)->gantryAxialPos = gantryAxialPos;


	/*              printf ("%lu => %lu %.0f\n",evCnts, t1, ax1); */
	/*              printf("axialPos = %hu\n************************************\n", */
	/*                     (*ppER)->gantryAxialPos);   */
      } else {
	(*ppER) = 0;
	/*              (*ppER)->gantryAxialPos = ax2; */
      }
      break;
    }
    if (dlist_is_tail(element)) {
      (*ppER) = 0;
      /*      previousEv = (gantryEVENT *) (element->data); */
      /*      (*ppER)->gantryAxialPos = previousEv->axialPos; */
      break;
    } else
      element = dlist_next(element);
  }
}

/*
  Main function: matches the Gantry position and the encoded event
  by using the others functions of setGantryPosition.c
*/
EVENT_RECORD *setGantryPosition(const ENCODING_HEADER * pEncoH,
				EVENT_HEADER * pEH,
				const GATE_DIGI_HEADER * pGDH,
				EVENT_RECORD * pER)
{
  gantryEVENT *pGEv;
  u32 evCnts;
  static int doneOnce;

  if (!doneOnce) {
    pEH->gantryAngularPosBool = 1;
    pEH->gantryAxialPosBool = 1;
    doneOnce = 1;
  }

  if (setMode == 255) {
    pER->gantryAngularPos = fixedGantryAngularPos;
    pER->gantryAxialPos = fixedGantryAxialPos;
  }

  else {
    if (pEH->coincidenceBool == 1)
      evCnts =
	  roundL((double) getTimeOfThisCOINCI(pER) /
		 (CONVERT_TIME_COINCI * getTimeStepFromCCH() / 1000 *
		  clkRatio));
    else
      evCnts =
	  roundL((double) getTimeOfThisEVENT(pER) /
		 (getTimeStepFromCCH() / 1000 * clkRatio));

    if (setMode == 0) {
      if (angularShift == axialShift) {
	pGEv = findAngularElement(evCnts);

	/*          printf("evCnts %lu => Gantry counts %lu\n",evCnts,pGEv->counts); */

	//   POSITIONS
	if (pGEv) {
	  pER->gantryAngularPos = pGEv->angularPos;	/* gantry's angular position */
	  pER->gantryAxialPos = pGEv->axialPos;	/* gantry's axial position */
	}
      } else if (setMode == 1) {
	pGEv = findAngularElement(evCnts);

	/*          printf("evCnts %lu => Gantry counts %lu\n",evCnts,pGEv->counts); */

	//   POSITIONS
	if (pGEv)
	  pER->gantryAngularPos = pGEv->angularPos;	/* gantry's angular position */


	pGEv = findAxialElement(evCnts);
	if (pGEv)
	  pER->gantryAxialPos = pGEv->axialPos;	/* gantry's axial position */
      }
    } else if (setMode == 1) {
      if (angularShift == axialShift)
	findPositions(evCnts, &pER);
      else {
	findAngularPosition(evCnts, &pER);
	if (pER)
	  findAxialPosition(evCnts, &pER);
      }
    }
  }
  return pER;
}

/*
  Destroy the List listG
*/
void positionListDestructor()
{
  dlist_destroy(&listG);
}
