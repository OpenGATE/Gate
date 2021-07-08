#include <stdio.h>
#include "lmf.h"

static u16 nbOfSct = 0;
static u16 *Rsector = NULL;
static i64 *shiftedTime = NULL;

void setShiftedTime(u16 inlineNbOfScts, u16 ** inlineRsector,
		    i64 ** inlineShiftedTime)
{
  nbOfSct = inlineNbOfScts;
  Rsector = *inlineRsector;
  shiftedTime = *inlineShiftedTime;
}

void shiftTime(const ENCODING_HEADER * pEncoH, EVENT_RECORD * pER)
{
  u8 i, j;
  u8 *bufCharTime = NULL;

  i64 tmp;

  for (i = 0; i < nbOfSct; i++)
    if (getRsectorID(pEncoH, pER) == Rsector[i]) {
      tmp = (i64) (u8ToU64(pER->timeStamp)) + shiftedTime[i];
      bufCharTime = u64ToU8((u64) tmp);
      for (j = 0; j < 8; j++)
	pER->timeStamp[j] = bufCharTime[j];
    }
}
