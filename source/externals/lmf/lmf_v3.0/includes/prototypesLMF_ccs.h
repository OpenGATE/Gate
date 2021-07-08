/*-------------------------------------------------------

List Mode Format 
                        
--  prototypesLMF_ccs.h  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of prototypesLMF_ccs.h:


function prototypes used for binary part of LMF.

---------------------------------------------------------------------------*/


#ifdef __cplusplus
extern "C" {
#endif
#include <stdlib.h>
#include <stdio.h>
#ifndef _PROTOTYPE_LMF_H
#define _PROTOTYPE_LMF_H

  /* PROTOTYPES OF FUNCTIONS FOR THE LMF BUILDER */

  /***************************
        intro.c
  ****************************/

  void intro();

  /***************************
        displayListOfFiles.c
  ****************************/
  void displayListOfFiles();


  /***************************
 testYourSystemForLMF.c
  ****************************/

  int testYourSystemForLMF();

  /***************************
           monteCarloEngine.c
  ****************************/

  /*generate a random float number between 0 and 1 */
  double randd();
  /*generate a random i16 number between a and b */
  u16 monteCarloInt(u16 a, u16 b);


  /* binary/decimal converter */
  int bindec(i16 * a);
  void decbin(int a);

  /***************************
           hardget.c
  ****************************/
  /*ask the user to give a number between min & max */
  i16 hardgeti16(i16 min, i16 max);
  /*ask the user to give a letter between y or n */
  i16 hardgetyesno();




  /***************************
           askTheMode.c
  ****************************/
  /*ask the user to give a number between 0 or 3 */
  /* 0 no records */
  /* 1 Event */
  /* 2 Count rate */
  /* 3 The Both */
  u16 askTheMode(void);



  /***************************
           poweri8.c
  ****************************/
  /* power of 2 i8 (or i16) : a**b */
  u16 poweri8(u8 a, u8 b);



  /***************************
           poweri16.c
  ****************************/
  u16 poweri16(u16 a, u8 b);



  /***************************
           findvnn.c
  ****************************/
  /* Find the number of neighbour from the ep (encoding pattern) */
  u16 findvnn(u16 ep2);		/* from pattern */
  u16 findvnn2(u16 nn);		/* from nn : order of neigh. */



  /***************************
           makeid.c
  ****************************/
  /* MAKE THE ID */
  u64 makeid(u16, u16, u16, u16, u16, const ENCODING_HEADER *, u16 *);

  /***************************
           demakeid.c
  ****************************/
  /* DEMAKE THE ID */
  u16 *demakeid(u64, const ENCODING_HEADER *);


  /***************************
       calculatesizeevent.c
  ****************************/
  /* compute in bytes the size of records  */
  u16 calculatesizeevent(u16);
  /***************************
    calculatesizecountrate.c
  ****************************/
  u16 calculatesizecountrate(u16, const ENCODING_HEADER *);


  /***************************
    calculatesizegatedigi.c
  ****************************/
  u16 calculatesizegatedigi(u16, u16);


  /***************************
       calculateSSrepeat.c
  ****************************/
  /* compute number of details for count rate records */
  i16 calculateSSrepeat(i16, const ENCODING_HEADER *);

  /***************************
       generateEncoH.c
  ****************************/
  /* generate a realist encoding Header : generateEncoH is doing itself the malloc */
  ENCODING_HEADER *generateEncoH(u16);
  void generateEncoHDestructor();
  /***************************
       fillEncoH.c
  ****************************/
  /* fillEncoH doesn't doing itself the malloc */
  ENCODING_HEADER *fillEncoH(ENCODING_HEADER *, u16);

  /***************************
       fillEncoHforGate.c
  ****************************/
  /* with intelligent rule maker ... */
  void fillEncoHforGate(int, int,	/* axial / tangeantial rsector */
			int, int,	/* ...  submodule */
			int, int,	/* module */
			int, int,	/* crystal */
			int, int,	/* axial / radial layer */
			ENCODING_HEADER *, u16);

  void inteligentRuleMaker(int, int, int, int, int, ENCODING_HEADER *);

  int findNumberOfBitsNeededFor(int);

  /***************************
       fillEncoHforAscii.c
  ****************************/
  /* more or less the same than fillEncoHforGate.c */

  ENCODING_HEADER *fillEncoHforAscii(int, int,	/* axial / tangeantial rsector */
				     int, int,	/* ...  submodule */
				     int, int,	/* module */
				     int, int,	/* crystal */
				     int, int,	/* axial / radial layer */
				     ENCODING_HEADER *, u16);


  /***************************
       generateEH.c
  ****************************/
  /* generate a realist event Header */
   EVENT_HEADER(*generateEH(ENCODING_HEADER * pEncoH));
  void generateEHDestructor();
  /***************************
       generateCRH.c
  ****************************/
  /* generate a realist count rate Header */
   COUNT_RATE_HEADER(*generateCRH(void));
  void generateCRHDestructor();

  /***************************
       generateGDH.c
  ****************************/
  /* generate a gate digi Header */
   GATE_DIGI_HEADER(*generateGDH(void));
  void generateGDHDestructor();



  /***************************
       generatecC.c
  ****************************/
  /* generate a realist current content */
   CURRENT_CONTENT(*generatecC());
  void generatecCDestructor();
  /***************************
       generateER.c
  ****************************/
  /* generate a realist event record and a count rate Record */
   EVENT_RECORD(*generateER(ENCODING_HEADER * pEncoH, EVENT_HEADER * pEH));

  void generateERDestructor();
  /***************************
       generateCRR.c
  ****************************/
   COUNT_RATE_RECORD(*generateCRR(ENCODING_HEADER * pEncoH,
				  COUNT_RATE_HEADER * pCRH));
  void generateCRRDestructor();

  /***************************
       generateGDR.c
  ****************************/
  GATE_DIGI_RECORD *generateGDR(ENCODING_HEADER * pEncoH,
				GATE_DIGI_HEADER * pGDH);
  void generateGDRDestructor();



  /***************************
       buildHead.c
  ****************************/
  /* build and write the head of the .ccs FILE */
  void buildHead(const ENCODING_HEADER *,
		 const EVENT_HEADER *,
		 const GATE_DIGI_HEADER *,
		 const COUNT_RATE_HEADER *, FILE *);

  /***************************
       buildER.c
  ****************************/
  /* build and write the body of the .ccs FILE (one record by one) */
  void buildER(const EVENT_HEADER *,
	       const u16, const EVENT_RECORD *, FILE *);

  /***************************
       buildCRR.c
  ****************************/
  void buildCRR(const ENCODING_HEADER *,
		const COUNT_RATE_HEADER *,
		const COUNT_RATE_RECORD *, FILE *);

  /***************************
       buildGDR.c
  ****************************/
  void buildGDR(const GATE_DIGI_HEADER *,
		const GATE_DIGI_RECORD *, const EVENT_HEADER *, FILE *);

  u8 makeOneNumberOfCOmptonWithTwo(u8 a, u8 b);




  /***************************
       makeEpattern.c
  ****************************/
  /* encode the patterns for each kind of records */
  u16 makeEpattern(const EVENT_HEADER *);
  /***************************
       makeCRpattern.c
  ****************************/
  u16 makeCRpattern(const COUNT_RATE_HEADER *);
  /***************************
       makeGDpattern.c
  ****************************/
  /* encode the patterns for each kind of records */
  u16 makeGDpattern(const GATE_DIGI_HEADER *);


  /***************************
       LMFbuilder.c
  ****************************/
  void LMFbuilder(const ENCODING_HEADER *,
		  const EVENT_HEADER *,
		  const COUNT_RATE_HEADER *,
		  const GATE_DIGI_HEADER *,
		  const CURRENT_CONTENT *,
		  const EVENT_RECORD *,
		  const COUNT_RATE_RECORD *, FILE **, const i8 *);

  void CloseLMFfile(FILE * pfile);



  /***************************
    coincidenceOutputModule.c
  ****************************/
  void setCoincidenceOutputMode(u8 value);
  u8 getCoincidenceOutputMode();
  void outputCoincidence(const ENCODING_HEADER * pEncoHC,
			 const EVENT_HEADER * pEHC,
			 const GATE_DIGI_HEADER * pGDHC,
			 const CURRENT_CONTENT * pcCC,
			 const EVENT_RECORD * pERC,
			 i8 coincidenceOutputMode);

  void closeOutputCoincidenceFile();
  void outputCoincidenceModuleDestructor();
   LIST(*getListOfCoincidenceOutput());
  void initializeListOfCoincidence();
  /***************************
       LMFCbuilder.c
  ****************************/

  void LMFCbuilder(const ENCODING_HEADER * pEncoH,
		   const EVENT_HEADER * pEH,
		   const GATE_DIGI_HEADER * pGDH,
		   const CURRENT_CONTENT * pcC,
		   const EVENT_RECORD * pER,
		   FILE ** ppfile, const i8 * nameOfFile);

  void FreeLMFCBuilderCarrier(ENCODING_HEADER * pEncoH,
			      EVENT_HEADER * pEH,
			      GATE_DIGI_HEADER * pGDH,
			      CURRENT_CONTENT * pcC, EVENT_RECORD * pER);

  void CloseLMFCfile(FILE * pfile);


  /***************************
      treatEventRecord.c
  ****************************/
  void treatEventRecord(const ENCODING_HEADER *,
			const EVENT_HEADER *,
			const GATE_DIGI_HEADER *, EVENT_RECORD **);

  /***************************
      keepOnlyTrue.c
  ****************************/
  EVENT_RECORD *keepOnlyTrue(const ENCODING_HEADER *,
			     const EVENT_HEADER *,
			     const GATE_DIGI_HEADER *, EVENT_RECORD *);

  /***************************
      cutEnergyModule.c
  ****************************/
  EVENT_RECORD *cutEnergy(const ENCODING_HEADER *,
			  const EVENT_HEADER *,
			  const GATE_DIGI_HEADER *, EVENT_RECORD * pER);
  void useCalfTable();
  void getTopology(int nbl, int nbc, int nbsm, int nbm, int nbrs);
  void setEnergyCalib(char energyCalibFile[40], int nbl, int nbc, int nbsm,
		      int nbm, int nbrs);
  void setUpEnergyLimit(int upKeVLimit);
  void setDownEnergyLimit(int downKeVLimit);
  void setFPGANeighSelect(int fpgaNeighSelect);

  /*EVENT_RECORD *cutEnergy(const ENCODING_HEADER *,
     const EVENT_HEADER *,
     const GATE_DIGI_HEADER *,
     EVENT_RECORD *);

     void setUpEnergyLimit(int);
     void setDownEnergyLimit(int); */
  u64 printEnergyModuleRejectedEventNumber(void);

  /***************************
      cutEventsModule.c
  ****************************/
  EVENT_RECORD *cutEventsNumber(const ENCODING_HEADER *,
				const EVENT_HEADER *,
				const GATE_DIGI_HEADER *, EVENT_RECORD *);

  void setRecordUpLimit(int);
  void setRecordDownLimit(int);


  /***************************
      delayLineModule.c
  ****************************/
  EVENT_RECORD *delayLine(const ENCODING_HEADER *,
			  const EVENT_HEADER *,
			  const GATE_DIGI_HEADER *, const EVENT_RECORD *);

  void initDelayList(u16);
  void setMyDelayBase(u64);


  void destroyDelayLine(void);

  /***************************
      sortTime.c
  ****************************/

  void sortTime(const ENCODING_HEADER *,
		const EVENT_HEADER *,
		const COUNT_RATE_HEADER *,
		const GATE_DIGI_HEADER *,
		const CURRENT_CONTENT *,
		const EVENT_RECORD *,
		const COUNT_RATE_RECORD *, FILE *, const i8 *);

  void destroySortTime(void);


  int finishTimeSorting(const ENCODING_HEADER *,
			const EVENT_HEADER *,
			const COUNT_RATE_HEADER *,
			const GATE_DIGI_HEADER *,
			const CURRENT_CONTENT *,
			const EVENT_RECORD *,
			const COUNT_RATE_RECORD *, FILE *, const i8 *);

  void setTimeListSize(int size);


  /***************************
       LMFreader.c
  ****************************/
  int LMFreader(const i8 *, const i8 *);
  int LMFreaderDestructor(void);


  /***************************
       getAndSetOF.c
  ****************************/
  FILE *getAndSetOutputFileName();
  FILE *getAndSetThisOutputFileName(i8 *);
  void destroyGetAndSetOutputFileName();
  int OF_is_Set();


  /***************************
       readHead.c
  ****************************/
  ENCODING_HEADER *readHead(FILE *);
  void destroyReadHead();

  /***************************
       extractEpat.c
  ****************************/
  EVENT_HEADER *extractEpat(u16);
  void destroyExtractEpat();

  /***************************
       extractCRpat.c
  ****************************/
  COUNT_RATE_HEADER *extractCRpat(u16);
  void destroyExtractCRpat();

  /***************************
       extractGDpat.c
  ****************************/
  GATE_DIGI_HEADER *extractGDpat(u16);
  void destroyExtractGDpat();



  /***************************
       readOneEventRecord.c
  ****************************/
  void readOneEventRecord(u8 * pBufEvent,
			  u16 Epattern,
			  EVENT_HEADER * pEH,
			  u16 encodingIDSize, EVENT_RECORD * pER);

  /***************************
    readOneCountRateRecord.c
  ****************************/
  COUNT_RATE_RECORD *readOneCountRateRecord(ENCODING_HEADER *, u8 *, u16);
  void destroyCRRreader(COUNT_RATE_HEADER *);

  /***************************
    readOneGateDigiRecord.c
  ****************************/
  u8 firstHalfOf(u8);
  u8 secondHalfOf(u8);
  void readOneGateDigiRecord(u8 * pBufGateDigi,
			     u16 GDpattern,
			     GATE_DIGI_HEADER * pGDH,
			     EVENT_HEADER * pEH, EVENT_RECORD * pER);

  /***************************
       processRecordCarrier.c
  ****************************/

  void processRecordCarrier(const ENCODING_HEADER *,
			    EVENT_HEADER *,
			    const GATE_DIGI_HEADER *,
			    const COUNT_RATE_HEADER *,
			    const CURRENT_CONTENT *,
			    EVENT_RECORD *,
			    const COUNT_RATE_RECORD *, const i8 *, FILE *);


  /***************************
       countRecords.c
  ****************************/
  void countRecords(const ENCODING_HEADER *,
		    const EVENT_HEADER *,
		    const COUNT_RATE_HEADER *,
		    const GATE_DIGI_HEADER *,
		    const CURRENT_CONTENT *,
		    const EVENT_RECORD *, const COUNT_RATE_RECORD *);

  void destroyCounter();

  /***************************
       dumpTheRecord.c
  ****************************/
  /* The dump functions are just to see the detail of
     a LMF binary file */
  void dumpTheRecord(const ENCODING_HEADER *,
		     const EVENT_HEADER *,
		     const COUNT_RATE_HEADER *,
		     const GATE_DIGI_HEADER *,
		     const CURRENT_CONTENT *,
		     const EVENT_RECORD *, const COUNT_RATE_RECORD *);

  /***************************
       dumpHead.c
  ****************************/
  void dumpHead(const ENCODING_HEADER *,
		const EVENT_HEADER *,
		const GATE_DIGI_HEADER *, const COUNT_RATE_HEADER *);

  /***************************
       dumpEventHeader.c
  ****************************/
  void dumpEventHeader(const EVENT_HEADER *);

  /***************************
       dumpCountRateHeader.c
  ****************************/
  void dumpCountRateHeader(const COUNT_RATE_HEADER *);
  /***************************
       dumpGateDigiHeader.c
  ****************************/
  void dumpGateDigiHeader(const GATE_DIGI_HEADER *);


  /***************************
       dumpEventRecord.c
  ****************************/
  void dumpEventRecord(const ENCODING_HEADER *,
		       const EVENT_HEADER *, const EVENT_RECORD *);

  /***************************
       dumpCountRateRecord.c
  ****************************/
  void dumpCountRateRecord(const ENCODING_HEADER *,
			   const COUNT_RATE_HEADER *,
			   const COUNT_RATE_RECORD *);

  /***************************
       dumpGateDigiRecord.c
  ****************************/
  void dumpGateDigiRecord(const ENCODING_HEADER *,
			  const EVENT_HEADER *,
			  const GATE_DIGI_HEADER *,
			  const GATE_DIGI_RECORD *);

  /***************************
       oneList_CoincidenceSorter.c
  ****************************/

  u64 setCSdtValue();
  int setCSdtMode();

  int setCSverboseLevel();
  int setCScoincidencewindow();
  int setCSstackcuttime();
  int setCSsaveMultipleBool();
  int setCSsaveAutoCoinciBool();
  int setCSrsectorNeighOrder();
  int getAndSetCSsearchMode();
  void setSearchMode(int);
  void setAllCSparameters(int, int, u64, int, int, int, int, u64, int);

  void initBonusKit();
  void initCleanKit();
  void initFlowChart();

  int sortCoincidence(const ENCODING_HEADER *,
		      const EVENT_HEADER *,
		      const COUNT_RATE_HEADER *,
		      const GATE_DIGI_HEADER *,
		      const CURRENT_CONTENT *,
		      const EVENT_RECORD *, const COUNT_RATE_RECORD *);

  void destroyList();


  /***************************
       coincidenceAnalyser.c
  ****************************/
  void coincidenceAnalyser(const ENCODING_HEADER *,
			   const EVENT_HEADER *,
			   const GATE_DIGI_HEADER *, const EVENT_RECORD *);

  void destroyCoincidenceAnalyser();


  /***************************
       outputAsciiMgr.c
  ****************************/
  void outputAscii(const ENCODING_HEADER *,
		   const EVENT_HEADER *,
		   const GATE_DIGI_HEADER *, const EVENT_RECORD *);

  void destroyOutputAsciiMgr();



  /***************************
       outputRootMgr.c
  ****************************/
  void outputRoot(const ENCODING_HEADER *,
		  const EVENT_HEADER *,
		  const GATE_DIGI_HEADER *, const EVENT_RECORD *);

  void destroyOutputRootMgr();



  /***************************
       coinciDeadTimeMgr.c
  ****************************/
  int deadTimeCoinciMgr(ELEMENT * first, ENCODING_HEADER * pEncoHC);
  void deastroyDeadTimeCoinciMgr();



  /***************************
        juelichDeadTime.c
  ****************************/


  void juelichDT(const ENCODING_HEADER *,
		 const EVENT_HEADER *,
		 const COUNT_RATE_HEADER *,
		 const GATE_DIGI_HEADER *,
		 const CURRENT_CONTENT *,
		 const EVENT_RECORD *,
		 const COUNT_RATE_RECORD *, const i8 *);

  void setPileUpAnalysis(u8 mybool);
  void setHighDepthDT(u64 value);
  void setLowDepthDT(u64 value);
  void setThreeSingleDT(u64 value);

  u8 sectorIsAlive(int erSector, u64 timeER);
  u8 moduleIsAlive(int erSector, int erModule, u64 timeER);


  void destroyJuelichDeadTime(void);


  /***************************
      getDtID.c
  ****************************/

  u16 getModuleID(const ENCODING_HEADER *, const EVENT_RECORD *);
  u16 getRsectorID(const ENCODING_HEADER *, const EVENT_RECORD *);
  u16 getLayerID(const ENCODING_HEADER *, const EVENT_RECORD *);
  u16 getCrystalID(const ENCODING_HEADER *, const EVENT_RECORD *);
  u16 getSubmoduleID(const ENCODING_HEADER *, const EVENT_RECORD *);

  u16 getModuleID2(const ENCODING_HEADER *, const EVENT_RECORD *);
  u16 getRsectorID2(const ENCODING_HEADER *, const EVENT_RECORD *);
  u16 getLayerID2(const ENCODING_HEADER *, const EVENT_RECORD *);
  u16 getCrystalID2(const ENCODING_HEADER *, const EVENT_RECORD *);
  u16 getSubmoduleID2(const ENCODING_HEADER *, const EVENT_RECORD *);


  /***************************
       deadTimeMgr.c
  ****************************/


  int getElementID(const ENCODING_HEADER *, const EVENT_RECORD *);
  void setDeadTimeModeWithThis(int, u64, float);
  void setDepthOfDeadTime(int);
  /* where
     0 layer
     1 crystal
     2 submodule
     3 module
     4 rsector (default)
     5 scanner
   */

  void setDeadTimeMode(void);
  int deadTimeMgr(const ENCODING_HEADER *,
		  const EVENT_HEADER *,
		  const GATE_DIGI_HEADER *, const EVENT_RECORD *);
  void destroyDeadTimeMgr();


  /***************************
       oneList_cleanKit.c
  ****************************/
/*   EVENT_RECORD *fillCoinciRecord(EVENT_RECORD *first,EVENT_RECORD *second,u16 multipleID); */

  void initCoinciFile(const ENCODING_HEADER *,
		      const EVENT_HEADER *,
		      const GATE_DIGI_HEADER *,
		      const COUNT_RATE_HEADER *, const CURRENT_CONTENT *);
  void destroyCoinciFile(void);

  int cleanListP1(LIST *, EVENT_RECORD *);

  u64 finalCleanListP1(LIST *);



  void multipleCoincidenceMgr(LIST *, ELEMENT *, int);
  void multipleCombinatCoincidenceMgr(LIST *, ELEMENT *, int);
  void incrementNumberOfSingleton(void);

  int checkForAutoCoinci(ELEMENT *, ELEMENT *, const ENCODING_HEADER *);

  int diffRsector(ELEMENT *, ELEMENT *, const ENCODING_HEADER *);
  int diffRsectorER(EVENT_RECORD *, EVENT_RECORD *,
		    const ENCODING_HEADER *);
  u64 diffTime(ELEMENT *, ELEMENT *, const ENCODING_HEADER *);


  /***************************
       oneList_BonusKit.c
  ****************************/

  EVENT_RECORD *newER(const EVENT_HEADER *);
  void freeER(EVENT_RECORD *);
  void copyER(const EVENT_RECORD *, EVENT_RECORD *, const EVENT_HEADER *);
  void copyGDR(GATE_DIGI_RECORD *, GATE_DIGI_RECORD *);
  void freeGDR(GATE_DIGI_RECORD *);
   GATE_DIGI_RECORD(*newGDR(void));
  int insertOK(LIST *, ELEMENT *, const EVENT_RECORD *);
  void print_list(const LIST *);
  int getOrderOfListSize(const LIST *);
  int tailCloser(const LIST *, u64);
  LINK locateInList(const LIST *, EVENT_RECORD *);	/* iteratif */
  LINK searchIterative(const LIST *, EVENT_RECORD *);
  LINK searchRecursive(const LIST *, EVENT_RECORD *);
  LINK searchRecursiveFromHead(const LIST *, EVENT_RECORD *);
  LINK searchRecursiveFromTail(const LIST *, EVENT_RECORD *);
  LINK searchIterativeFromTail(const LIST *, EVENT_RECORD *);
  LINK searchIterativeFromHead(const LIST *, EVENT_RECORD *);
  LINK downInList(int, LINK);
  LINK upInList(int, LINK);
  LINK bigStepUp(LINK, int, u64);
  LINK bigStepDown(LINK, int, u64);
  /***************************
       oneList_flowi8t.c
  ****************************/
  void fcWAY_1Y(LIST *, EVENT_RECORD *);
  void fcWAY_2N(void);
  void fcWAY_2Y(void);
  void fcWAY_3N(void);
  void fcWAY_3Y(void);
  void fcWAY_6N(void);
  void fcWAY_6Y(void);
  void fcWAY_7Y(void);
  /***********************************************/


  /***********************************************/





  /* make and demake the encoding rule ex : 111 0000 111 00000 1 */
  /***************************
       demakeRule.c
  ****************************/
  void demakeRule(ENCODING_HEADER *, u64);

  /***************************
       makeRule.c
  ****************************/
  u64 makeRule(const ENCODING_HEADER *);


  /* LMF INTERFACE NEEDED FUNCTION */
  /* take a i8 and convert it in integer
     Results 8 u8 pointer 
     These functions are in timeValueManager.c */

  /***************************
       timeValueManager.c
  ****************************/
  u64 getTimeStepFromCCH(void);
  u64 getTimeOfThisEVENT(const EVENT_RECORD *);
  u64 getTimeOfThisCOINCI(const EVENT_RECORD *);
  float getTimeOfFlightOfThisCOINCI(const EVENT_RECORD *);

   u8(*doubleToU8(double x));
  double u8ToDouble(u8 * pc);

  u64 u8ToU64(const u8 *);
  u8 *u64ToU8(const u64);

  void time38bitShifter(u8 *, u32);
  u32 getStrongBit(u8 *);

  /***************************
       read_LMF.c
  ****************************/
  /* reading function for building sinograms */
  FILE *open_CCH_file(const i8 * nameOfFileCCH);
  FILE *open_CCS_file(const i8 * nameOfFileCCS);
  int read_LMF(const i8 * nameOfFileCCS,
	       const i8 * nameOfFileCCH,
	       double *x1,
	       double *y1, double *z1, double *x2, double *y2, double *z2);

  void init_read_LMF();
  void destroy_read_LMF();
  /***************************
       findXYZinLMFfile.c
  ****************************/
  /* reading function for building sinograms */
  FILE *open_CCH_file2(const i8 * nameOfFileCCH);
  FILE *open_CCS_file2(const i8 * nameOfFileCCS);

  int findXYZinLMFfile(FILE * pfCCS,
		       double *x1,
		       double *y1,
		       double *z1,
		       double *x2,
		       double *y2, double *z2, ENCODING_HEADER * pEncoH);
  void init_findXYZinLMFfile(FILE * pfCCS, ENCODING_HEADER * pEncoH);

  void destroy_findXYZinLMFfile(ENCODING_HEADER * pEncoH);
  void setERforSTIR(EVENT_RECORD * pER);
   EVENT_RECORD(*getERforSTIR());



  /***************************
       findXYZforSingles.c
  ****************************/

  /*
     void init_findXYZforSingles(FILE *pfCCS,ENCODING_HEADER *pEncoH);

     int findXYZforSingles(FILE *pfCCS,
     double *x1,
     double *y1,
     double *z1,
     double *x2,
     double *y2,
     double *z2,
     ENCODING_HEADER *pEncoH,
     u16 myID);

     void destroy_findXYZforSingles(ENCODING_HEADER *pEncoH);
   */
  /***************************
       LMF_ccsReaderKit.c
  ****************************/

  FILE *openCCSfile(const i8 * nameOfFile);
  int closeCCSfile(FILE * pfile);





  /***************************
       fillEHforGATE.c
  ****************************/
  /* standard filling of a event Header  */
   EVENT_HEADER(*fillEHforGATE(EVENT_HEADER * peH));

  /***************************
       fillEHforAscii.c
  ****************************/
  /* standard filling of a event Header ; code = 1 (singles) or 2 (coincidences) */
   EVENT_HEADER(*fillEHforAscii(EVENT_HEADER * peH, int code));




  /***************************
       fillGDHforGATE.c
  ****************************/
  /* standard filling of a event Header */
   GATE_DIGI_HEADER(*fillGDHforGATE(GATE_DIGI_HEADER * pGDH));





  /* take a treated hit (treated by endOfHitCollection) 
     and fill a LMF Event Record EH structure */
   EVENT_RECORD(*fillERfromGATE(double minTime1,
				u8 maxE1,
				u16 maxId1,
				u16 angPos,
				u16 axiPos,
				ENCODING_HEADER * pEncoH,
				EVENT_HEADER * pEH, EVENT_RECORD * pER));



  /***************************
      tripletAnalysis.c
  ****************************/
  u16 tripletAnalysis(ENCODING_HEADER * pEncoHC,
		      EVENT_HEADER * pEH, EVENT_RECORD ** pER);

  void destroyTripletAnalysis();


  /***************************
       help.c
  ****************************/
  /* verbosity to help users */

  void helpForStackCutTime();
  void helpExample1(void);


  /***************************
       setGantryPosition.c
  ****************************/
  /* set the axial and angular positions of the gantry
     starting from a file */

  void setGantryPosMode(u8);
  void setClkRatio(u32);
  void setPositionsShifts(u32, u32);
  int readPosFromFile(char *);
  EVENT_RECORD *setGantryPosition(const ENCODING_HEADER *,
				  EVENT_HEADER *,
				  const GATE_DIGI_HEADER *,
				  EVENT_RECORD *);
  void positionListDestructor();

  /***************************
          shiftTime.c
  ****************************/

  void setShiftedTime(u16, u16 **, i64 **);
  void shiftTime(const ENCODING_HEADER *, EVENT_RECORD *);


  /***************************
          blockSorter.c
  ****************************/

  void setNbOfSct(u16);
  void sortBlocks(const ENCODING_HEADER *,
		  const EVENT_HEADER *,
		  const COUNT_RATE_HEADER *,
		  const GATE_DIGI_HEADER *,
		  const CURRENT_CONTENT *,
		  const EVENT_RECORD *, const COUNT_RATE_RECORD *, char *);
  void finalCleanListEv();

  /***************************
        lmfFilesMerger.c
  ****************************/

  void setNewLMFfileName(i8[charNum]);

  void mergeLMFfiles(const ENCODING_HEADER *,
		     const EVENT_HEADER *,
		     const COUNT_RATE_HEADER *,
		     const GATE_DIGI_HEADER *,
		     const CURRENT_CONTENT *,
		     const EVENT_RECORD *,
		     const COUNT_RATE_RECORD *, FILE **);

  /***************************
       geometrySelector.c
  ****************************/

  void setSectors(u16, u16 **);
  void setModules(u16, u16 **);
  void setCrystals(u16, u16 **);

  void geometrySelector(const ENCODING_HEADER *, EVENT_RECORD **);

  /***************************
       daqTimeCorrector.c
  ****************************/

  void setNbOfDaqSct(u16);
  void correctDaqTime(const ENCODING_HEADER *,
		      const EVENT_HEADER *,
		      const COUNT_RATE_HEADER *,
		      const GATE_DIGI_HEADER *,
		      const CURRENT_CONTENT *,
		      const EVENT_RECORD *,
		      const COUNT_RATE_RECORD *, char *);
  void finalCleanTable();

  /***************************
       5rings.c
  ****************************/

  void make5rings(const ENCODING_HEADER *,
		  const EVENT_HEADER *,
		  EVENT_RECORD **, ENCODING_HEADER **);

  /***************************
         bitUtilities.c
  ****************************/

  u64 putBits(u8);
  u16 getBitsNb(u16);
  u32 poweru32(u8, u8);

  /***************************
        fillCoinciRecord.c
  ****************************/

  void fillCoinciRecord(const ENCODING_HEADER * pEncoHC,
			const EVENT_HEADER * pEHC,
			const GATE_DIGI_HEADER * pGDHC,
			EVENT_RECORD * first,
			EVENT_RECORD * second,
			u16 multipleID, int nN,
			int verboseLevel, EVENT_RECORD * pERC);


  /***************************
    fillCoinciRecordForGate.c
  ****************************/

  void fillCoinciRecordForGate(const ENCODING_HEADER * pEncoHC,
			       const EVENT_HEADER * pEHC,
			       const GATE_DIGI_HEADER * pGDHC,
			       EVENT_RECORD * first, EVENT_RECORD * second,
			       int verboseLevel, EVENT_RECORD * pERC);


  /***************************
  multipleCoincidencesSorter.cc
  ****************************/

  void setMultipleCoincidencesSorterParams();
  void setMultipleCoincidencesSorterParamsDirectly(u64 inline_cw, 
						   u64 inline_stack_cut_time,
						   u16 inline_maxDifSct,
						   u8 inline_multi);

  void multipleCoincidencesSorter(const ENCODING_HEADER * pEncoH,
				  const EVENT_HEADER * pEH,
				  const COUNT_RATE_HEADER * pCRH,
				  const GATE_DIGI_HEADER * pGDH,
				  const CURRENT_CONTENT * pcC,
				  const EVENT_RECORD * pER,
				  const COUNT_RATE_RECORD * pCRR);

  void cleanMultipleCoincidencesSorterList();

  /***************************
  multipleCoincidencesSorter.cc
  ****************************/

  void setCutTimeModuleParams(u8 inStartTimeBool,
			      u32 inStartTime, u32 inTimeDuration);

  void cutTimeModule(const EVENT_HEADER * pEH, EVENT_RECORD ** ppER);

  /***************************
      onlyKeep2Detectors.c
  ****************************/

  void setDetector1(u16 sct, u16 mod);
  void setDetector2(u16 sct, u16 mod);
  void onlyKeep2Detectors(const ENCODING_HEADER * pEncoH,
			  EVENT_RECORD ** ppER);


  /***************************
      changeAxialPos.c
  ****************************/

  void setNewAxialPos();
  void changeAxialPos(EVENT_RECORD * pER);

  /***************************
      changeAngularPos.c
  ****************************/

  void setNewAngularPos();
  void changeAngularPos(EVENT_RECORD * pER);

  /***************************
         daqBuffer.cc
  ****************************/

  void initDaqBuffer(u16 NiCards, u16 sctnb, u16 sct[]);
  void setDaqBufferMode(u8 inlineMode);
  void setDaqBufferSize(u32 inlineBufferSize);
  void setDaqBufferReadingFrequency(double inlineFreq);
  void daqBuffer(const ENCODING_HEADER *pEncoH,
		 EVENT_RECORD **ppER);
  void daqBufferDestructor();

  /***************************
         timeAnalyser.cc
  ****************************/

  void setInitialParamsForTimeAnalyser(void);
  void setInitialParamsForTimeAnalyserDirectly(u16 nbOfSct, double inline_epsilon);
  void timeAnalyser(const ENCODING_HEADER * pEncoH,
		    const EVENT_RECORD * pER);

  /***************************
      followCountsRate.cc
  ****************************/

  void setInitialParamsForFollowCountRates(void);
  void followCountRates(const ENCODING_HEADER * pEncoH,
		      EVENT_RECORD **ppER);
  void followCountRatesDestructor(void);

  /***************************
      temporalResolution.cc
  ****************************/

  void setInitialParamsForTemporalResolution(void);
  void setInitialParamsForTemporalResolutionDirectly(u16 layers, 
						     double *resol);
  void temporalResolution(const ENCODING_HEADER *pEncoH,
			  EVENT_RECORD *pER);
  void temporalResolutionDestructor(void);

  /***************************
      energyResolution.cc
  ****************************/

  void setInitialParamsForEnergyResolution(u16 inlineLayerNb,
					   mean_std *inlineEnergyResolBase);
  void energyResolution(const ENCODING_HEADER *pEncoH,
			EVENT_RECORD **ppER);
  void energyResolutionDestructor(void);

  /***************************
    peakPositionDispersion.cc
  ****************************/

  void setPeakPositionDispersionParams(u16 inlineLayerNb, mean_std *inlineCalBase);
  void peakPositionDispersion(const ENCODING_HEADER *pEncoH,
			      EVENT_RECORD **ppER);
  void peakPositionDispersionDestructor();

  /***************************
         sigmoidCut.cc
  ****************************/

  void setSigmoidCutParams(double inputAlpha, double inputX0,
			   double inputEAlpha, double inputEX0);
  void sigmoidCut(const ENCODING_HEADER *pEncoH,
		  EVENT_RECORD **ppER);
  void sigmoidCutDestructor();

  /***************************
    rejectFractionOfEvent.c
  ****************************/

  void setInitialParamForRejectFractionOfEvent(double inline_fraction);
  void rejectFractionOfEvent(EVENT_RECORD **ppER);
  void rejectFractionOfEventDestructor();

  /***************************
     DOImisIdentification.c
  ****************************/

  void setInitialParamForDOImisID(double misID_fraction);
  void DOImisID(const ENCODING_HEADER *pEncoH, EVENT_RECORD *pER);
  void DOImisIDdestructor();

#endif

#ifdef __cplusplus
}
#endif
