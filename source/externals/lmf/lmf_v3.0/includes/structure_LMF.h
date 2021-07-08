/*-------------------------------------------------------

           List Mode Format 
                        
     --  structure_LMF.h  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of structure_LMF.h:
	 structures used for binary part of LMF, in particular
         for the LMF record carrier.

---------------------------------------------------------------------------*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-*
|  List Mode Format Record Carrier      |
|  Declaration of the structures        |
|  used to build the LMF .ccs files     |
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  -  1.  Headers  Structures : all the parameters for an acquisition   =
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#ifdef __cplusplus
extern "C" {
#endif

/* 1.A This structure gives the ID's format */
  struct LMF_ccs_scanEncodingID {	/* rrrssssmmmcccccl */
    u8 bitForRsectors;		/* number of bits reserved for rings/sectors in ID */
    u16 maximumRsectors;	/* maximum number of sectors = 2 ** bitForRsectors */
    u8 bitForModules;		/* number of bits reserved for modules in ID */
    u16 maximumModules;		/* maximum number of modules = 2**bitForModules */
    u8 bitForSubmodules;	/* number of bits reserved for submodules in ID */
    u16 maximumSubmodules;	/* maximum number of submodules = 2 ** bitForSubmodules */
    u8 bitForCrystals;		/* number of bits reserved for crystals in ID */
    u16 maximumCrystals;	/* maximum number of crystals = 2** bitForCrystals */
    u8 bitForLayers;		/* number of bits reserved for layers in ID */
    u16 maximumLayers;		/* maximum number of layers = 2** bitForLayers */
  };

/*-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=*/
/* 1.B This structure gives the Scan Design */
  struct LMF_ccs_scannerTopology {
    u16 numberOfRings;		/* number of rings */
    u16 numberOfSectors;	/* number of sectors */
    u16 totalNumberOfRsectors;	/* number of rings * number of sectors */

  /*-=-=-=- FOR EACH RSECTOR : =-=-=-=-=-=*/
    u16 axialNumberOfModules;	/* number of modules axially */
    u16 tangentialNumberOfModules;	/* number of modules tangentially */
    u16 totalNumberOfModules;	/* number of modules axially * tangentially */

  /*-=-=-=- FOR EACH MODULE : =-=-=-=-=-=*/
    u16 axialNumberOfSubmodules;	/* number of submodules axially */
    u16 tangentialNumberOfSubmodules;	/* number of submodules tangentially */
    u16 totalNumberOfSubmodules;	/* number of submodules = Axial * Tangential */

  /*-=-=-=- FOR EACH SUBMODULE : =-=-=-=-=-=*/
    u16 axialNumberOfCrystals;	/* number of crystals axially */
    u16 tangentialNumberOfCrystals;	/* number of crystals tangentially */
    u16 totalNumberOfCrystals;	/* number of crystals = Axial * Tangential */

  /*-=-=-=- FOR EACH CRYSTAL : =-=-=-=-=-=*/

    u16 axialNumberOfLayers;	/* number of layers axially (always 1) */
    u16 radialNumberOfLayers;	/* number of layers radially */
    u16 totalNumberOfLayers;	/* number of layers */
  };


/* 1.C This structure gives the kind(s) of records the .ccs file do contain */
  struct LMF_ccs_scanContent {
    u8 nRecord;			/* number of different records */
    /* first record : event record */
    u8 eventRecordBool;		/* event recorded if 1 */
    u8 eventRecordTag;		/* event tag=0(1st bit of encoding event) */
    /* second record : countrate record */
    u8 countRateRecordBool;	/* countrate recorded if 1 */
    u8 countRateRecordTag;	/* countrate tag = 1000 (4 bits) */
    /* third record : Gate Digi record */
    u8 gateDigiRecordBool;	/* Gate single digi recorded if 1 */
    u8 gateDigiRecordTag;	/* countrate tag = 1100 (4 bits) */
    /* + eventually other records  */
  };

/* 1.D This structure contains these 3 last structures */
  struct LMF_ccs_encodingHeader {
    u8 scanEncodingIDLength;
    struct LMF_ccs_scanEncodingID scanEncodingID;	/*rrrssssmmmcccccl = 1110000111000001 */
    struct LMF_ccs_scannerTopology scannerTopology;	/* scanner Design */
    struct LMF_ccs_scanContent scanContent;	/* definition of the records */
  };
  typedef struct LMF_ccs_encodingHeader ENCODING_HEADER;

/* 1.E  This structure gives what's in the event record for building
   the event encoding pattern */
  struct LMF_ccs_eventHeader {
    u8 coincidenceBool;		/* Coincidence if 1, singles if 0 */
    u8 detectorIDBool;		/* detector ID recorded if 1, not recorded if 0 */
    u8 energyBool;		/* energy recorded if 1 */
    u8 neighbourBool;		/* energy of neighbours recorded if 1 */
    u8 neighbourhoodOrder;	/* 0, 1, 2 , or 3 (cf fig. 1) */
    u8 numberOfNeighbours;	/*Number of neighbours */
    u8 gantryAxialPosBool;	/* gantry's axial position */
    u8 gantryAngularPosBool;	/* gantry's angular position */
    u8 sourcePosBool;		/* source's position */
    u8 gateDigiBool;		/* gate Digi event record extension */
    u8 fpgaNeighBool;		/* store FPGA of juelich NEIGHBOUR info */

  };
  typedef struct LMF_ccs_eventHeader EVENT_HEADER;

/* 1.F  This structure gives what's in the countrate record for building
   the countrate encoding pattern */
  struct LMF_ccs_countRateHeader {
    u8 singleRateBool;		/* singles countrate recorded if =1 */
    u8 singleRatePart;		/* Ring (1), sector(2), module(3) or total (0) */
    u8 totalCoincidenceBool;	/* total coincidence recorded if =1 */
    u8 totalRandomBool;		/* total random rate recorded if =1 */
    u8 angularSpeedBool;	/* angular speed recorded if =1 */
    u8 axialSpeedBool;		/* axial speed recorded if =1 */
  };
  typedef struct LMF_ccs_countRateHeader COUNT_RATE_HEADER;


/* 1.G  This structure gives what's in the gate digi record for building
   the gate digi encoding pattern :  TTTT CSpe rGRR RRRR
TTTT = 1100 

C = 1 if number of compton stored
p = 1 if Source decay XYZ pos stored
S = 1 if source ID stored
e = 1 if eventID stored
r = 1 if runID stored
G = 1 if global dig XYZ pos stored
R = 0 reserved bit

Time always stored
*/

  struct LMF_ccs_gateDigiHeader {

    u8 comptonBool;		/* C number of compton */
    u8 comptonDetectorBool;	/* D number of compton in detector */
    u8 sourceIDBool;		/* s */
    u8 sourceXYZPosBool;	/* S */
    u8 eventIDBool;		/* e */
    u8 runIDBool;		/* r */
    u8 globalXYZPosBool;	/* G global xyz pos */
    u8 multipleIDBool;		/* M multiple ID */
  };
  typedef struct LMF_ccs_gateDigiHeader GATE_DIGI_HEADER;


/* + eventually other record's structures */

/* 1.E This structure contain what s carrid in the Carrier */
  struct LMF_ccs_currentContent {
    u8 typeOfCarrier;		/* = tag of event or countrate or gate digi */

  };
  typedef struct LMF_ccs_currentContent CURRENT_CONTENT;


										  /*-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=*//*-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=*/



  struct LMF_ccs_XYZpos {
    i16 X;
    i16 Y;
    i16 Z;
  };

  struct LMF_ccs_gateDigiRecord {

    u32 runID;			/* 4 bytes */
    u32 eventID[2];		/* 4 bytes */
    u16 sourceID[2];
    struct LMF_ccs_XYZpos sourcePos[2];
    struct LMF_ccs_XYZpos globalPos[42];
    u8 numberCompton[2];
    u8 numberDetectorCompton[2];

    u32 multipleID;		// 0 if singles or non multiple coincidence, multiple ID else


  };
  typedef struct LMF_ccs_gateDigiRecord GATE_DIGI_RECORD;


/* 2.A This structure can contain 1 event */

  struct LMF_ccs_eventRecord {
    u8 timeStamp[8];		/*   time stamp on 63 bits for singles, 23 for coincidence */
    u8 timeOfFlight;		/*  time of flight on 8 bits */
    u64 *crystalIDs;		/* crystal's ID (1st & 2nd and neighbours), 16 bits each */
    u8 *energy;			/* energy in each crystal, 8 bits each */
    u16 gantryAxialPos;		/* gantry's axial position, 16 bits */
    u16 gantryAngularPos;	/* gantry's angular position, 16 bits */
    u16 sourceAngularPos;	/* external source's angular position, 16 bits */
    u16 sourceAxialPos;		/* external source's axial position, 16 bits */
    u8 fpgaNeighInfo[2];	/* store fpga neighbour information */
    struct LMF_ccs_gateDigiRecord *pGDR;	/* extension of event record to accept GATE simul. infos */

  };
  typedef struct LMF_ccs_eventRecord EVENT_RECORD;


										  /*-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=*//*-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=*/

/* 3.A This  structure can contain 1 countrate */

  struct LMF_ccs_countRateRecord {

    u8 timeStamp[4];		/* time stamp */
    u16 totalSingleRate[2];	/* total single rate */
    u16 *pRsectorRate;		/* rsector's rate pointer */
    u16 *pModuleRate;		/* module's rate */
    u16 *pSubmoduleRate;	/* submodule's rate */
    u16 coincidenceRate;	/* coincidence rate */
    u16 randomRate;		/* random rate */
    u8 angularSpeed;		/* rotation gantry's speed */
    u8 axialSpeed;		/* axial speed's gantry */
  };
  typedef struct LMF_ccs_countRateRecord COUNT_RATE_RECORD;

/* This structure contains clock counts versus angular and axial positions of the gantry */

  typedef struct {
    u32 counts;
    u16 angularPos;
    u16 axialPos;
  } gantryEVENT;

/* This union use a number as an u64 or as 8 times u8 */

  typedef union {
    u16 w16;
    u8 w8[2];
  } u16vsu8;

  typedef union {
    u32 w32;
    u16 w16[2];
    u8 w8[4];
  } u32vsu8;

  typedef union {
    u64 w64;
    u8 w8[8];
  } u64vsu8;

  typedef union {
    u64 w64;
    u32 w32[2];
  } w64vsw32;

  typedef union {
    u64 w64;
    u16 w16[4];
  } u64vsu16;

  typedef union {
    u32 u32Tab[2];
    u8 u8Tab[8];
  } t32;

  typedef struct{
    double mean;
    double std;
  } mean_std;

/*-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=*/
/*-=-=-=-       END       =-=-=-=-=-=-=*/
/*-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=*/


#ifdef __cplusplus
}
#endif
