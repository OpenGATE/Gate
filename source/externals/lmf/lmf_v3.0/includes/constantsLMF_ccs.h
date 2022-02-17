/*-------------------------------------------------------

           List Mode Format 
                        
     --  constantsLMF_ccs.h  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of constantsLMF_ccs.h:
	 symbolic constants used for the binary part of LMF.



-------------------------------------------------------*/


/* allows to be linked with a C++ compiler */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef VERSION_LMF
#define VERSION_LMF 1
#define SUBVERSION_LMF 0
#endif



#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif


#ifndef MAXNEIGH
#define MAXNEIGH (21)
#endif				/* maximum number of crystal neighbours */


#ifndef _SETSCAN_H
#define _SETSCAN_H

/*     tag for different records */
#define EVENT_RECORD_TAG (0)
#define COUNT_RATE_RECORD_TAG (8)
#define GATE_DIGI_RECORD_TAG (12)

/*   how the id is made !        -=-=-=-=-       */
#define BITS_FOR_RSECTORS (4)
#define BITS_FOR_MODULES (2)
#define BITS_FOR_SUBMODULES (2)
#define BITS_FOR_CRYSTALS (6)
#define BITS_FOR_LAYERS (2)
/* The sum of these 5 last constants must be 16 ! */

#define NUMBER_OF_RINGS (1)
#define NUMBER_OF_SECTORS (8)
#define AXIAL_NUMBER_OF_MODULES (1)
#define TANGENTIAL_NUMBER_OF_MODULES (3)
#define AXIAL_NUMBER_OF_SUBMODULES (4)
#define TANGENTIAL_NUMBER_OF_SUBMODULES (1)
#define AXIAL_NUMBER_OF_CRYSTALS (8)
#define TANGENTIAL_NUMBER_OF_CRYSTALS (8)
#define AXIAL_NUMBER_OF_LAYERS (1)
#define RADIAL_NUMBER_OF_LAYERS (2)
#endif

#ifndef _NAME_FILES		/* name of .ccs files */
#define _NAME_FILES
#define COINCIDENCE_FILE "./testlmfCoinci.ccs"
#define REWRITE_FILE "./testInvert.ccs"
#define WRITE_FILE "./testlmf2.ccs"
#define WRITE_FILE2 "./testlmf2bis.ccs"
#endif


  /* constants used by builder (generateER.c) */
#ifndef _BUILD_REAL
#define _BUILD_REAL
#define ACTIVITY_FOR_TEST 10000000	/*  Bq */
#define DEAD_TIME_FOR_TEST 10000	/*  ps */
#define READING_TIME_FOR_TEST 1
#define DETECTOR_YIELD 0.7
#endif



  /* constants used by coincidence sorter */
#ifndef _COINCIDENCE_SORTER_CONDTANT
#define _COINCIDENCE_SORTER_CONDTANT
#define NULL_UP NULL
#define NULL_DOWN NULL
#define COINCIDENCE_WINDOW (10)	/* DELTA TAU */
#define STACK_CUT_TIME (1000000000)	/* DELTA T */
#define CS_VERBOSE_LEVEL (3)
#define SAVE_MULTIPLE_BOOL (0)
#define SAVE_AUTO_COINCI_BOOL (0)
#define HIGH_ACTIVITY_CUT_NUMBER (200000)	/* used in cleanListP1 */
#define SIZE_FOR_CRYSTAL (1)
#define SIZE_FOR_ENERGY (1)
#define CONVERT_TIME_COINCI (1.E-9)	/* from pico to milli seconds */
#define CONVERT_TIME_OF_FLIGHT (1.E-6)
#define CONVERT_TOF 390625   /* DAQ time step in femto seconds, used for time-of-flight encoding */
#define DEFAULT_DELAY_BY_RSECYOR (20000)
  /* in delayModule.c the default delay is 
     rsectorID*20000 Ex: rsector3 delayed 60000ps */


#endif

#define CPUFPGA_SHIFT       200000000	// cpu-fpga clocks delay value in 400psec unit for 2.5GHZ CPU
#define FULL_FPGA_DYNAMIC   0x3E7FFFFFC1	// cpu-fpga Clok's ratios (=62.5) x 0xFFFFFFFF
#define CLK2NSSHIFT	    6	// Clock period (25ns)/TimeMarkResolution(25/64 ns) CLK2NSFACTOR = 64

#ifndef GATE_LMF_TIME_STEP_PICOS
#define GATE_LMF_TIME_STEP_PICOS 1
#define GATE_LMF_TIME_STEP_COINCI 250
#define ASCII2LMF_TIME_STEP_PICOS 1
#endif





#ifndef GATE_LMF_ENERGY_STEP_KEV
#define GATE_LMF_ENERGY_STEP_KEV 3
#define ASCII2LMF_ENERGY_STEP_KEV 3
#endif

#ifndef ENERGY_REF
#define ENERGY_REF 511.
#endif


#ifndef BIT_DEF
#define BIT_DEF
#define MAX16BIT (65535)
#define BIT1 (1)
#define BIT2 (2)
#define BIT3 (4)
#define BIT4 (8)
#define BIT5 (16)
#define BIT6 (32)
#define BIT7 (64)
#define BIT7ET8 (192)
#define BIT8 (128)
#define BIT9 (256)
#define BIT10 (512)
#define BIT10ET11 (1536)
#define BIT11 (1024)
#define BIT12 (2048)
#define BIT13 (4096)
#define BIT14 (8192)
#define BIT15 (16384)
#define BIT16 (32768)
#endif

#ifdef __cplusplus
}
#endif
