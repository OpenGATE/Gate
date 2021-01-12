
#ifndef EcatModel_h
#define EcatModel_h

/* 20-JUL-2004: Unused variable removed by M. Sibomana
// maximum number of buckets
static int MaxBuckets = 56;
// maximum number of crystals in the axial direction
static int MaxAxialCrystals = 32;
static int MaxCrossPlanes = 6;
*/

enum TransmissionSource {none, Ring, Rod};
enum Septa {NoSepta, Fixed, Retractable};
enum PPorder {PP_LtoR, PP_RtoL};

typedef struct _EcatModel {
	char *number;  			/* model number as an ascii string */
	int rings;			/* number of rings of buckets */
	int nbuckets;			/* total number of buckets */
	int transBlocksPerBucket;	/* transaxial blocks per bucket */
	int axialBlocksPerBucket;	/* axial blocks per bucket */
	int blocks;
	int tubesPerBlock;		/* PMTs per block */
	int axialCrystalsPerBlock;	/* number of crystals in the axial direction */
	int angularCrystalsPerBlock;	/* number of transaxial crystals */
	int dbStartAddr;		/* bucket database start address */
	int dbSize;			/* size of bucket database in bytes */
	int timeCorrectionBits;		/* no. bits used to store time correction */
	int maxcodepage;		/* number of highest code page */
	enum PPorder ppOrder;		/* display order for position profile */

	int dirPlanes;			/* number of direct planes */
	int def2DSpan;			/* default span for 2D plane definitions */
	int def3DSpan;			/* default span for 3D plane definitions */
	int defMaxRingDiff;		/* default maximum ring difference for 3D */
	enum TransmissionSource txsrc;  /* transmission source type */
	float txsrcrad;			/* transmission source radius */
	enum Septa septa;		/* septa type */
	int defElements;		/* default number of elements */
	int defAngles;			/* default number of angles */
	int defMashVal;			/* default angular compression (mash) value */
	float defAxialFOV;		/* default axial FOV (one bed pos in cm) */
	float transFOV;			/* transaxial FOV */
	float crystalRad;		/* detector radius */
	float maxScanLen;		/* maximum allowed axial scan length (cm) */
	int defUld;			/* default ULD */
	int defLld;			/* default LLD */
	int defScatThresh;		/* default scatter threshold */
	float planesep;			/* plane separation */
	float binsize;			/* bin size (spacing of transaxial elements) */
	float pileup;			/*pileup correction factor for count losses */
	float planecor;			/* plane correction factor for count losses */
	int rodoffset;			/* Rod encoder offset of zero point */
	float hbedstep;			/*  horizontal bed step */
	float vbedoffset;		/* vertical bed step */
	float intrTilt;			/* intrinsic tilt */
	int wobSpeed;			/* default wobble speed */
	int tiltZero;			/* tilt Zero */
	int rotateZero;			/* rotate Zero */
	int wobbleZero;			/* wobble Zero */
	int bedOverlap;			/* number of planes to overlap between bed positions */
	int prt;			/* flag to indicate if scanner is partial ring tomograph */

/* parameters from (in 70rel3) 	*/
	float blockdepth;		/*  */
	int minelements;		/*  */
	int bktsincoincidence;		/*  */
	int cormask;			/*  */
	int analogasibucket;		/*  */

/* geometric correction parameters for normalization (in 70rel0)
	int corrBump;
	float cOffset;
	float integTime;
	float bCorr;
	int sBump;
	int eBump;
*/

} EcatModel;
#if 0

#if defined(__STDC__) || defined(__cplusplus)
#if defined(__cplusplus)
extern "C" {
#endif	/* __cplusplus */
EcatModel *ecat_model(int);
#if defined(__cplusplus)
}
#endif	/* __cplusplus */
#else /* __STDC__ */
extern EcatModel *ecat_model();
#endif  /* __STDC__ */

#endif
#endif

