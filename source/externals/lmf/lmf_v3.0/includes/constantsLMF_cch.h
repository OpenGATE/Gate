/*-------------------------------------------------------

           List Mode Format 
                        
     --  constantsLMF_cch.h  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of constantsLMF_cch.h:
     Variables and constants used for the ascii part of LMF.


-------------------------------------------------------*/

/*---------------- parameters.h ----------------*/

#ifndef _parameters_h
#define _parameters_h

#include <time.h>

#define FALSE 0
#define TRUE 1
#define charNum 256
#define HEADER_FILE "./lmf_header.db"
#define DEFAULT_UNITS_FILE "./constantsLMF_cch.h"


extern i8 fileName[charNum];

/*  ccs file names */
extern i8 ccsFileName[charNum];
extern i8 coinci_ccsFileName[charNum];
extern i8 bis_ccsFileName[charNum];

/* cch file names */
extern i8 cchFileName[charNum];
extern i8 coinci_cchFileName[charNum];
extern i8 bis_cchFileName[charNum];

/*----------------------------------------------*/

typedef struct {
  i8 field[charNum];
  int type;
} lmf_header;

extern lmf_header *plist_lmf_header;	/* declares plist_lmf_header to be of type pointer 
					   to (the structures) lmf_header */
extern lmf_header *first_lmf_header;	/* store the address of the begin 
					   of the structures lmf_header array */

extern struct tm *structCchTimeDate;

typedef union {
  i8 vChar[charNum];
  double vNum;
  struct tm tps_date;
} VALUE;

typedef struct {
  i8 field[charNum];
  i8 data[charNum];
  i8 unit[charNum];
  i8 def_unit[charNum];
  VALUE value;
  VALUE def_unit_value;
} LMF_cch;

extern LMF_cch *plist_cch;	/* declares plist_cch to be of type pointer to (structures) LMF_cch */
extern LMF_cch *first_cch_list;	/* declares plist_cch to be of type pointer 
				   to (structures) LMF_cch and to store the address of the begin
				   of the structures LMF_cch array */
extern int last_cch_list;	/* index of the last element of the structures LMF_cch array */

/* Data used to create a list of Shift values */

extern double **ppShiftValuesList;
extern int lastShiftValuesList_index;


typedef struct {
  double value;
  i8 prefix[charNum];
  i8 unit[charNum];
} content_data_unit;

typedef struct {
  double numericalValue;
  i8 unit[charNum];
} contentLMFdata;

typedef struct {
  double factor;
  double constant;
} result_unit_conversion;

typedef struct {
  i8 numerator[charNum];
  i8 denominator[charNum];
} complex_unit_type;


#endif

/*=-=-=-=-=-=-=- default_units =-=-=-=-=-=-=-=-=*/

#ifndef _default_units_h
#define _default_units_h

/* Units are identified by their string representations 
   and concatenated with a standard prefix : */
/* a, f, p, n, mu, m, c, d, da, h, k, M, G, T, P, 
   wich defines the order of magnitude */

#define ENERGY_UNIT "keV"	/* eV, J, Wh, cal, erg */
#define DISTANCE_UNIT "mm"	/* m, in, ft */
#define DURATION_UNIT "ps"	/* h, min, s */
#define TIME_UNIT "s"		/* h, min, s, ns, ps ... */
#define ACTIVITY_UNIT "MBq"	/* Bq, Ci */
#define SPEED_UNIT "mm/s"	/* [distance]/[time] */
#define ANGLE_UNIT "rad"	/* degree, rad, grad */
#define ROTATION_SPEED_UNIT "rps"	/* [angle]/[time], rph, rpm, rps */
#define WEIGHT_UNIT "g"		/* g, oz, lb */
#define SURFACE_UNIT "mm2"	/* [distance]2 */
#define VOLUME_UNIT "mm3"	/* [distance]3 */
#define TEMPERATURE_UNIT "C"	/* C, F, K */
#define ELECTRIC_FIELD_UNIT "V"	/* V */
#define MAGNETIC_FIELD_UNIT "T"	/* gauss, T */
#define PRESSION_UNIT "hPa"	/* Pa, atm, bar, mmHg */

#endif

/*=-=-=-=-=-=-=- units.h =-=-=-=-=-=-=-=-=-=-=-=*/

#ifndef _units_h
#define _units_h

extern i8 symbol_units_list[20][5][7];
extern int nb_units[20];

#endif

/*=-=-=-=-=-=-=- prefixFactors.h =-=-=*/

#ifndef _prefixFactors_h
#define _prefixFactors_h

extern i8 standard_prefix_list[16][3];
extern double standard_prefix_factors_list[16][16];

#endif

/*=-=-=-=-=-=-=- unitsFactors.h =-=-*/

#ifndef _unitsFactors_h
#define _unitsFactors_h

extern double units_conversion_factors_list[20][5][5];
extern double units_conversion_constants_list[20][5][5];

#endif


/*=-=-=-=-=-=-=  error_messages.h =-=-=-=-=-=-=-*/

#ifndef _error_messages_h
#define _error_messages_h

#define ERROR1 "ERROR: Cannot open the file \"%s\".\n"
#define ERROR2 "ERROR: the file name %s is wrong, you must choose a name without extension, like \"test\" or \"gate_data\".\nPlease try again.\n"
#define ERROR3 "\n ERROR:Dynamic allocation of memory to the file \"%s\" failed.\n"
#define ERROR4 "Incorrect field: Field description must have less than %d i8acters\n"
#define ERROR5 "Please correct or add this information in the file \"%s\" and run the program.\n"
#define ERROR6 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a date and  must follow the pattern:\n\tfield description: Month/Day/Year\nThe month name can be Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov and Dec.\nThe day must be an integer between 1 and 31.\n\tex: scan date: Feb/01/2002 or scan date: Feb/1/02\n"
#define ERROR7 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a time and must follow the pattern:\n\tfield description: Hour:Minute:Second or Hour h Minute min Second s.\n\tHour must be an integer between 0 and 23.\n\tMinute must be an integer between 0 and 59.\n\tSecond must be an integer between 0 and 59.\n\tex: injection time: 14:23:45 or scan start time: 14 h 23 min 45 s\n"
#define ERROR8 "ERROR: on the value on the field \"%s\".\nIn (\"%s\") Second must be an integer between 0 and 59.\n\tex: 14:23:45 or 14 h 23 min 45 s\n"
#define ERROR9  "ERROR: on the value of the field \"%s\".\nIn (\"%s\") Minute must be an integer between 0 and 59.\n\tex: 14:23:45 or 14 h 23 min 45 s\n"
#define ERROR10 "ERROR: on the value of the field \"%s\".\nIn (\"%s\") Hour must be an integer between 0 and 23.\n\tex: 14:23:45 or 14 h 23 min 45 s\n"
#define ERROR11 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a speed and must follow the pattern:\n\tfield description: numerical value [Distance_unit]/[Time_unit]\nThe distance unit can be m,in and ft with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P) and the time unit can be h, min and s.\n\tex: bed axial speed: 2 cm/s.\n"
#define ERROR12 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a rotation speed and must follow the pattern:\n\tfield description: numerical value [Angle_unit]/[Time_unit]\n\tor field description: numerical value [rps] or [rpm] or [rph]\nThe angle unit can be degree,rad and grad with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\nThe time unit can be h, min and s.\n\tex: detector rotation speed: 30 rpm or detector rotation speed: 3,15 rad/s.\n"
#define ERROR13 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a surface and must follow the pattern:\n\tfield description: numerical value [Distance_unit]2\nThe distance unit can be m,in and ft with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: detector surface: 45mm2\n"
#define ERROR14 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a volume and must follow the pattern:\n\tfield description: numerical value [Distance_unit]3\nThe distance unit can be m,in and ft with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex:subject volume: 45mm3\n"
#define ERROR15 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be an energy and must follow the pattern:\n\tfield description: numerical value [Energy_unit]\nThe energy unit can be eV, J, Wh, cal and erg with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: energy threshold: 511keV\n"
#define ERROR16 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be an activity and must follow the pattern:\n\tfield description: numerical value [Activity_unit]\nThe activity unit can be Bq and Ci with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: injected dose: 10 mCi\n"
#define ERROR17 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a weigth and must follow the pattern:\n\tfield description: numerical value [Weigth_unit]\nThe weigth unit can be g, oz and lb with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: subject weigth: 150 g\n"
#define ERROR18 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a distance and must follow the pattern:\n\tfield description: numerical value [Distance_unit]\nThe distance unit can be m, in and ft with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: bed overlap: 45 mm\n"
#define ERROR19  "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be an angle and must follow the pattern:\n\tfield description: numerical value [Angle_unit]\nThe angle unit can be degree, rad and grad with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: angle: 45 degree\n"
#define ERROR20 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a temperature and must follow the pattern:\n\tfield description: numerical value [Temperature_unit]\nThe temperature unit can be C, F and K  with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: detector temperature: 25 C\n"
#define ERROR21 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be an electric field and must follow the pattern:\n\tfield description: numerical value [Electric_field_unit]\nThe electric field unit can be V with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: voltage: 25 mV\n"
#define ERROR22 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a magnetic field and must follow the pattern:\n\tfield description: numerical value [Magnetic_field_unit]\nThe magnetic field unit can be gauss and T with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: magnetic field: 1.5 T\n"
#define ERROR23 "ERROR, on the value of the field \"%s\".\nThe data \"%s\" should be a pression and must follow the pattern:\n\tfield description: numerical value [Pression_unit]\nThe pression unit can be Pa, atm, bar and mmHg with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: pression: 760 mmHg\n"
#define ERROR24 "ERROR, on the value of the field \"%s\".\nThe data \"%s\" is wrong.\nUnits must have a standard prefix such as a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P.\n\tex: mm or keV\n"
#define ERROR25 "ERROR: You must choose a default unit in \"%s\".\n"
#define ERROR26 "ERROR: \"%s\" is wrong. To write information in your input file \"%s\", you must follow the pattern;\n\tfield description: data\n\tex: scan type: emission or scan date: Feb/14/2002\n"
#define ERROR27 "ERROR: on the value of the field \"%s\", the data \"%s\" should be a number.\n"
#define ERROR28 "ERROR: the field \"%s\" isn't found in the file \"%s\".\n"
#define ERROR29 "ERROR: adding the new field \"%s\" in the file \"%s\" failed !!\n"
#define ERROR30 "ERROR: adding new result of the event position calculation in the file \"%s\" failed !!"
#define ERROR31 "ERROR: the field \"%s\" isn't found in the cch file (file.cch).\n"
#define ERROR32 "ERROR: the scan file name give in the keyboard \"%s\" and the scan file name store in the cch file \"%s\" are different.\n"
#define ERROR33 "ERROR: the file \"%s%s\" can't be create.\nYou can only create the file \"%s_coinci.ccs\" or the file \"%s_bis.ccs\".\nPlease try again.\n"
#define ERROR34 "\nERROR: this field \"%s\" doesn't exist in the file \"%s\".\n"
#define ERROR35 "ERROR: The file \"%s\" can't be create.\n"
#define ERROR36 "ERROR: In the function LMFcchReader, allocation of memory failed !!!\n"
#define ERROR37 "ERROR: on the value of the field \"%s\". The data \"%s\" should be a shift information and must follow the pattern:\n\tfield description: numerical value [Distance_unit]\n\tIn this case, the field description can be: X shift ring 1 (all the rsectors contained in the ring 1 are shifted on the x-axis), Y shift sector 2 or Z shift rsector 4.\nThe distance unit can be m, in and ft with a standard prefix (a,f,p,n,mu,m,c,d,da,h,k,M,G,T,P).\n\tex: X shift sector 2: 0.5 mm\n"
#define ERROR38 "ERROR: on the value of the field \"%s\".This field should be a shift information and must follow the pattern:\n\tfield description: numerical value [Distance_unit]\n\tIn this case, the field description can be: x shift (all the rsectors in the scanner geometry are shifted on the x-axis), y shift, z shift \nx shift ring 1 (all the rsectors contained in the ring 1 are shifted on the x-axis), y shift ring 1, z shift ring 1\nx shift sector 2, y shift sector 2, z shift sector 2\nx shift rsector 4, y shift rsector 4, z shift rsector 4.\n\tex: x shift sector 2: 0.5 mm\n"
#define ERROR39 "ERROR: on the value of the field \"%s\". The data \"%s\" should be a rsector shift information and must follow the pattern:\n\tfield description: numerical value [Distance_unit]\n\tIn this case, the field description can be: x shift rsector \"rsector number\", y shift rsector \"rsector number\"\nor z shift rsector \"rsector number\"mod \"number\".\n\tex: x shift rsector 0 mod 2: 0.5 mm, so the rsectors  0, 2, 4, 6, 8 ... are shifted on the x-axis\n"
#define ERROR40 "ERROR: To define the first ring which is shifted, you must choose an ring index between 0 and %d.\n"
#define ERROR41 "ERROR: To define the first sector which is shifted, you must choose an sector index between 0 and %d.\n"
#define ERROR42 "ERROR: To define the first rsector which is shifted, you must choose an rsector index between 0 and %d.\n"
#define ERROR43 "ERROR: The ring number defined to start the shift, must be choose between 0 and %d.\n"
#define ERROR44 "ERROR: The rsector number defined to start the shift, must be choose between 0 and %d.\n"
#define ERROR45 "ERROR: The sector number defined to start the shift, must be choose between 0 and %d.\n"
#define ERROR46 "ERROR: the field description \"%s\" is wrong, it's impossible to define a modulo number egal to 0.\n"
#define ERROR47 "ERROR: the field \"%s\" is wrong, you must define the first structure which is shifted.\n"
#define ERROR48 "ERROR: on the value of the field \"%s\".\nThe data \"%s\" should be a duration and must follow the pattern:\n\tfield description: Hour h Minute min Second s \n\tor numerical value + unit.\n\tHour, Minute and Second must be integer.\n\tex: scan duration: 14 h 23 min 45 s \n\tor time step: 150 ps or clock step: 20 ns\n"
#endif
#define ERROR49 "ERROR: on the value on the field \"%s\".\nIn (\"%s\") Second must be an integer between 0 and 59.\n\tex: 14 h 23 min 45 s\n"
#define ERROR50  "ERROR: on the value of the field \"%s\".\nIn (\"%s\") Minute must be an integer between 0 and 59.\n\tex: 14 h 23 min 45 s\n"
#define ERROR51 "ERROR: on the value of the field \"%s\".\nIn (\"%s\") Hour must be an integer between 0 and 23.\n\tex: 14 h 23 min 45 s\n"
