/*-------------------------------------------------------

           List Mode Format 
                        
     --  prototypesLMF_cch.h  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of prototypesLMF_cch.h:
	 Functions used for the ascii part of LMF.
	 Fill in the LMF record carrier with the data contained in the LMF ASCII header file:
	 ->LMFcchReader - Read the scan file and fill in structures LMF_cch in the LMF Record Carrier
	 ->LMFcchReaderDestructor - Destroy structures LMF_cch in the LMF Record Carrier
	 ->usefulOptions = Function allocOfMemoryForLMF_Header - Allocation of memory to store the data 
	                                                         contained in the LMF cch data base
	                 + Function allocOfMemoryForLMF_cch - Allocation of memory to store the data 
			                                      contained in the cch file
			 + Function openFile - Opening a file 
			 + Function initialize - Initializing strings
			 + Function copyFile - Duplicate a file
			 + Function modifyDataInFile: modify a data in a .cch file
	 ->fileNameManager - Manage the ccs file name, the coinci_ccs file name and the bis_ccs file name
	 ->readTheLMFcchDataBase - Read the LMF cch data base (lmf_header.db) and store these informations 
                                   in the structures lmf_header
	 ->fillInFieldAndData - Fill in the members: field and data of the structure LMF_cch 
	 -> newFieldInCchFile = Function writeNewFieldInDataBase - Add a new field in the lmf_header data base 
	                      + Function defineFieldType - Define the type of the new field adding 
			                                   in lmf_header data base
	 ->testField - Comparison between the fields described in the input file and the fields store 
	               in the lmf_header data base
	 ->correctUnknownField - Correct the unknown fields described in the input file
	 ->dataConversion = Function defineUnitAndValue - Define which part of the data is the numerical 
	                                                  value and which part is the unit
	                  + Function definePrefiAndUnit - Define which part of the unit is the prefix and 
			                                  which part is the real unit
			  + Function findPrefixConversionFactor - Convert data in a default format 
			  + Function findUnitConversionFactor - Convert data in a default format
			  + modifyDateFormat - Convert date in a default format
			  + modifySpeedFormat - Separate in speed and rotation speed, numerator and denominator
			  + modifySurfaceOrVolumeFormat - Convert surface unit or volume unit in a distance unit
			  + modifyTimeFormat - Convert time and duration in a default format
	 ->fillInCchDefaultUnitValue - Set the default unit value in the members def_unit and def_unit_value 
	                               of the structure LMF_cch after the data conversion 
	 ->getLMF_cchData - Find a data in the structures LMF_cch 
	 ->editLMF_cchInfo = Function getToEditLMF_cchInfo - Find and print a value in the cch_file
                           + Function editLMF_cchData - Print structures LMF_cch
	 ->stringConversion - Fill in the members value and default_value of the LMF_cch structure 
	                      with a string 
	 ->numberConversion - Fill in the members value and default_value of the LMF_cch structure 
	                      with a number
         ->dateConversion - Fill in the members value and default_value of the LMF_cch structure 
	                    with a date
	 ->timeConversion - Fill in the members value and default_value of the LMF_cch structure 
	                              with a time
	 ->surfaceConversion - Fill in the members value and default_value of the LMF_cch structure 
	                       with a surface (numerical value + unit)
         ->volumeConversion - Fill in the members value and default_value of the LMF_cch structure 
	                      with a volume (numerical value + unit)

         ->speedConversion - Fill in the members value and default_value of the LMF_cch structure 
	                     with a speed (numerical value + unit)
	 ->rotationSpeedConversion - Fill in the members value and default_value of the LMF_cch structure 
	                             with a rotation speed (numerical value + unit)
				     
	 ->setShiftValues - set for each rsector a shift value
	 ->chooseColumnIndex - choose the shift way: x-axis, y-axis, z-axis 
	 ->findModuloNumber - if the shift concerns 1 ring in 2 or 1 ring in 3, modulo is respectively equal to 2 and 3 
	 ->findStartNumber - the shift starts with the ring number 1 or 2, startNumber is respectively equal to 1 and 2 
---------------------------------------------------------------------------*/

/* LMFcchReader - Read the scan file and fill in structures LMF_cch in the LMF Record Carrier */

#ifndef _LMFcchReader_h
#define _LMFcchReader_h

int LMFcchReader(i8 inputFile[charNum]);

#endif


/* LMFcchReaderDestructor - Destroy structures LMF_cch in the LMF Record Carrier */

#ifndef _LMFcchReaderDestructor_h
#define _LMFcchReaderDestructor_h

void LMFcchReaderDestructor();

#endif


/*   usefulOptions = */
/*   Function allocOfMemoryForLMF_Header - Allocation of memory to store the data contained 
                                           in the LMF cch data base */
/* + Function allocOfMemoryForLMF_cch - Allocation of memory to store the data contained in the cch file */
/* + Function openFile - Opening a file */
/* + Function initialize - Initializing strings */
/* + Function copyFile - Duplicate a file */

#ifndef _usefulOptions_h
#define _usefulOptions_h

lmf_header *allocOfMemoryForLMF_Header(int lmf_header_index);
LMF_cch *allocOfMemoryForLMF_cch(int cch_index);
int openFile(FILE * popnf, i8 fileName[charNum]);
void initialize(i8 buffer[charNum]);
int copyFile(i8 infileName[charNum], i8 copyingFileName[charNum]);
int modifyDataInFile(i8 dataDescription[charNum], i8 newData[charNum],
		     i8 file[charNum]);

#endif

/* fileNameManager - Manage the ccs file name, the coinci_ccs file name and the bis_ccs file name */

#ifndef _fileNameManager_h
#define _fileNameManager_h

int setFileName(i8 inputFile[charNum]);
int get_extension_ccs_FileName(i8 extension[charNum]);
int get_extension_cch_FileName(i8 extension[charNum]);
int copyNewCCHfile(i8[charNum]);

#endif

/* readTheLMFcchDataBase - Read the LMF cch data base (lmf_header.db) and store
   these informations in the structures lmf_header */

#ifndef _readTheLMFcchDataBase_h
#define _readTheLMFcchDataBase_h

int readTheLMFcchDataBase();

#endif

/* fillInFieldAndData - Fill in the members: field and data of the structure LMF_cch */

#ifndef _fillInFieldAndData_h
#define _fillInFieldAndData_h

int fillInFieldAndData();

#endif

/*   newFieldInCchFile = */
/*   Function writeNewFieldInDataBase - Add a new field in the lmf_header data base */
/* + Function defineFieldType - Define the type of the new field adding in lmf_header data base */


#ifndef _newFieldInCchFile_h
#define _newFieldInCchFile_h

int writeNewFieldInDataBase(FILE * lmf_header_infile, i8 field[charNum]);
int defineFieldType(i8 field[charNum]);

#endif

/* testField - Comparison between the fields described in the input file and the fields 
   store in the lmf_header data base */

#ifndef _testField_h
#define _testField_h

int testField(int last_lmf_header, int cch_index);

#endif


/* correctUnknownField -  Correct the unknown fields described in the input file */

#ifndef _correctUnknownField_h
#define _correctUnknownField_h

int correctUnknownField(i8 field[charNum]);

#endif


/*   dataConversion = */
/*   Function defineUnitAndValue - Define which part of the data is the numerical value and 
                                   which part is the unit */
/* + Function definePrefiAndUnit - Define which part of the unit is the prefix and 
                                   which part is the real unit */
/* + Function findPrefixConversionFactor - Convert data in a default format */
/* + Function findUnitConversionFactor - Convert data in a default format */
/* + modifyDateFormat - Convert date in a default format */
/* + modifySpeedFormat - Separate in speed and rotation speed, numerator and denominator */
/* + modifySurfaceOrVolumeFormat - Convert surface unit or volume unit in a distance unit */
/* + modifyTimeFormat - Convert time in a default format */
/* + modifyDurationFormat - Convert duration in a default format */

#ifndef _dataConversion_h
#define _dataConversion_h

content_data_unit defineUnitAndValue(i8 data[charNum]);

content_data_unit definePrefixAndUnit(i8 undefined_unit[charNum],
				      int unit_type);

double findPrefixConversionFactor(content_data_unit unit,
				  content_data_unit default_unit);

result_unit_conversion findUnitConversionFactor(content_data_unit unit,
						content_data_unit
						default_unit,
						int unit_type);
struct tm modifyDateFormat(i8 date[charNum], i8 field[charNum]);

complex_unit_type modifySpeedFormat(i8 undefined_unit[charNum]);

i8 *modifySurfaceOrVolumeFormat(i8 unit[charNum],
				int unit_type, i8 power[charNum]);

struct tm modifyTimeFormat(i8 time[charNum], i8 field[charNum]);

content_data_unit modifyDurationFormat(i8 duration[charNum],
				       i8 field[charNum]);

#endif

/* fillInCchDefaultUnitValue - Set the default unit value in the members 
   def_unit and def_unit_value of the structure LMF_cch after the data conversion */

#ifndef _fillInCchDefaultUnitValue_h
#define _fillInCchDefaultUnitValue_h

int fillInCchDefaultUnitValue(int last_lmf_header, int cch_index,
			      ENCODING_HEADER * pEncoHforGeometry);

#endif

/* getLMF_cchData - Find a data in the structures LMF_cch */

#ifndef _getLMF_cchData_h
#define _getLMF_cchData_h

contentLMFdata getLMF_cchNumericalValue(i8 field[charNum]);
/** !!!! only numerical value + unit 
    (energy, distance, surface, volume, time,  activity, speed, angle,
    rotation speed, weigth, temperature, electric field, magnetic field, pression) **/

int getLMF_cchInfo(i8 field[charNum]);
/** all type of data (date, string, number, energy, distance, surface, volume, time, activity, 
    speed, angle,rotation speed, weigth, temperature, electric field, magnetic field, pression) **/


#endif

/*   editLMF_cchInfo = */
/* + Function getToEditLMF_cchInfo - Find and print a value in the cch_file */
/* + Function editLMF_cchData - Print structures LMF_cch */

#ifndef _editLMF_cchInfo_h
#define _editLMF_cchInfo_h

int getToEditLMF_cchData(int last_lmf_header);

int editLMF_cchData(int last_lmf_header);

#endif

/* Function stringConversion - Fill in the members value and default_value of the LMF_cch structure 
   with a string */

#ifndef _stringConversion_h
#define _stringConversion_h

int stringConversion(int cch_index);

#endif

/* Function numberConversion - Fill in the members value and default_value of the LMF_cch structure 
   with a number */

#ifndef _numberConversion_h
#define _numberConversion_h

int numberConversion(int cch_index);

#endif

/* Function dateConversion - Fill in the members value and default_value of the LMF_cch structure 
   with a date */

#ifndef _dateConversion_h
#define _dateConversion_h

int dateConversion(int cch_index);

#endif

/* durationConversion - Fill in the members value and default_value of the LMF_cch structure
    with a duration */
#ifndef _durationConversion_h
#define _durationConversion_h

int durationConversion(int cch_index);

#endif

/* Function timeConversion - Fill in the members value and default_value of the LMF_cch structure 
   with a time */

#ifndef _timeConversion_h
#define _timeConversion_h

int timeConversion(int cch_index);

#endif

/* Function  surfaceConversion - Fill in the members value and default_value of the LMF_cch structure 
   with a surface (numerical value + unit) */

#ifndef _surfaceConversion_h
#define _surfaceConversion_h

int surfaceConversion(int cch_index, int dataType);

#endif

/* Function volumeConversion - Fill in the members value and default_value of the LMF_cch structure 
   with a volume (numerical value + unit) */

#ifndef _volumeConversion_h
#define _volumeConversion_h

int volumeConversion(int cch_index, int dataType);

#endif

/* Function speedConversion - Fill in the members value and default_value of the LMF_cch structure 
   with a speed (numerical value + unit) */

#ifndef _speedConversion_h
#define _speedConversion_h

int speedConversion(int cch_index, int dataType);

#endif

/* Function rotationSpeedConversion - Fill in the members value and default_value of the LMF_cch structure 
   with a rotation speed (numerical value + unit) */

#ifndef _rotationSpeedConversion_h
#define _rotationSpeedConversion_h

int rotationSpeedConversion(int cch_index, int dataType);

#endif

/* setShiftValues - set for each rsector a shift value */

#ifndef _setShiftValues_h
#define _setShiftValues_h

int setShiftValues(ENCODING_HEADER * pEncoHforGeometry, int cch_index);

#endif

/* chooseColumnIndex - choose the shift way: x-axis, y-axis, z-axis */

#ifndef _chooseColumnIndex_h
#define _chooseColumnIndex_h

int chooseColumnIndex(int cch_index);

#endif

/* findModuloNumber - if the shift concerns 1 ring in 2 or 1 ring in 3, modulo is respectively equal to 2 and 3 */

#ifndef _findModuloNumber_h
#define _findModuloNumber_h

int findModuloNumber(i8 stringbuf[charNum]);

#endif

/* findStartNumber - the shift starts with the ring number 1 or 2, startNumber is respectively equal to 1 and 2 */

#ifndef _findStartNumber_h
#define _findStartNumber_h

int findStartNumber(i8 stringbuf[charNum],
		    ENCODING_HEADER * pEncoHforGeometry,
		    int structure_index, int cch_index);

#endif


#ifndef _getEnergyStepFromCCH_h
#define _getEnergyStepFromCCH_h

int getEnergyStepFromCCH();

#endif
