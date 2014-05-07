/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*----------------------
   Modifications history

     Gate 6.2

	C. Comtat, CEA/SHFJ, 10/02/2011	   Allows for an interfile-like ("ecat8") output instead of ecat7.
					   It does not require the ecat library! (GATE_USE_ECAT7 not set)
----------------------*/

#include "GateConfiguration.h"

#ifndef GateSinoToEcat7_H
#define GateSinoToEcat7_H

#include <fstream>
#include "G4Timer.hh"

#include "GateVOutputModule.hh"

#include <stdio.h>

//ECAT7 include file
#ifdef GATE_USE_ECAT7
#include "matrix.h"
#include "machine_indep.h"
#else
// CC, 10.02.2011 : some structures of ECAT7 in matrix.h are included here if Gate is linked without the ecat library.
//                  They are required for the interfile-like output
enum    DataSetType {
	NoData, Sinogram, PetImage, AttenCor, Normalization,
	PolarMap, ByteVolume, PetVolume, ByteProjection,
	PetProjection, ByteImage, Short3dSinogram, Byte3dSinogram, Norm3d,
	Float3dSinogram,InterfileImage, NumDataSetTypes};
enum    MatrixDataType {
	UnknownMatDataType, ByteData, VAX_Ix2, VAX_Ix4,
	VAX_Rx4, IeeeFloat, SunShort, SunLong,  NumMatrixDataTypes,
	UShort_BE, UShort_LE, Color_24, Color_8, BitData};
enum    ScanType {UndefScan, BlankScan,
        TransmissionScan, StaticEmission,
        DynamicEmission, GatedEmission,
        TransRectilinear, EmissionRectilinear,
        NumScanTypes};
enum    SeptaPos {SeptaExtended, SeptaRetracted, NoSeptaInstalled};
typedef struct XMAIN_HEAD {
	char magic_number[14];
	char original_file_name[32];
	short sw_version;
	short system_type;
	short file_type;
	char serial_number[10];
	short align_0;
	unsigned int scan_start_time;
	char isotope_code[8];
	float isotope_halflife;
	char radiopharmaceutical[32];
	float gantry_tilt;
	float gantry_rotation;
	float bed_elevation;
	float intrinsic_tilt;
	short wobble_speed;
	short transm_source_type;
	float distance_scanned;
	float transaxial_fov;
	short angular_compression;
	short coin_samp_mode;
	short axial_samp_mode;
	short align_1;
	float calibration_factor;
	short calibration_units;
	short calibration_units_label;
	short compression_code;
	char study_name[12];
	char patient_id[16];
	char patient_name[32];
	char patient_sex[1];
	char patient_dexterity[1];
	float patient_age;
	float patient_height;
	float patient_weight;
	int patient_birth_date;
	char physician_name[32];
	char operator_name[32];
	char study_description[32];
	short acquisition_type;
	short patient_orientation;
	char facility_name[20];
	short num_planes;
	short num_frames;
	short num_gates;
	short num_bed_pos;
	float init_bed_position;
	float bed_offset[15];
	float plane_separation;
	short lwr_sctr_thres;
	short lwr_true_thres;
	short upr_true_thres;
	char user_process_code[10];
	short acquisition_mode;
	short align_2;
	float bin_size;
	float branching_fraction;
	unsigned int dose_start_time;
	float dosage;
	float well_counter_factor;
	char data_units[32];
	short septa_state;
	short align_3;
} Main_header;
typedef struct X3DSCAN_SUB {
	short data_type;
	short num_dimensions;
	short num_r_elements;
	short num_angles;
	short corrections_applied;
	short num_z_elements[64];
	short ring_difference;
	short storage_order;
	short axial_compression;
	float x_resolution;
	float v_resolution;
	float z_resolution;
	float w_resolution;
	unsigned int gate_duration;
	int r_wave_offset;
	int num_accepted_beats;
	float scale_factor;
	short scan_min;
	short scan_max;
	int prompts;
	int delayed;
	int multiples;
	int net_trues;
	float tot_avg_cor;
	float tot_avg_uncor;
	int total_coin_rate;
	unsigned int frame_start_time;
	unsigned int frame_duration;
	float loss_correction_fctr;
	float uncor_singles[128];
} Scan3D_subheader;
#endif

class GateSinoToEcat7Messenger;
class GateEcatSystem;
class GateToSinogram;
class GateSinogram;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateSinoToEcat7 :  public GateVOutputModule
{
public:
  GateSinoToEcat7(const G4String& name, GateOutputMgr* outputMgr,GateEcatSystem* itsSystem,DigiMode digiMode);
  virtual ~GateSinoToEcat7(); //!< Destructor
  const G4String& GiveNameOfFile();

  //! It opens the ECAT7 file
  void RecordBeginOfAcquisition();
  //! It closes the ECAT7 file.
  void RecordEndOfAcquisition();

  //! Reset the data array
  void RecordBeginOfRun(const G4Run *);
  //! Saves the latest results
  void RecordEndOfRun(const G4Run *);

  void RecordBeginOfEvent(const G4Event * ) {}
  void RecordEndOfEvent(const G4Event * ) {}
  void RecordStepWithVolume(const GateVVolume * , const G4Step *) {}


  //! saves the geometry voxel information
  void RecordVoxels(GateVGeometryVoxelStore *) {};

  //! Get the output file name
  const  G4String& GetFileName()                { return m_fileName;       };
  //! Set the output file name
  void   SetFileName(const G4String& aName)     { m_fileName = aName;      };
  //! Set azimutal mashing factor
  void SetMashing(const G4int aNumber)          { m_mashing = aNumber;     };
  //! Set span factor
  void SetSpan(const G4int aNumber)             { m_span = aNumber;        };
  //! Set maximum ring difference
  void SetMaxRingDiff(const G4int aNumber)      { m_maxRingDiff = aNumber; };
  //! Set ECAT camera number
  void SetEcatCameraNumber(const G4int aNumber) { m_ecatCameraNumber = aNumber; } ;
  //! Set the isotope code
  void SetIsotopeCode(const G4String& aName)    { m_isotope_code = aName;   };
  //! Set the isotope half-life
  void SetIsotopeHalflife(const G4double aNumber) { m_isotope_halflife = aNumber; };
  //! Set the isotope branching fraction
  void SetIsotopeBranchingFraction(const G4double aNumber) { m_isotope_branching_fraction = aNumber; };
  #ifdef GATE_USE_ECAT7
  //! Set ecat version
  void SetEcatVersion(const G4int aNumber) { m_ecatVersion = aNumber; } ;
  #endif
  /*! \brief Overload of the base-class' virtual method to print-out a description of the module

	\param indent: the print-out indentation (cosmetic parameter)
  */
  void Describe(size_t indent=0);

  //! Fill the main header
  void FillMainHeader();
  //! Write the GATE specific scanner information into the main header

  //! Fill the sub-header and data
  void FillData(GateSinogram* m_sinogram);

private:
  GateSinoToEcat7Messenger* m_asciiMessenger;

  //! Pointer to the system, used to get the system information and the sinogram
  GateEcatSystem *m_system;

  G4String m_fileName;

  #ifdef GATE_USE_ECAT7
  MatrixFile *m_ptr;
  #endif
  Main_header *mh;
  Scan3D_subheader *sh;

  G4int                 m_mashing;           //! azimutal mashing factor
  G4int                 m_span;              //! polar mashing factor
  G4int                 m_maxRingDiff;       //! maximum ring difference

  G4int                 m_segmentNb;
  G4int                *m_delRingMinSeg;
  G4int                *m_delRingMaxSeg;
  G4int                *m_zMinSeg;
  G4int                *m_zMaxSeg;
  G4int                *m_segment;
  G4int                 m_ecatCameraNumber;   //! Unique number identifying the scanner type, required by ecat7 reconstruction software
  G4String              m_isotope_code;       //! Isotope code to be stored in ecat7 main header
  G4double              m_isotope_halflife;   //! Isotope half-life to be stored in ecat7 main header
  G4double              m_isotope_branching_fraction; //! Isotope branching fration to be stored in ecat7 main header
  //#ifdef GATE_USE_ECAT7
  G4int                 m_ecatVersion; //! version V7: regular ecat7 file, version V8: interfile-like file
  //#endif
  //std::ofstream     	      	m_headerFile; 	      	    //!< Output stream for the header file
  //std::ofstream     	      	m_dataFile;   	      	    //!< Output stream for the data file


};
#endif
