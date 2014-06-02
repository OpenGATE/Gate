/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*----------------------
   Modifications history

     Gate 6.2

	C. Comtat, CEA/SHFJ, 10/02/2011	   Allows for span 1 (means less slices per segment)

                                           Allows for an interfile-like ("ecat8") output instead of ecat7.
					   It does not require the ecat library! (GATE_USE_ECAT7 not set)
----------------------*/

#include "GateConfiguration.h"

#include "GateSinoToEcat7.hh"
#include "GateSinoToEcat7Messenger.hh"

#include "globals.hh"
#include "G4RunManager.hh"
#include "G4Run.hh"

#include "GateOutputMgr.hh"
#include "GateTools.hh"
#include "GateEcatSystem.hh"
#include "GateToSinogram.hh"
#include "GateSinogram.hh"
#include "GateVVolume.hh"
#include "GateVolumePlacement.hh"


GateSinoToEcat7::GateSinoToEcat7(const G4String& name, GateOutputMgr* outputMgr,GateEcatSystem* itsSystem,DigiMode digiMode)
  : GateVOutputModule(name,outputMgr,digiMode)
  , m_system(itsSystem)
  , m_fileName(" ") // All default output file from all output modules are set to " ".
                    // They are then checked in GateApplicationMgr::StartDAQ, using
                    // the VOutputModule pure virtual method GiveNameOfFile()
  , m_isotope_code("")
{
   m_isEnabled = false; // Keep this flag false: all output are disabled by default
   m_asciiMessenger = new GateSinoToEcat7Messenger(this);
   nVerboseLevel = 0;
   #ifdef GATE_USE_ECAT7
   m_ptr = NULL;
   #endif
   mh = NULL;
   sh = NULL;
   m_mashing = -1;
   m_span = -1;
   m_maxRingDiff = -1;
   m_segmentNb = 0;
   m_delRingMinSeg = NULL;
   m_delRingMaxSeg = NULL;
   m_zMinSeg = NULL;
   m_zMaxSeg = NULL;
   m_segment = NULL;
   m_ecatCameraNumber = 0;
   m_isotope_halflife = 0.0;
   m_isotope_branching_fraction = 1.0;
   #ifdef GATE_USE_ECAT7
   m_ecatVersion = 7;
   #else
   m_ecatVersion = 8;
   #endif
   if (nVerboseLevel > 0) G4cout << " >> GateSinoToEcat7 created" << G4endl;
}

GateSinoToEcat7::~GateSinoToEcat7()
{
  delete m_asciiMessenger;
  #ifdef GATE_USE_ECAT7
  if (m_ptr != NULL) {
    if (nVerboseLevel > 1) G4cout << " >> ECAT7 file " << m_ptr->fname << " will be closed" << G4endl;
    matrix_close(m_ptr);
  }
  #endif
  if (mh != NULL) free(mh);
  if (sh != NULL) free(sh);
  if (nVerboseLevel > 0) G4cout << " >> GateSinoToEcat7 deleted" << G4endl;
}

const G4String& GateSinoToEcat7::GiveNameOfFile()
{
  return m_fileName;
}

void GateSinoToEcat7::RecordBeginOfAcquisition()
{

  // 24.03.2006 C. Comtat, study start time
  GateToSinogram* setMaker = m_system->GetSinogramMaker();

  if (!(m_system->GetSinogramMaker()->IsEnabled())) return;
  if (nVerboseLevel > 0) G4cout << " >> entering [GateSinoToEcat7::RecordBeginOfAcquisition]" << G4endl;

  // Create main header
  mh = (Main_header *) calloc(1,sizeof(Main_header));
  if (mh == NULL) {
     G4Exception( "GateSinoToEcat7::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, "Could not allocate memory for main header");
  }
  if (nVerboseLevel > 2) G4cout << "    Memory allocated for main header " << G4endl;
  // Fill main header
  FillMainHeader();
  if (nVerboseLevel > 2) G4cout << "    Main header filled" << G4endl;
  // Create subheader
  sh = (Scan3D_subheader *) calloc(1,sizeof(Scan3D_subheader));
  if (sh == NULL) {
     G4Exception( "GateSinoToEcat7::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, "Could not allocate memory for subheader");
  }
  if (nVerboseLevel > 2) G4cout << "    Memory allocated for sub header " << G4endl;
  sh->frame_duration = 0;

  // 24.03.2006 C. Comtat, study start time
  sh->frame_start_time = (int) (setMaker->GetStudyStartTime() / second * 1000.0);

  #ifdef GATE_USE_ECAT7
  // Create ECAT7 file and write the main header
  if (m_ecatVersion == 7) {
    m_ptr = matrix_create((m_fileName+".S").c_str(),MAT_CREATE,mh);
    if (m_ptr == NULL) {
			G4String msg = "Could not create ECAT7 file '"+m_fileName+".S' !";
     G4Exception( "GateSinoToEcat7::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, msg );
    }
    if (nVerboseLevel > 1) G4cout << "    ECAT7 file " << m_fileName << ".S created" << G4endl;
  } else {
    m_ptr = NULL;
  }
  #endif
  if (nVerboseLevel > 0) G4cout << " >> leaving [GateSinoToEcat7::RecordBeginOfAcquisition]" << G4endl;
}

void GateSinoToEcat7::RecordEndOfAcquisition()
{
  if (!(m_system->GetSinogramMaker()->IsEnabled())) return;
  if (nVerboseLevel > 0) G4cout << " >> entering [GateSinoToEcat7::RecordEndOfAcquisition]" << G4endl;

  // Delete main header
  free(mh);
  mh = NULL;
  // Delete subheader
  free(sh);
  sh = NULL;
  #ifdef GATE_USE_ECAT7
  // Close the ECAT file
  if (m_ptr != NULL) {
    if (nVerboseLevel > 1) G4cout << "    ECAT7 file " << m_ptr->fname << " will be closed" << G4endl;
    matrix_close(m_ptr);
    m_ptr = NULL;
  }
  #endif
  if (nVerboseLevel > 0) G4cout << " >> leaving [GateSinoToEcat7::RecordEndOfAcquisition]" << G4endl;
}

void GateSinoToEcat7::RecordBeginOfRun(const G4Run * )
{
  GateToSinogram* setMaker = m_system->GetSinogramMaker();
  G4int  seg,segm,segment_occurance,nplane;
  if (!(m_system->GetSinogramMaker()->IsEnabled())) return;
  if (nVerboseLevel > 0) G4cout << " >> entering [GateSinoToEcat7::RecordBeginOfRun]" << G4endl;
  if (nVerboseLevel > 1) {
    G4cout << "    Frame ID:     " << setMaker->GetSinogram()->GetCurrentFrameID() << G4endl;
    G4cout << "    Gate ID:      " << setMaker->GetSinogram()->GetCurrentGateID() << G4endl;
    G4cout << "    Bed position: " << setMaker->GetSinogram()->GetCurrentBedID() << G4endl;

    // 07.02.2006, C. Comtat, Store randoms and scatters sino
    if (setMaker->IsTruesOnly()) {
      G4cout << "    Only true prompt coincidences are recorded" << G4endl;
    } else if (setMaker->IsStoreDelayeds()) {
      G4cout << "    Prompt coincidences are recorded with data = " << setMaker->GetSinogram()->GetCurrentDataID() << G4endl;
      G4cout << "    Delayed coincidences (if simulated) are recorded with data = " << setMaker->GetSinoDelayeds()->GetCurrentDataID() << G4endl;
    } else {
      G4cout << "    Prompt minus delayed (if simulated) coincidences are recorded" << G4endl;
    }
    if (setMaker->IsStoreScatters()) {
      G4cout << "    True scattered coincidences are recorded with data = " << setMaker->GetSinoScatters()->GetCurrentDataID() << G4endl;
    }


  }
  nplane = 2*setMaker->GetRingNb() - 1;
  if (m_mashing <= 0) m_mashing = 1;
  if (m_maxRingDiff < 0) m_maxRingDiff = setMaker->GetRingNb()-1;
  // CC, 10.02.2011 : allows for span 1
  if (m_span < 1) m_span = 3;
  if (m_maxRingDiff >= setMaker->GetRingNb()) {
    G4cout << " !!! [GateSinoToEcat7]: maximum ring difference (" << m_maxRingDiff
           << ") is too big (should be <" << setMaker->GetRingNb() << ")" << G4endl;
    G4Exception("GateSinoToEcat7::RecordBeginOfRun", "RecordBeginOfRun", FatalException,"Could not fill subheader");
  }
  if ((m_span<1)||((float)((m_span-1)/2)) != (((float)m_span-1.0)/2.0)) {
    G4cout << " !!! [GateSinoToEcat7]: span factor (" << m_maxRingDiff
           << ") should be odd" << G4endl;
    G4Exception("GateSinoToEcat7::RecordBeginOfRun", "RecordBeginOfRun", FatalException,"Could not fill subheader");
  }
  if ((float)((2*m_maxRingDiff+1-m_span)/(2*m_span)) != (2.0*(float)m_maxRingDiff+1.0-(float)m_span)/(2.0*(float)m_span)) {
    G4int bin,maxRingDiff;
    G4cout << " !!! [GateSinoToEcat7]: maximum ring difference (" << m_maxRingDiff
           << ") is not coherent with span factor (" << m_span << ")" << G4endl;
    G4cout << "                                       possible maximum ring differences are: " << G4endl;
    bin = 0;
    maxRingDiff = (m_span-1)/2;
    while (maxRingDiff < setMaker->GetRingNb()) {
      G4cout << "                                           " << maxRingDiff << G4endl;
      bin++;
      maxRingDiff = ((2*bin+1)*m_span-1)/2;
    }
    G4Exception("GateSinoToEcat7::RecordBeginOfRun", "RecordBeginOfRun", FatalException,"Could not fill subheader");
  }
  m_segmentNb = (2*m_maxRingDiff+1-m_span)/(2*m_span);
  if (m_delRingMinSeg != NULL) free(m_delRingMinSeg);
  if (m_delRingMaxSeg != NULL) free(m_delRingMaxSeg);
  if (m_zMinSeg != NULL) free(m_zMinSeg);
  if (m_zMaxSeg != NULL) free(m_zMaxSeg);
  if (m_segment != NULL) free(m_segment);
  m_delRingMinSeg = (int *) calloc(2*m_segmentNb+1,sizeof(int));
  m_delRingMaxSeg = (int *) calloc(2*m_segmentNb+1,sizeof(int));
  m_zMinSeg = (int *) calloc(2*m_segmentNb+1,sizeof(int));
  m_zMaxSeg = (int *) calloc(2*m_segmentNb+1,sizeof(int));
  m_segment = (int *) calloc(2*m_segmentNb+1,sizeof(int));
  seg = 0;
  m_segment[0] = 0;
  segment_occurance = 0;
  m_delRingMinSeg[segment_occurance] = ((2*m_segment[segment_occurance]-1)*m_span+1)/2;
  m_delRingMaxSeg[segment_occurance] = ((2*m_segment[segment_occurance]+1)*m_span-1)/2;
  m_zMinSeg[segment_occurance] = 0;
  m_zMaxSeg[segment_occurance] = nplane-1;
  for (seg=1;seg<=m_segmentNb;seg++) {
    for (segm=seg;segm>=-seg;segm-=2*seg) {
      segment_occurance++;
      m_segment[segment_occurance] = segm;
      m_delRingMinSeg[segment_occurance] = ((2*m_segment[segment_occurance]-1)*m_span+1)/2;
      m_delRingMaxSeg[segment_occurance] = ((2*m_segment[segment_occurance]+1)*m_span-1)/2;
      m_zMinSeg[segment_occurance] = m_zMinSeg[0] + ((2*seg-1)*m_span+1)/2;
      m_zMaxSeg[segment_occurance] = m_zMaxSeg[0] - ((2*seg-1)*m_span+1)/2;
    }
  }
  if (nVerboseLevel > 0) G4cout << " >> leaving [GateSinoToEcat7::RecordBeginOfRun]" << G4endl;
}

void GateSinoToEcat7::RecordEndOfRun(const G4Run * )
{
  GateToSinogram* setMaker = m_system->GetSinogramMaker();
  if (!(m_system->GetSinogramMaker()->IsEnabled())) return;
  if (nVerboseLevel > 0) G4cout << " >> entering [GateSinoToEcat7::RecordEndOfRun]" << G4endl;
  sh->prompts = 0;
  sh->delayed = 0;
  FillData(setMaker->GetSinogram());

  // 07.02.2006, C. Comtat, Store randoms and scatters sino
  if (setMaker->IsStoreDelayeds()) FillData(setMaker->GetSinoDelayeds());
  if (setMaker->IsStoreScatters()) FillData(setMaker->GetSinoScatters());

  if (nVerboseLevel > 0) G4cout << " >> leaving [GateSinoToEcat7::RecordEndOfRun]" << G4endl;
}

/* Overload of the base-class' virtual method to print-out a description of the module

	indent: the print-out indentation (cosmetic parameter)
*/
void GateSinoToEcat7::Describe(size_t indent)
{
  GateVOutputModule::Describe(indent);
  G4cout << GateTools::Indent(indent) << " >> Job:                   write a set of 2D sinograms into an Ecat7 output file" << G4endl;
  G4cout << GateTools::Indent(indent) << " >> Is enabled ?           " << ( IsEnabled() ? "Yes" : "No") << G4endl;
  G4cout << GateTools::Indent(indent) << " >> File name:             " << m_fileName << ".S" << G4endl;
  G4cout << GateTools::Indent(indent) << " >> Attached to system:    " << m_system->GetObjectName() << G4endl;
}

// Fill the main header
void GateSinoToEcat7::FillMainHeader()
{
  time_t       timer;
  struct tm   *date;

  GateToSinogram* setMaker = m_system->GetSinogramMaker();


  // Retrieve the parameters of the blocks and crystals
  GateSystemComponent* blockComponent    = m_system->GetMainComponent();
  GateArrayComponent*   crystalComponent = m_system->GetDetectorComponent();
  G4ThreeVector         crystalPitchVector = crystalComponent->GetRepeatVector();
  G4int                 RingNb = blockComponent->GetLinearRepeatNumber() * crystalComponent->GetRepeatNumber(2) + setMaker->GetVirtualRingPerBlockNb()*(blockComponent->GetLinearRepeatNumber()-1);

  if (setMaker->GetRingNb() != RingNb) {
    G4cout << " !!! [GateSinoToEcat7::FillMainHeader]: Number of rings per block: " << crystalComponent->GetRepeatNumber(2) << G4endl;
    G4cout << " !!! [GateSinoToEcat7::FillMainHeader]: Number of axial blocks: " << blockComponent->GetLinearRepeatNumber() << G4endl;
    G4cout << " !!! [GateSinoToEcat7::FillMainHeader]: Number of virtual rings between blocks: " << setMaker->GetVirtualRingPerBlockNb() << G4endl;
    G4cout << " !!! [GateSinoToEcat7::FillMainHeader]: Number of rings from Sinogram: " << setMaker->GetRingNb() << G4endl;

     G4Exception("GateSinoToEcat7::FillMainHeader", "FillMainHeader", FatalException,"Uncoherent crystal rings number");
  }
  if (nVerboseLevel > 1) {
    G4cout << "    Number of crystal rings: " << RingNb << G4endl;
    G4cout << "    crystal axial pitch: " << crystalPitchVector.z()/mm << " mm" << G4endl;
    G4cout << "    crystal transverse pitch: " << crystalPitchVector.y()/mm << " mm" << G4endl;
  }

  strncpy(mh->original_file_name,(m_fileName+".S").c_str(),32);
  mh->original_file_name[31]='\0';
  if (m_ecatVersion == 7) mh->sw_version = 72;
  else mh->sw_version = (short) 8;
  mh->system_type = m_ecatCameraNumber;
  mh->file_type = Short3dSinogram;
  strncpy(mh->serial_number,"GATE",10);
  mh->serial_number[9]='\0';
  timer = time(NULL);
  date = (struct tm*) localtime(&timer);
  date->tm_mday = 31;
  date->tm_mon = 11;
  date->tm_year = 69;
  date->tm_hour = 13;
  date->tm_min = 0;
  date->tm_sec = 0;
  mh->scan_start_time = (int) difftime(time(NULL),mktime(date));
  mh->intrinsic_tilt =  0;
  mh->transm_source_type = 0;
  mh->distance_scanned = crystalPitchVector.z()/cm * RingNb;
  mh->transaxial_fov = 0.0;
  mh->angular_compression = 0;
  if (setMaker->IsStoreDelayeds()) {
    mh->coin_samp_mode = 1; /* prompts and delayed */
  } else {
    mh->coin_samp_mode = 0; /* net trues */
  }
  mh->axial_samp_mode = 0;
  mh->calibration_factor = 0.0;
  mh->calibration_units = 0;
  mh->calibration_units_label = 0;
  mh->compression_code = 0;
  mh->acquisition_type = StaticEmission;
  mh->patient_orientation = 3;  /* Head first Supine */
  mh->num_planes = 1;
  mh->num_frames = 1; // This field will be updating for each new frame
  mh->num_gates = 1;
  mh->num_bed_pos = 0;
  mh->init_bed_position = 0.0;
  mh->plane_separation = crystalPitchVector.z()/cm/2.0;
  mh->bin_size = crystalPitchVector.y()/cm/2.0;
  mh->lwr_true_thres = 0;
  mh->upr_true_thres = 0;
  mh->acquisition_mode = 0;
  mh->branching_fraction = m_isotope_branching_fraction;
  mh->isotope_halflife = m_isotope_halflife/s;
  strncpy(mh->isotope_code,m_isotope_code,8);
  mh->isotope_code[7] ='\0';
  mh->septa_state = SeptaRetracted;
}

// Fill the data
void GateSinoToEcat7::FillData(GateSinogram* setSino)
{

  GateToSinogram* setMaker = m_system->GetSinogramMaker();
  G4int  bin,seg,segment_occurance,data_size,nz,frame,data,
         tot_data_size,file_pos,offset,ringdiff,ring_1_min,ring_1_max,
	 view,ring_1,ring_2,elem,z,sinoID,bin_sdata,bin_m_data,nsino;
#ifdef GATE_USE_ECAT7
  G4int  plane,gate,bed,csize;
  char   *cdata=NULL;
#endif
  short  *sdata;
  G4String frameFileName;
  char             ctemp[512];
  std::ofstream    m_dataFile,m_headerFile;

  #ifdef GATE_USE_ECAT7
  struct MatDir matdir, dir_entry;
  int    matnum,nblks,blkno;
  #endif
  GateSinogram::SinogramDataType *m_data, *m_randoms;

  // Fill subheader
  frame = setSino->GetCurrentFrameID();
#ifdef GATE_USE_ECAT7
  plane = 1;
  gate = setSino->GetCurrentGateID();
  bed = setSino->GetCurrentBedID();
#endif
  data = setSino->GetCurrentDataID();
  seg = 0;
  if (m_ecatVersion == 7) sh->data_type = SunShort;
  sh->num_dimensions = 4;
  sh->num_r_elements = setMaker->GetRadialElemNb();
  sh->num_angles = setMaker->GetCrystalNb()/2/m_mashing;
  // CC, 10.02.2011 : allows for span 1
  if (m_span == 1) {
    sh->num_z_elements[seg] = (m_zMaxSeg[seg] - m_zMinSeg[seg])/2 + 1;
    for (seg=1;seg<=m_segmentNb;seg++) sh->num_z_elements[seg] = 2*((m_zMaxSeg[2*seg-1] - m_zMinSeg[2*seg-1])/2 + 1);
  } else {
    sh->num_z_elements[seg] = m_zMaxSeg[seg] - m_zMinSeg[seg] + 1;
    for (seg=1;seg<=m_segmentNb;seg++) sh->num_z_elements[seg] = 2*(m_zMaxSeg[2*seg-1] - m_zMinSeg[2*seg-1] + 1);
  }
  sh->ring_difference = m_maxRingDiff;
  sh->axial_compression = m_span;
  sh->scale_factor = 1.0;
  sh->scan_min = -1;
  sh->scan_max = -1;
  sh->x_resolution = mh->bin_size;
  // CC, 10.02.2011 : allows for span 1
  if (m_span == 1) {
    sh->z_resolution = 2*mh->plane_separation;
  } else {
    sh->z_resolution = mh->plane_separation;
  }
  sh->corrections_applied = 0;
  if (data == 0) sh->frame_start_time += sh->frame_duration; // increment from last frame duration
  sh->frame_duration = (int) (setMaker->GetFrameDuration() / second * 1000.0);
  if (nVerboseLevel > 1) {
    G4cout << "    Frame ID:                             " << setSino->GetCurrentFrameID() << G4endl;
    G4cout << "    Gate ID:                             " << setSino->GetCurrentGateID() << G4endl;
    G4cout << "    Data ID:                             " << setSino->GetCurrentDataID() << G4endl;
    G4cout << "    Bed ID:                             " << setSino->GetCurrentBedID() << G4endl;
    G4cout << "    Number of sinogram radial elements:   " << sh->num_r_elements << G4endl;
    G4cout << "    Number of sinogram azimutal elements: " << sh->num_angles << G4endl;
    G4cout << "    Maximum ring difference:              " << m_maxRingDiff << G4endl;
    G4cout << "    Span factor:                          " << m_span << G4endl;
    G4cout << "     ==> Number of segments               " << 2*m_segmentNb+1 << G4endl;
    for (segment_occurance=0;segment_occurance<(2*m_segmentNb+1);segment_occurance++) {
      G4cout << "              Segment " << m_segment[segment_occurance] << " is for ring difference "
             << m_delRingMinSeg[segment_occurance] << " to " << m_delRingMaxSeg[segment_occurance] << G4endl;
      G4cout << "                and goes from slice " << m_zMinSeg[segment_occurance] << " to slice "
             << m_zMaxSeg[segment_occurance] << G4endl;
    }
    G4cout << "    Frame start time:                       " << sh->frame_start_time << " msec" << G4endl;
    G4cout << "    Frame duration:                       " << sh->frame_duration << " msec" << G4endl;
  }
  // CC, 10.02.2011 : allows for span 1
  if (m_span == 1) {
    data_size = ((m_zMaxSeg[0] - m_zMinSeg[0])/2 + 1) * sh->num_r_elements * sh->num_angles;
  } else {
    data_size = (m_zMaxSeg[0] - m_zMinSeg[0] + 1) * sh->num_r_elements * sh->num_angles;
  }
  sdata = (short*) calloc(sizeof(short),data_size);
#ifdef GATE_USE_ECAT7
  if (m_ecatVersion == 7) cdata = (char*) calloc(sizeof(short),data_size);
#endif
  tot_data_size = 0;
  seg = 0;
  while (sh->num_z_elements[seg]>0) {
    tot_data_size += sh->num_r_elements*sh->num_angles*sh->num_z_elements[seg];
    seg++;
  }
  #ifdef GATE_USE_ECAT7
  if (m_ecatVersion == 7) {
    nblks = (tot_data_size*sizeof(short)+MatBLKSIZE-1)/MatBLKSIZE;
    matnum = mat_numcod(frame,plane,gate,data,bed);
    if (matrix_find(m_ptr,matnum,&matdir) == -1) { // the matrix does not already exist
      // create new matrix entry
      blkno = mat_enter(m_ptr->fptr,m_ptr->mhptr,matnum,nblks+1);
      dir_entry.matnum = matnum;
      dir_entry.strtblk = blkno;
      dir_entry.endblk  = dir_entry.strtblk + nblks + 1;
      dir_entry.matstat = 1;
      insert_mdir(dir_entry,m_ptr->dirlist);
      matdir = dir_entry;
    }
    // write subheader
    mat_write_Scan3D_subheader(m_ptr->fptr,m_ptr->mhptr,matdir.strtblk,sh);
  } else {
  #endif
    if (data == 0 && mh->coin_samp_mode == 0) {
      sprintf(ctemp,"%s_frame%0d.tr",m_fileName.c_str(),(int) setSino->GetCurrentFrameID()-1);
    } else if (data == 1) {
      sprintf(ctemp,"%s_frame%0d.ra",m_fileName.c_str(),(int) setSino->GetCurrentFrameID()-1);
    } else if (data == 4) {
      sprintf(ctemp,"%s_frame%0d.sc",m_fileName.c_str(),(int) setSino->GetCurrentFrameID()-1);
    } else {
      sprintf(ctemp,"%s_frame%0d",m_fileName.c_str(),(int) setSino->GetCurrentFrameID()-1);
    }
    frameFileName = ctemp;
    G4cout << "    sinograms written to the interfile-like files " << frameFileName << ".s and " << frameFileName << ".s.hdr" << G4endl;
    m_dataFile.open((frameFileName+".s").c_str(),std::ios::out | std::ios::trunc | std::ios::binary);
  #ifdef GATE_USE_ECAT7
  }
  #endif
  offset = 0;
  // loop on the segments
  for (segment_occurance=0;segment_occurance<(2*m_segmentNb+1);segment_occurance++) {
    // CC, 10.02.2011 : allows for span 1
    if (m_span == 1) {
      nz = (m_zMaxSeg[segment_occurance] - m_zMinSeg[segment_occurance])/2 + 1;
    } else {
      nz = m_zMaxSeg[segment_occurance] - m_zMinSeg[segment_occurance] + 1;
    }
    data_size = nz * sh->num_r_elements * sh->num_angles;
    for (bin=0;bin<data_size;bin++) sdata[bin] = 0;
    // loop on the ring differences
    for (ringdiff = m_delRingMinSeg[segment_occurance]; ringdiff <= m_delRingMaxSeg[segment_occurance]; ringdiff++) {
      if (ringdiff <= 0) {
        ring_1_min = -ringdiff;
        ring_1_max = setMaker->GetRingNb() - 1;
      } else {
        ring_1_min = 0;
        ring_1_max = setMaker->GetRingNb() - ringdiff -1;
      }
      // loop on the azimuthal angle
      for (view=0;view<sh->num_angles*m_mashing;view++) {
        // loop on the axial position
	for (ring_1 = ring_1_min; ring_1 <= ring_1_max; ++ring_1) {
	  ring_2 = ring_1 + ringdiff;
	  z = ring_1 + ring_2 - m_zMinSeg[segment_occurance];
	  // sinoID = ring_1 + ring_2 * setMaker->GetRingNb();
	  sinoID = setSino->GetSinoID(ring_1,ring_2);
	  if (sinoID < 0 || sinoID >= (G4int) setSino->GetSinogramNb()) {
	    G4Exception("GateToSinogram::FillData", "FillData", FatalException, "Wrong 2D sinogram ID");
	  }
	  if (nVerboseLevel>2 && view==0) {
            G4cout << " >> ring difference " << ringdiff << ", slice " << z << G4endl;
            G4cout << "    rings " << ring_1 << "," << ring_2  << " give sino ID " << sinoID << G4endl;
	  }
          m_data = setSino->GetSinogram(sinoID);
	  bin_m_data = view * setMaker->GetRadialElemNb(); // sino ordering
          // CC, 10.02.2011 : allows for span 1
          if (m_span == 1) {
            if (m_ecatVersion == 7) {
	      bin_sdata = (z/2) * sh->num_r_elements + view / m_mashing * nz * sh->num_r_elements; // view ordering
            } else {
	      bin_sdata = view / m_mashing * sh->num_r_elements + (z/2) * sh->num_angles * sh->num_r_elements; // sino ordering
            }
          } else {
            if (m_ecatVersion == 7) {
 	      bin_sdata = z * sh->num_r_elements + view / m_mashing * nz * sh->num_r_elements; // view ordering
            } else {
	      bin_sdata = view / m_mashing * sh->num_r_elements + z * sh->num_angles * sh->num_r_elements; // sino ordering
            }
          }
	  for (elem=0; elem<sh->num_r_elements; elem++) sdata[bin_sdata+elem] += (short int) m_data[bin_m_data+elem];
	}
      }
      for (ring_1 = ring_1_min; ring_1 <= ring_1_max; ++ring_1) {
        ring_2 = ring_1 + ringdiff;
	sinoID = setSino->GetSinoID(ring_1,ring_2);
	if (sinoID < 0 || sinoID >= (G4int) setSino->GetSinogramNb()) {
	  G4Exception("GateToSinogram::FillData", "FillData", FatalException,   "Wrong 2D sinogram ID");
	}
	m_randoms = setSino->GetRandoms();
        if (data == 0) {
          // only valid for the prompt (or net trues) matrix (data = 0)
          // for the delayed (data = 1) or scattered coincidences, use numbers from data = 0
	  sh->delayed += (short int) m_randoms[sinoID];
	}
      }
    }
    if (segment_occurance == 0) {
      sh->scan_min = sh->scan_max = sdata[0];
    }
#ifdef GATE_USE_ECAT7
    csize = 0;
#endif
    for (bin=0;bin<data_size;bin++) {
      // ecat7: convert short --> SunShort
      #ifdef GATE_USE_ECAT7
      if (m_ecatVersion == 7) bufWrite_s(sdata[bin],cdata,&csize);
      #endif
      if (sdata[bin] < sh->scan_min) sh->scan_min = sdata[bin];
      else if (sdata[bin] > sh->scan_max) sh->scan_max = sdata[bin];
      if (data == 0) {
        // only valid for the prompt (or net trues) matrix (data = 0)
        // for the delayed (data = 1) or scattered coincidences, use numbers from data = 0
        sh->prompts += sdata[bin];
      }
    }
    // write data segment
    #ifdef GATE_USE_ECAT7
    if (m_ecatVersion == 7) {
      file_pos = (matdir.strtblk+1)*MatBLKSIZE + offset;
      if (fseek(m_ptr->fptr,file_pos,0) == EOF) {
        G4cout << " !!! [GateSinoToEcat7::FillData]: can not seek position " << file_pos
               << " for " << m_ptr->fname << G4endl;
        G4Exception("GateSinoToEcat7::FillData", "FillData", FatalException, "Could not fill data");
      } else if (fwrite(cdata,sizeof(short),data_size,m_ptr->fptr) != (size_t) data_size) {
        G4cout << " !!! [GateSinoToEcat7::FillData]: can not write segment " << segment_occurance
               << " in " << m_ptr->fname << G4endl;
        G4Exception("GateSinoToEcat7::FillData", "FillData", FatalException, "Could not fill data");
      }
    } else {
    #endif
      file_pos = offset;
      m_dataFile.seekp(file_pos,std::ios::beg);
      if ( m_dataFile.bad() ) G4Exception( "\n[GateToSinogram]:\n", "FillData", FatalException,
      	      	              	          "Could not write sinograms onto the disk (out of disk space?)!\n");
      m_dataFile.write((const char*)(sdata),data_size*sizeof(short) );
      if ( m_dataFile.bad() ) G4Exception( "\n[GateToSinogram]:\n", "FillData", FatalException,
      	      	              	           "Could not write sinograms onto the disk (out of disk space?)!\n");
      m_dataFile.flush();
    #ifdef GATE_USE_ECAT7
    }
    #endif
    offset += data_size * sizeof(short);
  }
  // update scan_min and scan_max
  sh->net_trues = sh->prompts - sh->delayed;
  if (nVerboseLevel > 1) {
    G4cout << "    Number of prompts in ECAT sinogram                    " << sh->prompts << G4endl;
    G4cout << "    Number of delayeds in ECAT sinogram                   " << sh->delayed << G4endl;
    G4cout << "    Number of net trues in ECAT sinogram                  " << sh->net_trues << G4endl;
    G4cout << "    Maximum bin content:                  " << sh->scan_max << G4endl;
  }
  #ifdef GATE_USE_ECAT7
  if (m_ecatVersion == 7) {
    mat_write_Scan3D_subheader(m_ptr->fptr,m_ptr->mhptr,matdir.strtblk,sh);
    if (mh_update(m_ptr)) {
      G4cout << " !!! [GateSinoToEcat7::FillData]: can not update main header in  "
             << m_ptr->fname << G4endl;
      G4Exception("GateSinoToEcat7::FillData", "FillData", FatalException, "Could not fill data");
    }
    free(cdata);
  } else {
  #endif
    m_dataFile.close();
    // ASCII header
    m_headerFile.open((frameFileName+".s.hdr").c_str(),std::ios::out | std::ios::trunc | std::ios::binary);
    m_headerFile << "!INTERFILE" << G4endl;
    m_headerFile << "%comment:=Created from " << mh->serial_number << G4endl;
    m_headerFile << "!name of data file := "  << (frameFileName+".s").c_str() << G4endl;
    m_headerFile << "!originating system := " << mh->system_type << G4endl;
    // Time corresponding to the start of the acquisition, stored in mh->scan_start_time
    time_t timer = time(NULL);
    struct tm  *date,*date0;
    date0 = (struct tm*) localtime(&timer);
    date0->tm_mday = 31;
    date0->tm_mon = 11;
    date0->tm_year = 69;
    date0->tm_hour = 13;
    date0->tm_min = 0;
    date0->tm_sec = 0;
    time_t t1 = mktime(date0);
    time_t t = mh->scan_start_time + t1;
    date = (struct tm*) localtime(&t);
    m_headerFile << "%study date (yyyy:mm:dd):=" << 1900+date->tm_year << ":" << 1+date->tm_mon << ":" << date->tm_mday << G4endl;
    m_headerFile << "%study time (hh:mm:ss):=" <<  date->tm_hour << ":" << date->tm_min << ":" << date->tm_sec << G4endl;
    m_headerFile << "!PET data type := emission" << G4endl;
    m_headerFile << "data format := sinogram" << G4endl;
    m_headerFile << "number format := signed integer" << G4endl;
    m_headerFile << "!number of bytes per pixel := 2" << G4endl;
    m_headerFile << "!image duration (sec) := " << sh->frame_duration/1000 << G4endl;
    m_headerFile << "!image relative start time (sec) := " << sh->frame_start_time/1000 << G4endl;
    m_headerFile << "isotope name := " << mh->isotope_code << G4endl;
    m_headerFile << "isotope gamma halflife (sec) := " << mh->isotope_halflife << G4endl;
    m_headerFile << "isotope branching factor := " << mh->branching_fraction << G4endl;
    m_headerFile << "number of dimensions := 3" << G4endl;
    m_headerFile << "matrix size [1] := " << sh->num_r_elements << G4endl;
    m_headerFile << "matrix size [2] := " << sh->num_angles << G4endl;
    nsino = 0;
    for (bin=0; bin<64; bin++) nsino += sh->num_z_elements[bin];
    m_headerFile << "matrix size [3] := " << nsino << G4endl;
    m_headerFile << "scaling factor (mm/pixel) [1] := " << sh->x_resolution*10. << G4endl;
    m_headerFile << "scaling factor (mm/pixel) [2] := " << 1.0 << G4endl;
    m_headerFile << "scaling factor (mm/pixel) [3] := " << sh->z_resolution*10. << G4endl;
    m_headerFile << "axial compression := "<< sh->axial_compression << G4endl;
    m_headerFile << "maximum ring difference := " << sh->ring_difference << G4endl;
    m_headerFile << "number of rings := " << setMaker->GetRingNb() << G4endl;
    m_headerFile << "number of segments := "<< 2*m_segmentNb+1 << G4endl;
    m_headerFile << "segment table := {" << sh->num_z_elements[0];
    bin = 1;
    while ((sh->num_z_elements[bin] > 0) && (bin<64)) {
      m_headerFile << "," << sh->num_z_elements[bin]/2 << "," << sh->num_z_elements[bin]/2;
      bin++;
    }
    m_headerFile << "}" << G4endl;
    if (data == 0 && mh->coin_samp_mode == 0)  m_headerFile << "scan data type description[1]:=net trues" << G4endl;
    else if (data == 0 && mh->coin_samp_mode == 1)  m_headerFile << "scan data type description[1]:=prompts" << G4endl;
    else if (data == 1) m_headerFile << "scan data type description[1]:=delayed" << G4endl;
    else if (data == 4) m_headerFile << "scan data type description[1]:=scatters" << G4endl;
    m_headerFile << "frame := " << frame - 1 << G4endl;
    m_headerFile << "total prompts := " << sh->prompts << G4endl;
    m_headerFile << "total randoms := " << sh->delayed << G4endl;
    m_headerFile << "total net trues := "<< sh->net_trues << G4endl;
    m_headerFile.close();
  #ifdef GATE_USE_ECAT7
  }
  #endif
  free(sdata);
}
