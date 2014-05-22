/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_ECAT7

#include "GateSinoAccelToEcat7.hh"
#include "GateSinoAccelToEcat7Messenger.hh"

#include "globals.hh"
#include "G4RunManager.hh"
#include "G4Run.hh"

#include "GateOutputMgr.hh"
#include "GateTools.hh"
#include "GateEcatAccelSystem.hh"
#include "GateToSinoAccel.hh"
#include "GateSinogram.hh"
#include "GateVVolume.hh"
#include "GateVolumePlacement.hh"


GateSinoAccelToEcat7::GateSinoAccelToEcat7(const G4String& name, GateOutputMgr* outputMgr,GateEcatAccelSystem* itsSystem,DigiMode digiMode)
  : GateVOutputModule(name,outputMgr,digiMode)
  , m_system(itsSystem)
  , m_fileName(" ") // All default output file from all output modules are set to " ".
                    // They are then checked in GateApplicationMgr::StartDAQ, using
                    // the VOutputModule pure virtual method GiveNameOfFile()
{
   m_isEnabled = false; // Keep this flag false: all output are disabled by default
   m_asciiMessenger = new GateSinoAccelToEcat7Messenger(this);
   nVerboseLevel = 0;
   m_ptr = NULL;
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
   m_ecatAccelCameraNumber = 0;
   if (nVerboseLevel > 0) G4cout << " >> GateSinoAccelToEcat7 created" << G4endl;
}

GateSinoAccelToEcat7::~GateSinoAccelToEcat7()
{
  delete m_asciiMessenger;
  if (m_ptr != NULL) {
    if (nVerboseLevel > 1) G4cout << " >> ECAT7 file " << m_ptr->fname << " will be closed" << G4endl;
    matrix_close(m_ptr);
  }
  if (mh != NULL) free(mh);
  if (sh != NULL) free(sh);
  if (nVerboseLevel > 0) G4cout << " >> GateSinoAccelToEcat7 deleted" << G4endl;
}

const G4String& GateSinoAccelToEcat7::GiveNameOfFile()
{
  return m_fileName;
}

void GateSinoAccelToEcat7::RecordBeginOfAcquisition()
{
  if (!(m_system->GetSinogramMaker()->IsEnabled())) return;
  if (nVerboseLevel > 0) G4cout << " >> entering [GateSinoAccelToEcat7::RecordBeginOfAcquisition]" << G4endl;

  // Create main header
  mh = (Main_header *) calloc(1,sizeof(Main_header));
  if (mh == NULL) {
     G4Exception( "GateSinoAccelToEcat7::RecordBeginOfAcquisition", "Could not allocate memory for main header",
			FatalException, "Could not allocate memory for main header" );
  }
  if (nVerboseLevel > 2) G4cout << "    Memory allocated for main header " << G4endl;
  // Fill main header
  FillMainHeader();
  if (nVerboseLevel > 2) G4cout << "    Main header filled" << G4endl;
  // Create subheader
  sh = (Scan3D_subheader *) calloc(1,sizeof(Scan3D_subheader));
  if (sh == NULL) {
     G4Exception( "GateSinoAccelToEcat7::RecordBeginOfAcquisition", "Could not allocate memory for subheader",
				FatalException, "Could not allocate memory for subheader");
  }
  if (nVerboseLevel > 2) G4cout << "    Memory allocated for sub header " << G4endl;
  sh->frame_duration = 0;
  sh->frame_start_time = 0;

  // Create ECAT7 file and write the main header
  m_ptr = matrix_create((m_fileName+".S").c_str(),MAT_CREATE,mh);
  if (m_ptr == NULL) {
			G4String msg = "Could not create ECAT7 file '"+m_fileName+".S' !";
     G4Exception( "GateSinoAccelToEcat7::RecordBeginOfAcquisition", "RecordBeginOfAcquisition",
				FatalException, msg );
  }
  if (nVerboseLevel > 1) G4cout << "    ECAT7 file " << m_fileName << ".S created" << G4endl;


  if (nVerboseLevel > 0) G4cout << " >> leaving [GateSinoAccelToEcat7::RecordBeginOfAcquisition]" << G4endl;
}

void GateSinoAccelToEcat7::RecordEndOfAcquisition()
{
  if (!(m_system->GetSinogramMaker()->IsEnabled())) return;
  if (nVerboseLevel > 0) G4cout << " >> entering [GateSinoAccelToEcat7::RecordEndOfAcquisition]" << G4endl;

  // Delete main header
  free(mh);
  mh = NULL;
  // Delete subheader
  free(sh);
  sh = NULL;
  // Close the ECAT file
  if (m_ptr != NULL) {
    if (nVerboseLevel > 1) G4cout << "    ECAT7 file " << m_ptr->fname << " will be closed" << G4endl;
    matrix_close(m_ptr);
    m_ptr = NULL;
  }
  if (nVerboseLevel > 0) G4cout << " >> leaving [GateSinoAccelToEcat7::RecordEndOfAcquisition]" << G4endl;
}

void GateSinoAccelToEcat7::RecordBeginOfRun(const G4Run * )
{
  GateToSinoAccel* setMaker = m_system->GetSinogramMaker();
  G4int  seg,segm,segment_occurance,nplane;
  if (!(m_system->GetSinogramMaker()->IsEnabled())) return;
  if (nVerboseLevel > 0) G4cout << " >> entering [GateSinoAccelToEcat7::RecordBeginOfRun]" << G4endl;
  if (nVerboseLevel > 1) {
    G4cout << "    Frame ID:     " << setMaker->GetSinogram()->GetCurrentFrameID() << G4endl;
    G4cout << "    Gate ID:      " << setMaker->GetSinogram()->GetCurrentGateID() << G4endl;
    G4cout << "    Bed position: " << setMaker->GetSinogram()->GetCurrentBedID() << G4endl;
  }
  nplane = 2*setMaker->GetRingNb() - 1;
  if (m_mashing <= 0) m_mashing = 1;
  if (m_maxRingDiff < 0) m_maxRingDiff = setMaker->GetRingNb()-1;
  if (m_span < 3) m_span = 3;
  if (m_maxRingDiff >= setMaker->GetRingNb()) {
    G4cout << " !!! [GateSinoAccelToEcat7]: maximum ring difference (" << m_maxRingDiff
           << ") is too big (should be <" << setMaker->GetRingNb() << ")" << G4endl;
    G4Exception( "GateSinoAccelToEcat7::RecordBeginOfRun", "RecordBeginOfRun", FatalException, "Could not fill subheader");
  }
  if ((m_span<1)||((float)((m_span-1)/2)) != (((float)m_span-1.0)/2.0)) {
    G4cout << " !!! [GateSinoAccelToEcat7]: span factor (" << m_maxRingDiff
           << ") should be odd" << G4endl;
    G4Exception( "GateSinoAccelToEcat7::RecordBeginOfRun", "RecordBeginOfRun", FatalException, "Could not fill subheader");
  }
  if ((float)((2*m_maxRingDiff+1-m_span)/(2*m_span)) != (2.0*(float)m_maxRingDiff+1.0-(float)m_span)/(2.0*(float)m_span)) {
    G4int bin,maxRingDiff;
    G4cout << " !!! [GateSinoAccelToEcat7]: maximum ring difference (" << m_maxRingDiff
           << ") is not coherent with span factor (" << m_span << ")" << G4endl;
    G4cout << "                                       possible maximum ring differences are: " << G4endl;
    bin = 0;
    maxRingDiff = (m_span-1)/2;
    while (maxRingDiff < setMaker->GetRingNb()) {
      G4cout << "                                           " << maxRingDiff << G4endl;
      bin++;
      maxRingDiff = ((2*bin+1)*m_span-1)/2;
    }
    G4Exception( "GateSinoAccelToEcat7::FillData", "FillData", FatalException, "Could not fill subheader");
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
  if (nVerboseLevel > 0) G4cout << " >> leaving [GateSinoAccelToEcat7::RecordBeginOfRun]" << G4endl;
}

void GateSinoAccelToEcat7::RecordEndOfRun(const G4Run * )
{
  if (!(m_system->GetSinogramMaker()->IsEnabled())) return;
  if (nVerboseLevel > 0) G4cout << " >> entering [GateSinoAccelToEcat7::RecordEndOfRun]" << G4endl;
  FillData();
  if (nVerboseLevel > 0) G4cout << " >> leaving [GateSinoAccelToEcat7::RecordEndOfRun]" << G4endl;
}

/* Overload of the base-class' virtual method to print-out a description of the module

	indent: the print-out indentation (cosmetic parameter)
*/
void GateSinoAccelToEcat7::Describe(size_t indent)
{
  GateVOutputModule::Describe(indent);
  G4cout << GateTools::Indent(indent) << " >> Job:                   write a set of 2D sinograms into an Ecat7 output file" << G4endl;
  G4cout << GateTools::Indent(indent) << " >> Is enabled ?           " << ( IsEnabled() ? "Yes" : "No") << G4endl;
  G4cout << GateTools::Indent(indent) << " >> File name:             " << m_fileName << ".S" << G4endl;
  G4cout << GateTools::Indent(indent) << " >> Attached to system:    " << m_system->GetObjectName() << G4endl;
}

// Fill the main header
void GateSinoAccelToEcat7::FillMainHeader()
{
  time_t       timer;
  struct tm   *date;

  GateToSinoAccel* setMaker = m_system->GetSinogramMaker();


  // Retrieve the parameters of the blocks and crystals
  GateSystemComponent* blockComponent    = m_system->GetMainComponent();
  GateArrayComponent*   crystalComponent = m_system->GetDetectorComponent();
  G4ThreeVector         crystalPitchVector = crystalComponent->GetRepeatVector();
  G4int                 RingNb = blockComponent->GetSphereAxialRepeatNumber() * crystalComponent->GetRepeatNumber(2);
  if (setMaker->GetRingNb() != RingNb) {
     G4Exception( "GateSinoAccelToEcat7::FillMainHeader", "FillMainHeader", FatalException, "Uncoherent crystal rings number");
  }
  if (nVerboseLevel > 1) {
    G4cout << "    Number of crystal rings: " << RingNb << G4endl;
    G4cout << "    crystal axial pitch: " << crystalPitchVector.z()/mm << " mm" << G4endl;
    G4cout << "    crystal transverse pitch: " << crystalPitchVector.x()/mm << " mm" << G4endl;
  }

  strncpy(mh->original_file_name,(m_fileName+".S").c_str(),32);
  mh->original_file_name[31]='\0';
  mh->sw_version = V7;
  mh->system_type = m_ecatAccelCameraNumber;
  mh->file_type = Short3dSinogram;
  strncpy(mh->serial_number,"OpenGATE",10);
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
  mh->coin_samp_mode = 1; /* prompts and delayed */
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
  mh->bin_size = crystalPitchVector.x()/cm/2.0;
  mh->lwr_true_thres = 0;
  mh->upr_true_thres = 0;
  mh->acquisition_mode = 0;
  mh->branching_fraction = 1.0;
  mh->septa_state = SeptaRetracted;
}

// Fill the main header
void GateSinoAccelToEcat7::FillData()
{

  GateToSinoAccel* setMaker = m_system->GetSinogramMaker();
  G4int  bin,seg,segment_occurance,data_size,nz,frame,plane,gate,data,bed,matnum,
         nblks,tot_data_size,blkno,file_pos,offset,csize,ringdiff,ring_1_min,ring_1_max,
	 view,ring_1,ring_2,elem,z,sinoID,bin_sdata,bin_m_data;
  short  *sdata;
  char   *cdata;
  struct MatDir matdir, dir_entry;
  GateSinogram::SinogramDataType *m_data, *m_randoms;

  // Fill subheader
  seg = 0;
  sh->data_type = SunShort;
  sh->num_dimensions = 4;
  sh->num_r_elements = setMaker->GetRadialElemNb();
  sh->num_angles = setMaker->GetCrystalNb()/2/m_mashing;
  sh->num_z_elements[seg] = m_zMaxSeg[seg] - m_zMinSeg[seg] + 1;
  for (seg=1;seg<=m_segmentNb;seg++) sh->num_z_elements[seg] = 2*(m_zMaxSeg[2*seg-1] - m_zMinSeg[2*seg-1] + 1);
  sh->ring_difference = m_maxRingDiff;
  sh->axial_compression = m_span;
  sh->scale_factor = 1.0;
  sh->scan_min = -1;
  sh->scan_max = -1;
  sh->prompts = 0;
  sh->delayed = 0;
  sh->x_resolution = mh->bin_size;
  sh->z_resolution = mh->plane_separation;
  sh->corrections_applied = 0;
  sh->frame_start_time += sh->frame_duration; // increment from last frame duration
  sh->frame_duration = (int) (setMaker->GetFrameDuration() / second * 1000.0);
  if (nVerboseLevel > 1) {
    G4cout << "    Frame ID:                             " << setMaker->GetSinogram()->GetCurrentFrameID() << G4endl;
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
  data_size = (m_zMaxSeg[0] - m_zMinSeg[0] + 1) * sh->num_r_elements * sh->num_angles;
  sdata = (short*) calloc(sizeof(short),data_size);
  cdata = (char*) calloc(sizeof(short),data_size);
  tot_data_size = 0;
  seg = 0;
  while (sh->num_z_elements[seg]>0) {
    tot_data_size += sh->num_r_elements*sh->num_angles*sh->num_z_elements[seg];
    seg++;
  }
  nblks = (tot_data_size*sizeof(short)+MatBLKSIZE-1)/MatBLKSIZE;
  frame = setMaker->GetSinogram()->GetCurrentFrameID();
  plane = 1;
  gate = 1;
  data = 0;
  bed = 0;
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
  offset = 0;
  // loop on the segments
  for (segment_occurance=0;segment_occurance<(2*m_segmentNb+1);segment_occurance++) {
    nz = m_zMaxSeg[segment_occurance] - m_zMinSeg[segment_occurance] + 1;
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
	  sinoID = setMaker->GetSinogram()->GetSinoID(ring_1,ring_2);
	  if (sinoID < 0 || sinoID >= (G4int) setMaker->GetSinogram()->GetSinogramNb()) {
	    G4Exception( "GateToSinoAccel::RecordEndOfRun", "RecordEndOfRun", FatalException, "Wrong 2D sinogram ID");
	  }
	  if (nVerboseLevel>2 && view==0) {
            G4cout << " >> ring difference " << ringdiff << ", slice " << z << G4endl;
            G4cout << "    rings " << ring_1 << "," << ring_2  << " give sino ID " << sinoID << G4endl;
	  }
          //  m_system->GetProjectionSetMaker()->GetProjectionSet()->StreamOut( m_dataFile , headID );
          m_data = setMaker->GetSinogram()->GetSinogram(sinoID);
	  bin_m_data = view * setMaker->GetRadialElemNb(); // sinogram ordering
	  bin_sdata = z * sh->num_r_elements + view / m_mashing * nz * sh->num_r_elements; // view ordering
	  for (elem=0; elem<sh->num_r_elements; elem++) sdata[bin_sdata+elem] += (short int) m_data[bin_m_data+elem];
	}
      }
      for (ring_1 = ring_1_min; ring_1 <= ring_1_max; ++ring_1) {
        ring_2 = ring_1 + ringdiff;
	//sinoID = ring_1 + ring_2 * setMaker->GetRingNb();
	sinoID = setMaker->GetSinogram()->GetSinoID(ring_1,ring_2);
	if (sinoID < 0 || sinoID >= (G4int) setMaker->GetSinogram()->GetSinogramNb()) {
	  G4Exception( "GateToSinoAccel::RecordEndOfRun", "RecordEndOfRun", FatalException, "Wrong 2D sinogram ID");
	}
	m_randoms = setMaker->GetSinogram()->GetRandoms();
	sh->delayed += (short int) m_randoms[sinoID];
      }
    }
    csize = 0;
    // convert short --> SunShort
    if (segment_occurance == 0) {
      sh->scan_min = sh->scan_max = sdata[0];
    }
    for (bin=0;bin<data_size;bin++) {
      bufWrite_s(sdata[bin],cdata,&csize);
      if (sdata[bin] < sh->scan_min) sh->scan_min = sdata[bin];
      else if (sdata[bin] > sh->scan_max) sh->scan_max = sdata[bin];
      sh->prompts += sdata[bin];
    }
    // write data segment
    file_pos = (matdir.strtblk+1)*MatBLKSIZE+offset;
    if (fseek(m_ptr->fptr,file_pos,0) == EOF) {
      G4cout << " !!! [GateSinoAccelToEcat7::FillData]: can not seek position " << file_pos
             << " for " << m_ptr->fname << G4endl;
      G4Exception( "GateSinoAccelToEcat7::FillData", "FillData", FatalException,  "Could not fill data");
    } else if (fwrite(cdata,sizeof(short),data_size,m_ptr->fptr) != (size_t) data_size) {
      G4cout << " !!! [GateSinoAccelToEcat7::FillData]: can not write segment " << segment_occurance
             << " in " << m_ptr->fname << G4endl;
      G4Exception( "GateSinoAccelToEcat7::FillData", "FillData", FatalException, "Could not fill data");
    }
    offset += data_size * sizeof(short);
  }
  // update scan_min and scan_max
  sh->net_trues = sh->prompts - sh->delayed;
  if (nVerboseLevel > 1) {
    G4cout << "    Number of prompts:                    " << sh->prompts << G4endl;
    G4cout << "    Number of randoms:                    " << sh->delayed << G4endl;
    G4cout << "    Number of net trues:                  " << sh->net_trues << G4endl;
    G4cout << "    Maximum bin content:                  " << sh->scan_max << G4endl;
  }
  mat_write_Scan3D_subheader(m_ptr->fptr,m_ptr->mhptr,matdir.strtblk,sh);
  if (mh_update(m_ptr)) {
       G4cout << " !!! [GateSinoAccelToEcat7::FillData]: can not update main header in  "
              << m_ptr->fname << G4endl;
       G4Exception( "GateSinoAccelToEcat7::FillData", "FillData", FatalException, "Could not fill data");
  }
  free(cdata);
  free(sdata);
}
#endif
