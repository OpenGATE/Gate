/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateToSinoAccel.hh"

#include "globals.hh"
#include "G4UnitsTable.hh"
#include "G4Run.hh"

#include "GateCoincidenceDigi.hh"
#include "GateOutputMgr.hh"
#include "GateSinogram.hh"
#include "GateToSinoAccelMessenger.hh"
#include "GateTools.hh"
#include "GateVSystem.hh"
#include "GateApplicationMgr.hh"

// #include "GatePlacementMove.hh"

// Public constructor (creates an empty, uninitialised, project set)
GateToSinoAccel::GateToSinoAccel(const G4String& name, GateOutputMgr* outputMgr,GateVSystem* itsSystem,DigiMode digiMode)
  : GateVOutputModule(name,outputMgr,digiMode)
  , m_sinogram(0)
  // , m_sinoRandoms(0)
  , m_crystalNb(0)
  , m_ringNb(0)
  , m_studyDuration(0.)
  , m_frameDuration(0.)
  , m_frameNb(0)
  , m_system(itsSystem)
  , m_flagTruesOnly(false)
  , m_radialElemNb(0)
  , m_flagIsRawOutputEnabled(false)
  , m_fileName(" ") // All default output file from all output modules are set to " ".
                    // They are then checked in GateApplicationMgr::StartDAQ, using
                    // the VOutputModule pure virtual method GiveNameOfFile()
  , m_tangCrystalResolution(0.)
  , m_axialCrystalResolution(0.)
  , m_inputDataChannel("Coincidences")
{
  m_isEnabled = false; // Keep this flag false: all output are disabled by default
  m_sinogram = new GateSinogram();
  // m_sinoRandoms = new GateSinogram();
  m_messenger = new GateToSinoAccelMessenger(this);
  SetVerboseLevel(0);
}

GateToSinoAccel::~GateToSinoAccel()
{
  delete m_sinogram;
  // delete m_sinoRandoms;
  delete m_messenger;

}

const G4String& GateToSinoAccel::GiveNameOfFile()
{
  return m_fileName;
}

// Initialisation of the projection set
void GateToSinoAccel::RecordBeginOfAcquisition()
{
  if (nVerboseLevel>0) G4cout << " >> entering [GateToSinoAccel::RecordBeginOfAcquisition]\n";

  // Retrieve the parameters of the experiment
  G4double timeStart = GateApplicationMgr::GetInstance()->GetTimeStart();
  G4double timeStop  = GateApplicationMgr::GetInstance()->GetTimeStop();
  G4double timeStep  = GateApplicationMgr::GetInstance()->GetTimeSlice();
  G4double duration  = timeStop-timeStart;
  m_studyDuration = duration; // Total acquisition duration
  m_frameDuration = timeStep; // Divide acquisition into multiple frames

  G4double fstepNumber = duration / timeStep;
  if ( fabs(fstepNumber-rint(fstepNumber)) >= 1.e-5 ) {
    G4cerr  <<  Gateendl << " !!! [GateToSinoAccel::RecordBeginOfAcquisition]:\n"
	    <<   "Sorry, but the study duration (" << G4BestUnit(duration,"Time") << ") "
	    <<   " does not seem to be a multiple of the time-slice (" << G4BestUnit(timeStep,"Time") << ").\n";
    G4Exception( "GateToSinoAccel::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, "You must change these parameters then restart the simulation\n");
  }
  m_frameNb = static_cast<size_t>(rint(fstepNumber));
  if (nVerboseLevel>1) G4cout << "    Number of frames: " << m_frameNb << Gateendl;

  // Retrieve the number of crystal rings and crystals per crystal ring
  GateSystemComponent* blockComponent   = m_system->GetMainComponent();
  GateArrayComponent*  crystalComponent = m_system->GetDetectorComponent();
  m_ringNb    = blockComponent->GetSphereAxialRepeatNumber() * crystalComponent->GetRepeatNumber(2);
  m_crystalNb = blockComponent->GetSphereAzimuthalRepeatNumber() * crystalComponent->GetRepeatNumber(0);
  if (nVerboseLevel > 1) {
    G4cout << "    Number of crystals per crystal rings: " << m_crystalNb << Gateendl
	   << "    Number of crystal rings:              " << m_ringNb << Gateendl;
  }

  // 10.04.2003; Claude & Arion
  // Default value for the number of radial sinogram bins
  m_radialElemNb = GetRadialElemNb();
  if (m_radialElemNb <= 0) {
    m_radialElemNb = m_crystalNb/2;
  } else if (m_radialElemNb > m_crystalNb) {
    G4cerr  <<  Gateendl << " !!! [GateToSinoAccel::RecordBeginOfAcquisition]:\n"
	    <<   "Sorry, but the number of radial sinogram bins (" << m_radialElemNb << ") should be smaller or equal to " << m_crystalNb << Gateendl
	    <<   "The default is " << m_crystalNb/2 << Gateendl;
    G4Exception( "GateToSinoAccel::RecordBeginOfAcquisition" , "RecordBeginOfAcquisition", FatalException,"You must change this parameter then restart the simulation");
  }
  if (nVerboseLevel > 1) {
    G4cout << "    Number of radial sinogram bins:    " << m_radialElemNb << Gateendl;
    G4cout << "    Number of azimuthal sinogram bins: " << m_crystalNb/2 << Gateendl;

  }

  // Crystal location blurring
  m_tangCrystalResolution = GetTangCrystalResolution();
  if (m_tangCrystalResolution  <= 0.) m_tangCrystalResolution = 0.;
  m_axialCrystalResolution = GetAxialCrystalResolution();
  if (m_axialCrystalResolution <= 0.) m_axialCrystalResolution = 0.;
  if (nVerboseLevel > 1) {
    G4cout << "    Crystal location blurring in tangential direction: " << m_tangCrystalResolution/mm << " mm\n";
    G4cout << "    Crystal location blurring in axial direction: " << m_axialCrystalResolution/mm << " mm\n";
  }

  // Prepare the sinogram
  m_sinogram->Reset(m_ringNb,m_crystalNb,m_radialElemNb);
  // m_sinoRandoms->Reset(m_ringNb,m_crystalNb);

  if (nVerboseLevel>0) {
    if (m_flagTruesOnly) {
      G4cout << "    Only true coincidences are recorded\n";
    } else {
      G4cout << "    True and random coincidences are recorded\n";
    }
  }


  if (nVerboseLevel>0) G4cout << " >> leaving [GateToSinoAccel::RecordBeginOfAcquisition]\n";
}

// We leave the 2D sinograms as it is (so that it can be stored afterwards)
// but we still have to destroy the array of sinogram IDs
void GateToSinoAccel::RecordEndOfAcquisition()
{
}

// Reset the projection data
void GateToSinoAccel::RecordBeginOfRun(const G4Run * r)
{
  if (nVerboseLevel>0) G4cout << " >> entering [GateToSinoAccel::RecordBeginOfRun]\n";
  G4cout << "    Frame ID = " << r->GetRunID()+1 << Gateendl;
  // One frame per RUN
  m_sinogram->ClearData(r->GetRunID()+1,1,0,0);
  // m_sinoRandoms->ClearData(r->GetRunID()+1,1,1,0);
  if (nVerboseLevel>0) G4cout << " >> leaving [GateToSinoAccel::RecordBeginOfRun]\n";

}

void GateToSinoAccel::RecordEndOfRun(const G4Run * r)
{
  G4String         frameFileName;
  std::ofstream  m_dataFile,m_infoFile,m_dimFile;
  char             ctemp[512];
  G4int            aringdiff,nseg,seg,ringdiff,ring_1_min,ring_1_max,ring_1,ring_2,sinoID;
  size_t           seekID;

  if (nVerboseLevel>0) G4cout << " >> entering [GateToSinoAccel::RecordEndOfRun]\n";

  // Write the projection sets
  if (m_flagIsRawOutputEnabled) {
    sprintf(ctemp,"%s_%0d",m_fileName.c_str(),r->GetRunID()+1);
    frameFileName = ctemp;
    G4cout << "    sinograms written to the raw file " << frameFileName << ".ima\n";
    m_dataFile.open((frameFileName+".ima").c_str(),std::ios::out | std::ios::trunc | std::ios::binary);
    seekID = 0;
    for (aringdiff=0 ; aringdiff<(G4int)m_ringNb; aringdiff++) {
      if (aringdiff == 0) nseg = 1;
      else nseg = 2;
      for (seg=0 ; seg<nseg; seg++) {
        if (seg == 0) { /* Positive ring difference */
	  ringdiff = aringdiff;
	  ring_1_min = 0;
	  ring_1_max = m_ringNb - ringdiff - 1;
	} else { /* Negative ring difference */
	  ringdiff = -aringdiff;
	  ring_1_min = -ringdiff;
	  ring_1_max = m_ringNb - 1;
	}
	for (ring_1 = ring_1_min; ring_1 <= ring_1_max ; ++ring_1) {
	  ring_2 = ring_1 + ringdiff;
	  sinoID = m_sinogram->GetSinoID(ring_1,ring_2);
	  if (sinoID < 0 || (unsigned)sinoID >= m_sinogram->GetSinogramNb()) {
	    G4Exception( "GateToSinoAccel::RecordEndOfRun", "RecordEndOfRun", FatalException,"Wrong 2D sinogram ID");
          }
	  if (nVerboseLevel>2) {
            G4cout << " >> rings " << ring_1 << "," << ring_2  << " give sino ID " << sinoID << Gateendl;
	  }
	  m_sinogram->StreamOut( m_dataFile , sinoID, seekID );
	  seekID++;
        }
      }
    }
    m_dataFile.close();
    m_infoFile.open((frameFileName+".info").c_str(),std::ios::out | std::ios::trunc | std::ios::binary);
    m_infoFile << m_sinogram->GetSinogramNb() << " 2D sinograms\n";
    m_infoFile << " [RadialPosition;AzimuthalAngle;AxialPosition;RingDifference]\n";
    m_infoFile << " RingDifference varies as 0,+1,-1,+2,-2, ...,+" << m_ringNb-1 << ",-" << m_ringNb-1 << Gateendl;
    m_infoFile << " AxialPosition varies as |RingDifference|,...," << 2*m_ringNb-2 << "-|RingDifference| per increment of 2\n";
    m_infoFile << " AzimuthalAngle varies as 0,...," << m_crystalNb/2-1 << " per increment of 1\n";
    m_infoFile << " RadialPosition varies as 0,...," << m_radialElemNb-1 << " per increment of 1\n";
    m_infoFile << " Date type : unsigned short integer (U" << 8*sizeof(unsigned short) << ")\n";
    m_infoFile.close();
    m_dimFile.open((frameFileName+".dim").c_str(),std::ios::out | std::ios::trunc | std::ios::binary);
    m_dimFile << " " << m_radialElemNb << " " << m_crystalNb/2 << " " << m_ringNb*m_ringNb << Gateendl;
    m_dimFile << "-type U" << 8*sizeof(unsigned short) << Gateendl << "-dx 1.0\n" << "-dy 1.0\n" << "-dz 1.0";
    m_dimFile.close();
  }

  if (nVerboseLevel>0) G4cout << " >> leaving [GateToSinoAccel::RecordEndOfRun]\n";
}


// Update the target sinogram with regards to the digis acquired for this event
void GateToSinoAccel::RecordEndOfEvent(const G4Event* )
{
  double x1,x2,y1,y2;
  // double orig1,orig2,azipos1,azipos2,alpha,view,elem;

  const GateCoincidenceDigiCollection * CDC = GetOutputMgr()->GetCoincidenceDigiCollection(m_inputDataChannel);

  if (!CDC) {
    return;
  }

  if (nVerboseLevel>3) G4cout << " >> entering [GateToSinoAccel::RecordEndOfEvent] with a digi collection\n";

  G4int n_digi =  CDC->entries();
  // Retrieve the block and the crystal component
  GateSystemComponent* blockComponent   = m_system->GetMainComponent();
  GateArrayComponent*  crystalComponent = m_system->GetDetectorComponent();
  G4ThreeVector        crystalPitchVector = crystalComponent->GetRepeatVector();

  if (nVerboseLevel>3) G4cout << " >> Total Digits: " << n_digi << Gateendl;
  for (G4int iDigi=0;iDigi<n_digi;iDigi++) {
    // crystal block ID
    G4int block1ID = m_system->GetMainComponentID( (*CDC)[iDigi]->GetDigi(0) );
    G4int block2ID = m_system->GetMainComponentID( (*CDC)[iDigi]->GetDigi(1) );
    // crystal ID within a crystal block
    G4int crystal1ID = m_system->GetDetectorComponentID( (*CDC)[iDigi]->GetDigi(0) );
    G4int crystal2ID = m_system->GetDetectorComponentID( (*CDC)[iDigi]->GetDigi(1) );
    // crystal ring ID
    G4int ring1 = (int)(block1ID/blockComponent->GetSphereAzimuthalRepeatNumber()*crystalComponent->GetRepeatNumber(2))+
		  (int)(crystal1ID/crystalComponent->GetRepeatNumber(0));
    G4int ring2 = (int) (block2ID/blockComponent->GetSphereAzimuthalRepeatNumber()*crystalComponent->GetRepeatNumber(2))+
		  (int)(crystal2ID/crystalComponent->GetRepeatNumber(0));
    // crystal ID within a crystal ring
    G4int crystal1 = (block1ID % blockComponent->GetSphereAzimuthalRepeatNumber())*crystalComponent->GetRepeatNumber(0)+
		     (crystal1ID % crystalComponent->GetRepeatNumber(0));
    G4int crystal2 = (block2ID % blockComponent->GetSphereAzimuthalRepeatNumber())*crystalComponent->GetRepeatNumber(0)+
		     (crystal2ID % crystalComponent->GetRepeatNumber(0));
    G4int eventID1 = ((*CDC)[iDigi]->GetDigi(0))->GetEventID();
    G4int eventID2 = ((*CDC)[iDigi]->GetDigi(1))->GetEventID();

    if (m_flagTruesOnly && (eventID1 != eventID2)) {
      if (nVerboseLevel>3) {
        G4cout << "    random coincidence not recorded \n";
      }
      return;
    }

    // DEBUG
    //G4float xpos1 = ((*CDC)[iDigi]->GetPulse(0)).GetGlobalPos().x()/mm;
    //G4float xpos2 = ((*CDC)[iDigi]->GetPulse(1)).GetGlobalPos().x()/mm;
    //G4float ypos1 = ((*CDC)[iDigi]->GetPulse(0)).GetGlobalPos().y()/mm;
    //G4float ypos2 = ((*CDC)[iDigi]->GetPulse(1)).GetGlobalPos().y()/mm;

    // offset crystal origin by half-block
    crystal1 -= crystalComponent->GetRepeatNumber(0)/2;
    crystal2 -= crystalComponent->GetRepeatNumber(0)/2;
    if (crystal1 < 0) crystal1 += m_crystalNb;
    if (crystal2 < 0) crystal2 += m_crystalNb;

    // DEBUG
    // calculate crystal ID according to pulse position
    //azipos1 = atan2(ypos1,xpos1)*m_crystalNb/(2.*M_PI);
    //azipos2 = atan2(ypos2,xpos2)*m_crystalNb/(2.*M_PI);
    //if (azipos1 < 0.) azipos1 += m_crystalNb;
    //if (azipos2 < 0.) azipos2 += m_crystalNb;
    // deduct offset
    //orig1 = azipos1 - crystal1;
    //orig2 = azipos2 - crystal2;
    //if (orig1 < 0.) orig1 += m_crystalNb;
    //if (orig2 < 0.) orig2 += m_crystalNb;

    if (nVerboseLevel>3) {
      G4cout << " >>  Digi # " << iDigi << Gateendl;
      G4cout << " >>     Block IDs are " << block1ID << " ; " << block2ID << Gateendl;
      G4cout << " >>     Block crystal  IDs are " << crystal1ID << " ; " << crystal2ID << Gateendl;
      G4cout << " >>     Crystal ring IDs are " << ring1 << " ; " << ring2 << Gateendl;
      G4cout << " >>     Ring crystal IDs are " << crystal1 << " ; " << crystal2 << Gateendl;
      G4cout << " >>     Event IDs are " << eventID1 << " ; " << eventID2 << Gateendl;
      //G4cout << " >>     DEBUG: Gamma azimuthal angle are " << azipos1 << " ; " << azipos2 << " crystal\n";
      //G4cout << " >>     DEBUG: Crystal origines are " << orig1 << " ; " << orig2 << " crystal\n";
    }

    // change crystal origine to be compatible with ECAT systems
    crystal1 += m_crystalNb/4;
    crystal2 += m_crystalNb/4;
    if (crystal1 >= (G4int) m_crystalNb) crystal1 -= m_crystalNb;
    if (crystal2 >= (G4int) m_crystalNb) crystal2 -= m_crystalNb;
    if (ring1 < 0 || ring1 >= (G4int) m_ringNb || ring2 < 0 || ring2 >= (G4int) m_ringNb) {
      G4cout << " !!! out of range crystal ring number (" << ring1 << " ; " << ring2 << ")\n";
      return;
    }
    if (crystal1 < 0 || crystal1 >= (G4int) m_crystalNb || crystal2 < 0 || crystal2 >= (G4int) m_crystalNb) {
      G4cout << " !!! out of range ring crystal number (" << crystal1 << " ; " << crystal2 << ")\n";
      return;
    }

    //  Add spatial blurring to crystal IDs
    //G4cout << " DEBUG: gamma one IDs before blurring = " << crystal1 << " ; " << ring1 << Gateendl;
    m_sinogram->CrystalBlurring(&ring1, &crystal1,
                                m_axialCrystalResolution/crystalPitchVector.z(),
				m_tangCrystalResolution/crystalPitchVector.x());
    //G4cout << " DEBUG: gamma one IDs after  blurring = " << crystal1 << " ; " << ring1 << Gateendl;
    //G4cout << " DEBUG: gamma two IDs before blurring = " << crystal2 << " ; " << ring2 << Gateendl;
    m_sinogram->CrystalBlurring(&ring2, &crystal2,
                                m_axialCrystalResolution/crystalPitchVector.z(),
				m_tangCrystalResolution/crystalPitchVector.x());
    //G4cout << " DEBUG: gamma two IDs after  blurring = " << crystal2 << " ; " << ring2 << Gateendl;

    //  ordering between detector 1 and detector 2 : x1 >= x2 (convention)
    //  important for polar angle sign (ring2 - ring1)
    y1 = sin((0.5+crystal1)*(twopi/(double)m_crystalNb));
    y2 = sin((0.5+crystal2)*(twopi/(double)m_crystalNb));
    x1 = cos((0.5+crystal1)*(twopi/(double)m_crystalNb));
    x2 = cos((0.5+crystal2)*(twopi/(double)m_crystalNb));
    if (y1 > y2) {
      if ((m_sinogram->Fill( ring2, ring1, crystal1, crystal2, +1) == 0) && (eventID1 != eventID2))
        m_sinogram->FillRandoms( ring2, ring1);
      //alpha = atan2(x1-x2,y2-y1);
    } else if (y1 < y2) {
      if ((m_sinogram->Fill( ring1, ring2, crystal2, crystal1, +1) == 0) && (eventID1 != eventID2))
        m_sinogram->FillRandoms( ring1, ring2);
      //alpha = atan2(x2-x1,y1-y2);
    } else {
      if (x1 < x2) {
	if ((m_sinogram->Fill( ring2, ring1, crystal1, crystal2, +1) == 0) && (eventID1 != eventID2))
	  m_sinogram->FillRandoms( ring2, ring1);
	//alpha = atan2(x1-x2,y2-y1);
      } else if (x1 > x2) {
	if ((m_sinogram->Fill( ring1, ring2, crystal2, crystal1, +1) == 0) && (eventID1 != eventID2))
	  m_sinogram->FillRandoms( ring1, ring2);
	//alpha = atan2(x2-x1,y1-y2);
      } else {
	G4cout << " !!! uncoherent crystal numbering (" << crystal1 << " ; " << crystal2 << ") !!!\n";
	G4cout << " !!! Event skiped\n";
	return;
      }
    }

    // DEBUG
    //view = alpha * m_crystalNb / (2.0 * M_PI);
    //elem = (cos(alpha)*x1 + sin(alpha)*y1) / (crystalComponent->GetBoxLength(1)/mm/2.) + m_crystalNb/4;
    //if (nVerboseLevel>3) {
    //  G4cout << " >>     DEBUG: elem according to crystal coordinates, no arc effect: " << elem << Gateendl;
    //  G4cout << " >>     DEBUG: view according to crystal coordinates: " << view << Gateendl;
    //}
  }
  if (nVerboseLevel>3) G4cout << " >> leaving [GateToSinogram::RecordEndOfEvent]\n";
}




/* Overload of the base-class' virtual method to print-out a description of the module

	indent: the print-out indentation (cosmetic parameter)
*/
void GateToSinoAccel::Describe(size_t indent)
{
  GateVOutputModule::Describe(indent);
  G4cout << GateTools::Indent(indent) << " >> Job:                                build a set of 2D sinograms from a PET simulation\n";
  G4cout << GateTools::Indent(indent) << " >> Is enabled?                         " << ( IsEnabled() ? "Yes" : "No") << Gateendl;
  G4cout << GateTools::Indent(indent) << " >> Number of crystals per crystal ring " << m_crystalNb << Gateendl;
  G4cout << GateTools::Indent(indent) << " >> Number of crystal rings             " << m_ringNb << Gateendl;
  G4cout << GateTools::Indent(indent) << " >> Number of radial sinogram bins      " << m_radialElemNb << Gateendl;
  G4cout << GateTools::Indent(indent) << " >> Filled?                             " << ( m_sinogram->GetData() ? "Yes" : "No" ) << Gateendl;
  G4cout << GateTools::Indent(indent) << " >> Attached to system:                 " << m_system->GetObjectName() << Gateendl;
  G4cout << GateTools::Indent(indent) << " >> Input data                          " << m_inputDataChannel;
}
