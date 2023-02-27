/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*  Optical Photons: V. Cuplov -  2012
    - New function RecordOpticalData(event).
    - New tree for optical photon data is defined in GateToRoot class (previously was in GateFastAnalysis)
    - Revision v6.2   2012/07/09  by vesna.cuplov@gmail.com
    output ROOT file is dedicated to optical photons
    - Revision v6.2   2012/07/24  by vesna.cuplov@gmail.com
    Unique output file with Gate default trees (Hits,Singles,Coincidences...) + OpticalData Tree.
    - Revision v6.2 2012/08/06  Added optical photon momentum direction (x,y,z) in tree.
    - Revision 2012/09/17  /gate/output/root/setRootOpticalFlag functionality added.
    Set the flag for Optical ROOT output.
    - Revision 2012/11/14  - added new leaves: position (x,y,z) of fluorescent (OpticalWLS process) hits
    - Scintillation counter bug-fixed
    - 2023/02/22 PDG code for optical photon is changed from 0 to -22
*/

#include "GateToRoot.hh"
#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include <iomanip>
#include "globals.hh"
#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "G4Trajectory.hh"
#include "G4VProcess.hh"
#include "G4ios.hh"
#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"
#include "G4Gamma.hh"

#include "GateCrystalHit.hh"
#include "GatePhantomHit.hh"
#include "GateApplicationMgr.hh"
#include "GatePrimaryGeneratorAction.hh"
#include "GateHitConvertor.hh"
#include "GateSingleDigi.hh"
#include "GateCoincidenceDigi.hh"
#include "GateSourceMgr.hh"
#include "GateOutputMgr.hh"
#include "GateVVolume.hh"
#include "GateToRootMessenger.hh"
#include "GateVGeometryVoxelStore.hh"

#include "TROOT.h"
#include "TApplication.h"
#include "TGClient.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "G4DigiManager.hh"

// v. cuplov - optical photons
#include "G4OpticalPhoton.hh"
#include "GateTrajectoryNavigator.hh"
// v. cuplov - optical photons

ComptonRayleighData::ComptonRayleighData() { ; }

ComptonRayleighData::ComptonRayleighData(ComptonRayleighData &aCRData) {
    photon1_phantom_Rayleigh = aCRData.photon1_phantom_Rayleigh;
    photon2_phantom_Rayleigh = aCRData.photon2_phantom_Rayleigh;
    photon1_phantom_compton = aCRData.photon1_phantom_compton;
    photon2_phantom_compton = aCRData.photon2_phantom_compton;
    strcpy(theComptonVolumeName1, aCRData.theComptonVolumeName1);
    strcpy(theComptonVolumeName2, aCRData.theComptonVolumeName2);
    strcpy(theRayleighVolumeName1, aCRData.theRayleighVolumeName1);
    strcpy(theRayleighVolumeName2, aCRData.theRayleighVolumeName2);
}

ComptonRayleighData &ComptonRayleighData::operator=(const ComptonRayleighData &aCR) {
    photon1_phantom_Rayleigh = aCR.photon1_phantom_Rayleigh;
    photon2_phantom_Rayleigh = aCR.photon2_phantom_Rayleigh;
    photon1_phantom_compton = aCR.photon1_phantom_compton;
    photon2_phantom_compton = aCR.photon2_phantom_compton;
    strcpy(theComptonVolumeName1, aCR.theComptonVolumeName1);
    strcpy(theComptonVolumeName2, aCR.theComptonVolumeName2);
    strcpy(theRayleighVolumeName1, aCR.theRayleighVolumeName1);
    strcpy(theRayleighVolumeName2, aCR.theRayleighVolumeName2);
    return *this;
}

//--------------------------------------------------------------------------
GateToRoot::GateToRoot(const G4String &name, GateOutputMgr *outputMgr, DigiMode digiMode)
        : GateVOutputModule(name, outputMgr, digiMode), m_hfile(0), m_treeHit(0),
          m_rootHitFlag(digiMode == kruntimeMode), m_rootNtupleFlag(true), m_saveRndmFlag(true),
          m_fileName(" ") // All default output file from all output modules are set to " ".
        // They are then checked in GateApplicationMgr::StartDAQ, using
        // the VOutputModule pure virtual method GiveNameOfFile()
        , m_rootMessenger(0) {
    /*
      if (digiMode==kofflineMode)
      m_fileName="digigate";
    */
    m_isEnabled = false; // Keep this flag false: all output are disabled by default
    nVerboseLevel = 0;

    m_rootMessenger = new GateToRootMessenger(this);

    m_recordFlag = 0; // Design to embrace obsolete functions (histogram, recordVoxels, ...)
    latestEventID = 0.; // Used by gjs and gjm programs (cluster mode)
    nbPrimaries = 0.; // To have the total number of emitted primaries (stored at endOfAcq in an histo)

    // required by newer versions of Root to open a TCanvas on the screen
    //TApplication *theApp = new TApplication("App", ((int *)0), ((char **)0),0,0);
    //TApplication *theApp = new TApplication("App",0, 0,0,0);

    /* PY Descourt Tracker/Detector 08/09/2009  */
    m_TracksFile = 0;
    m_EOF = 0;
    m_currentRSData = 0;
    fSkipRecStepData = 0;
    last_RSEventID = 0;
    m_currentTracksData = 0;
    m_currentGTrack = 0;
    /* PY Descourt Tracker/Detector 08/09/2009  */

    // v. cuplov - optical photons
    m_trajectoryNavigator = new GateTrajectoryNavigator();
    // v. cuplov - optical photons

}
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
GateToRoot::~GateToRoot() {
    delete m_rootMessenger;
    if (nVerboseLevel > 0) G4cout << "GateToRoot deleting...\n";
    for (size_t i = 0; i < m_outputChannelList.size(); ++i)
        delete m_outputChannelList[i];

    // v. cuplov - optical photons
    delete m_trajectoryNavigator;
    // v. cuplov - optical photons

}
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
const G4String &GateToRoot::GiveNameOfFile() {
    return m_fileName;

}
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
void GateToRoot::Book() {

    if (nVerboseLevel > 2)
        G4cout << "GateToRoot::Book\n";

    if (m_recordFlag > 0) {
        //TH1F *hist;
        G4String hist_name;
        G4String hist_title;
        hist_name = "Positron_Kinetic_Energy_MeV";
        hist_title = "Positron Kinetic Energy (MeV)";
        //hist = new TH1F(hist_name,hist_title,100,0.0, 5.0);
        new TH1F(hist_name, hist_title, 100, 0.0, 5.0);

        hist_name = "Ion_decay_time_s";
        hist_title = "Ion decay time (s)";
        //hist = new TH1F(hist_name,hist_title,100,0., 1.E3);
        new TH1F(hist_name, hist_title, 100, 0., 1.E3);

        hist_name = "Positron_annihil_distance_mm";
        hist_title = "Positron annihilation distance (mm)";
        //hist = new TH1F(hist_name,hist_title,100,0., 1.E1);
        new TH1F(hist_name, hist_title, 100, 0., 1.E1);

        hist_name = "Acolinea_Angle_Distribution_deg";
        hist_title = "Acolinearity Angle Distribution (deg)";
        // hist = new TH1F(hist_name,hist_title,100,-5, 5);
        new TH1F(hist_name, hist_title, 100, -5, 5);

        //! Root simple ntuple, float sequence
        //TNtuple *ntuple;
        G4String ntuple_name = "Gate";
        if (nVerboseLevel > 0)
            G4cout
                    << "GateToRoot: ROOT: Ntuple " << ntuple_name << " being Created\n";
        //ntuple = new TNtuple(ntuple_name,"Gate","event:iontime:poskinene:posannihildist");
        new TNtuple(ntuple_name, "Gate", "event:iontime:poskinene:posannihildist");

    }

    //! This histogram will be needed when using the cluster mode of Gate.
    //! When we merge all root files, we need the last event ID
    //TH1D* latest_histo;
    G4String hist_name = "latest_event_ID";
    G4String hist_title = "latest_event_ID(#)";
    //latest_histo=new TH1D(hist_name,hist_title,100,0,900000000000.);
    new TH1D(hist_name, hist_title, 100, 0, 900000000000.);

    //! This histogram will be used to store in a ROOT tree the total
    //! number of emitted primaries.
    //TH1D* primaries_histo;
    hist_name = "total_nb_primaries";
    hist_title = "total_nb_primaries(#)";
    //primaries_histo = new TH1D(hist_name,hist_title,100,0,900000000000.);
    m_total_nb_primaries_hist = new TH1D(hist_name, hist_title, 100, 0, 900000000000.);

    // Additional tree/branches to store data used for PET analysis
    /* This is duplicated from the previous 'total_nb_primaries' value, but it was stored as an histogram
       (dont ask me why) and it is no easy to deal with, in particular with uproot.
    */
    auto pet_data = new TTree("pet_data", "data for PET analysis");
    pet_data->Branch("total_nb_primaries", &nbPrimaries);
    pet_data->Branch("latest_event_ID", &latestEventID);
    pet_data->Branch("start_time_sec", &mTimeStart);
    pet_data->Branch("stop_time_sec", &mTimeStop);

    m_treeHit = new GateHitTree(GateHitConvertor::GetOutputAlias());
    m_treeHit->Init(m_hitBuffer);

    // v. cuplov - optical photons
    OpticalTree = new TTree(G4String("OpticalData").c_str(), "OpticalData");

    OpticalTree->Branch(G4String("NumScintillation").c_str(), &nScintillation, "nScintillation/I");
    OpticalTree->Branch(G4String("NumCrystalWLS").c_str(), &NumCrystalWLS, "NumCrystalWLS/I");
    OpticalTree->Branch(G4String("NumPhantomWLS").c_str(), &NumPhantomWLS, "NumPhantomWLS/I");
    OpticalTree->Branch(G4String("CrystalLastHitPos_X").c_str(), &CrystalLastHitPos_X, "CrystalLastHitPos_X/D");
    OpticalTree->Branch(G4String("CrystalLastHitPos_Y").c_str(), &CrystalLastHitPos_Y, "CrystalLastHitPos_Y/D");
    OpticalTree->Branch(G4String("CrystalLastHitPos_Z").c_str(), &CrystalLastHitPos_Z, "CrystalLastHitPos_Z/D");
    OpticalTree->Branch(G4String("CrystalLastHitEnergy").c_str(), &CrystalLastHitEnergy, "CrystalLastHitEnergy/D");
    OpticalTree->Branch(G4String("PhantomLastHitPos_X").c_str(), &PhantomLastHitPos_X, "PhantomLastHitPos_X/D");
    OpticalTree->Branch(G4String("PhantomLastHitPos_Y").c_str(), &PhantomLastHitPos_Y, "PhantomLastHitPos_Y/D");
    OpticalTree->Branch(G4String("PhantomLastHitPos_Z").c_str(), &PhantomLastHitPos_Z, "PhantomLastHitPos_Z/D");
    OpticalTree->Branch(G4String("PhantomLastHitEnergy").c_str(), &PhantomLastHitEnergy, "PhantomLastHitEnergy/D");
    //  OpticalTree->Branch(G4String("CrystalAbsorbedPhotonHitPos_X").c_str(),&CrystalAbsorbedPhotonHitPos_X,"CrystalAbsorbedPhotonHitPos_X/D");
    //  OpticalTree->Branch(G4String("CrystalAbsorbedPhotonHitPos_Y").c_str(),&CrystalAbsorbedPhotonHitPos_Y,"CrystalAbsorbedPhotonHitPos_Y/D");
    //  OpticalTree->Branch(G4String("CrystalAbsorbedPhotonHitPos_Z").c_str(),&CrystalAbsorbedPhotonHitPos_Z,"CrystalAbsorbedPhotonHitPos_Z/D");
    //  OpticalTree->Branch(G4String("PhantomAbsorbedPhotonHitPos_X").c_str(),&PhantomAbsorbedPhotonHitPos_X,"PhantomAbsorbedPhotonHitPos_X/D");
    //  OpticalTree->Branch(G4String("PhantomAbsorbedPhotonHitPos_Y").c_str(),&PhantomAbsorbedPhotonHitPos_Y,"PhantomAbsorbedPhotonHitPos_Y/D");
    //  OpticalTree->Branch(G4String("PhantomAbsorbedPhotonHitPos_Z").c_str(),&PhantomAbsorbedPhotonHitPos_Z,"PhantomAbsorbedPhotonHitPos_Z/D");
    OpticalTree->Branch(G4String("PhantomWLSPos_X").c_str(), &PhantomWLSPos_X, "PhantomWLSPos_X/D");
    OpticalTree->Branch(G4String("PhantomWLSPos_Y").c_str(), &PhantomWLSPos_Y, "PhantomWLSPos_Y/D");
    OpticalTree->Branch(G4String("PhantomWLSPos_Z").c_str(), &PhantomWLSPos_Z, "PhantomWLSPos_Z/D");
    //  OpticalTree->Branch(G4String("NumCrystalOptAbs").c_str(),&nCrystalOpticalAbsorption,"nCrystalOpticalAbsorption/I");
    //  OpticalTree->Branch(G4String("NumCrystalOptRay").c_str(),&nCrystalOpticalRayleigh,"nCrystalOpticalRayleigh/I");
    //  OpticalTree->Branch(G4String("NumCrystalOptMie").c_str(),&nCrystalOpticalMie,"nCrystalOpticalMie/I");
    //  OpticalTree->Branch(G4String("NumPhantomOptAbs").c_str(),&nPhantomOpticalAbsorption,"nPhantomOpticalAbsorption/I");
    //  OpticalTree->Branch(G4String("NumPhantomOptRay").c_str(),&nPhantomOpticalRayleigh,"nPhantomOpticalRayleigh/I");
    //  OpticalTree->Branch(G4String("NumPhantomOptMie").c_str(),&nPhantomOpticalMie,"nPhantomOpticalMie/I");
    OpticalTree->Branch(G4String("PhantomProcessName").c_str(), &NameOfProcessInPhantom, "PhantomProcessName/C");
    OpticalTree->Branch(G4String("CrystalProcessName").c_str(), &NameOfProcessInCrystal, "CrystalProcessName/C");
    OpticalTree->Branch(G4String("MomentumDirectionx").c_str(), &MomentumDirectionx, "MomentumDirectionx/D");
    OpticalTree->Branch(G4String("MomentumDirectiony").c_str(), &MomentumDirectiony, "MomentumDirectiony/D");
    OpticalTree->Branch(G4String("MomentumDirectionz").c_str(), &MomentumDirectionz, "MomentumDirectionz/D");
    // v. cuplov - optical photons

    for (size_t i = 0; i < m_outputChannelList.size(); ++i)
        m_outputChannelList[i]->Book();


    m_working_root_directory = TDirectory::CurrentDirectory();

}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
// Method called at the beginning of each acquisition by the application manager: opens the ROOT file and prepare the trees
void GateToRoot::RecordBeginOfAcquisition() {

    if (nVerboseLevel > 2)
        G4cout << "GateToRoot::RecordBeginOfAcquisition\n";

    GateSteppingAction *myAction = ((GateSteppingAction *) (GateRunManager::GetRunManager()->GetUserSteppingAction()));
    TrackingMode theMode = myAction->GetMode();
    if (nVerboseLevel > 1)
        G4cout << " GateToRoot::RecordBeginOfAcquisition()  Tracking Mode " << int(theMode) << Gateendl;

    // PY. Descourt 11/12/2008
    // NORMAL OR DETECTOR MODE
    if ((theMode == TrackingMode::kBoth) || (theMode == TrackingMode::kDetector)) {
        /////////////////////////////////////
        //////////////////////
        //////////
        ////
        //DETECTOR MODE : open the Tracks data Root File
        if (theMode == TrackingMode::kDetector) {

            m_currentGTrack = new GateTrack();

            /// OPEN ROOT DETECTOR OUTPUT FILE

            OpenTracksFile();
            if (nVerboseLevel > 1) G4cout << "GateToRoot::RecordBeginOfAcquisition  gROOT is " << gROOT << Gateendl;
        }
        ////
        //////////
        // Open the output file
        if (nVerboseLevel > 0) G4cout << "GateToRoot: ROOT: files creation...\n";
        switch (m_digiMode) {
            case kruntimeMode:
                // In run-time mode, we open the file in RECREATE mode

                // v. cuplov - m_fileName from SetFileName is defined without ".root" (see changes in GateToRoot.hh)
                // Additionnal root files names will be of the form GateOutPut_additionnalName.root.
                // In the previous version of the code, file names would appear as GateOutPut.root_additionnalName.root
                //    m_hfile = new TFile( GetFilePath() ,"RECREATE","ROOT file with histograms");
                m_hfile = new TFile((m_fileName + ".root").c_str(), "RECREATE", "ROOT file with histograms");
                // v. cuplov

                break;
            case kofflineMode:
                // In DigiGate mode, we first check that the file does not exist. If it does, we abort as we want to make sure the ROOT file is not overwritten by accident

                // v. cuplov - m_fileName from SetFileName is defined without ".root" (see changes in GateToRoot.hh)
                // Additionnal root files names will be of the form GateOutPut_additionnalName.root.
                // In the previous version of the code, file names would appear as GateOutPut.root_additionnalName.root
                //    FILE* rootFile = fopen ( GetFilePath() , "r");
                FILE *rootFile = fopen((m_fileName + ".root").c_str(), "r");
                // v. cuplov

                if (rootFile != NULL) {
                    fclose(rootFile);
                    G4String msg = " I am sorry, but there is already a ROOT file '";
                    msg += GetFilePath();
                    msg += "' on your directory. In Digigate mode, you are not allowed to overwrite ROOT files (security to avoid accidental data destruction). You must remove the file '";
                    msg += GetFilePath();
                    msg += "' of the current directory (move it elsewhere or delete it), then launch DigiGate again.\n Note that you could also change the name of your output ROOT file with the command:\n\t /gate/output/root/setFileName NEW_NAME";

                    G4Exception("GateToRoot::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException,
                                msg);
                }

                // v. cuplov - m_fileName from SetFileName is defined without ".root" (see changes in GateToRoot.hh)
                // Additionnal root files names will be of the form GateOutPut_additionnalName.root.
                // In the previous version of the code, file names would appear as GateOutPut.root_additionnalName.root
                //    m_hfile = new TFile( GetFilePath(),"CREATE","ROOT file with histograms");
                m_hfile = new TFile((m_fileName + ".root").c_str(), "CREATE", "ROOT file with histograms");
                // v. cuplov

                break;
        }

        // Check that we succeeded in opening the file
        if (!m_hfile) {
            G4String msg = "Could not open the requested output ROOT file '" + m_fileName + ".root'!";
            G4Exception("GateToRoot::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, msg);
        }
        if (!(m_hfile->IsOpen())) {
            G4String msg = "Could not open the requested output ROOT file '" + m_fileName + ".root'!";
            G4Exception("GateToRoot::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, msg);
        }
        //! We book histos and ntuples only once per acquisition
        Book();


        return;
    }
    if (theMode == TrackingMode::kTracker) {

        m_currentGTrack = new GateTrack();

        //////////////////////////////////////////   TRACKS INFOS ROOT OUTPUT FILE /////////////////////////////////////////////////////////////////////////////////
        G4cout << "Tracker Mode detected..." << Gateendl;
        G4cout << "GateToRoot::RecordBeginOfAcquisition()   OPENING " << (m_fileName + "_TrackerData.root") << " file "
               << Gateendl;

        m_hfile = new TFile((m_fileName + "_TrackerData.root").c_str(), "RECREATE", "ROOT file with Tracker Data");

        // Check that we succeeded in opening the file
        if (!m_hfile) {
            G4String msg = "Could not open the requested output ROOT file '" + m_fileName + "_TrackerData.root'!";
            G4Exception("GateToRoot::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, msg);
        }
        if (!(m_hfile->IsOpen())) {
            G4String msg = "Could not open the requested output ROOT file '" + m_fileName + "_TrackerData.root'!";
            G4Exception("GateToRoot::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, msg);
        }
        if (nVerboseLevel > 0) G4cout << "GateToRoot: ROOT: Ntuple " << "PhTracksData" << " being Created\n";

        tracksTuple = new TTree(G4String("PhTracksData").c_str(), "PhantomTracksData");

        tracksTuple->Branch(G4String("RunID").c_str(), &RunID, "RunID/I");
        tracksTuple->Branch(G4String("TrackID").c_str(), &TrackID, "TrackID/I");
        tracksTuple->Branch(G4String("ParentID").c_str(), &ParentID, "ParentID/I");
        tracksTuple->Branch(G4String("Pos_x").c_str(), &posx, "posx/D");
        tracksTuple->Branch(G4String("Pos_y").c_str(), &posy, "posy/D");
        tracksTuple->Branch(G4String("Pos_z").c_str(), &posz, "posz/D");
        tracksTuple->Branch(G4String("LTime").c_str(), &LTime, "LTime/D");
        tracksTuple->Branch(G4String("GTime").c_str(), &GTime, "GTime/D");
        tracksTuple->Branch(G4String("PTime").c_str(), &PTime, "PTime/D");
        tracksTuple->Branch(G4String("MDirection_x").c_str(), &MDirectionx, "MDirectionx/D");
        tracksTuple->Branch(G4String("MDirection_y").c_str(), &MDirectiony, "MDirectiony/D");
        tracksTuple->Branch(G4String("MDirection_z").c_str(), &MDirectionz, "MDirectionz/D");
        tracksTuple->Branch(G4String("Momentum_x").c_str(), &Momentumx, "Momentumx/D");
        tracksTuple->Branch(G4String("Momentum_y").c_str(), &Momentumy, "Momentumy/D");
        tracksTuple->Branch(G4String("Momentum_z").c_str(), &Momentumz, "Momentumz/D");
        tracksTuple->Branch(G4String("Energy").c_str(), &Energy, "Energy/D");
        tracksTuple->Branch(G4String("Wavelength").c_str(), &Wavelength, "Wavelength/D"); // v. cuplov - wavelength
        tracksTuple->Branch(G4String("Kinenergy").c_str(), &KinEnergy, "KinEnergy/D");
        tracksTuple->Branch(G4String("Velocity").c_str(), &Velocity, "Velocity/D");
        tracksTuple->Branch(G4String("Vertexposition_x").c_str(), &VertexPositionx, "VertexPositionx/D");
        tracksTuple->Branch(G4String("Vertexposition_y").c_str(), &VertexPositiony, "VertexPositiony/D");
        tracksTuple->Branch(G4String("Vertexposition_z").c_str(), &VertexPositionz, "VertexPositionz/D");
        tracksTuple->Branch(G4String("Vertexmomentumdirection_x").c_str(), &VtxMomDirx, "VtxMomDirx/D");
        tracksTuple->Branch(G4String("Vertexmomentumdirection_y").c_str(), &VtxMomDiry, "VtxMomDiry/D");
        tracksTuple->Branch(G4String("Vertexmomentumdirection_z").c_str(), &VtxMomDirz, "VtxMomDirz/D");
        tracksTuple->Branch(G4String("VertexKineticEnergy").c_str(), &VertexKineticEnergy, "VertexKineticEnergy/D");
        tracksTuple->Branch(G4String("Polarization_x").c_str(), &Polarizationx, "Polarizationx/D");
        tracksTuple->Branch(G4String("Polarization_y").c_str(), &Polarizationy, "Polarizationy/D");
        tracksTuple->Branch(G4String("Polarization_z").c_str(), &Polarizationz, "Polarizationz/D");
        tracksTuple->Branch(G4String("Weight").c_str(), &Weight, "Weight/D");
        tracksTuple->Branch(G4String("EventID").c_str(), &EventID, "EventID/I");
        tracksTuple->Branch(G4String("EventTime").c_str(), &m_EventTime, "EventTime/D");
        tracksTuple->Branch(G4String("PDGCode").c_str(), &PDGCode, "PDGCode/I");
        tracksTuple->Branch(G4String("SourceID").c_str(), &m_sourceID, "Source_ID/I");
        tracksTuple->Branch(G4String("WasKilled").c_str(), &m_wasKilled, "WasKIlled/I");
        tracksTuple->Branch(G4String("ProcessName").c_str(), &m_processName, "ProcessName/C");
        tracksTuple->Branch(G4String("PPName").c_str(), &m_parentparticleName, "parentparticleName/C");
        tracksTuple->Branch(G4String("LogAtVertex").c_str(), &m_volumeName, "LogicalVolAtVertex/C");

        // we also store the data collected by the RecordStep method during stepping process and the datas for each event on the number of compton &  rayleigh scatterings

        m_RecStepTree = new TTree(G4String("RecStepData").c_str(), "RecordSteppingData");

        m_RecStepTree->Branch(G4String("IonDecayPos").c_str(), &m_ionDecayPos,
                              "m_ionDecayPos.x()/D:m_ionDecayPos.y()/D:m_ionDecayPos.z()/D");
        m_RecStepTree->Branch(G4String("PositronGenerationPos").c_str(), &m_positronGenerationPos,
                              "m_positronGenerationPos.x()/D:m_positronGenerationPos.y()/D:m_positronGenerationPos.z()/D");
        m_RecStepTree->Branch(G4String("PositronAnnihilPos").c_str(), &m_positronAnnihilPos,
                              "m_positronAnnihilPos.x()/D:m_positronAnnihilPos.y()/D:m_positronAnnihilPos.z()/D");
        m_RecStepTree->Branch(G4String("PositronKinEnergy").c_str(), &m_positronKinEnergy, "m_positronKinEnergy/D");
        m_RecStepTree->Branch(G4String("dxg1").c_str(), &dxg1, "dxg1/D");
        m_RecStepTree->Branch(G4String("dyg1").c_str(), &dyg1, "dyg1/D");
        m_RecStepTree->Branch(G4String("dzg1").c_str(), &dzg1, "dzg1/D");
        m_RecStepTree->Branch(G4String("dxg2").c_str(), &dxg2, "dxg2/D");
        m_RecStepTree->Branch(G4String("dyg2").c_str(), &dyg2, "dyg2/D");
        m_RecStepTree->Branch(G4String("dzg2").c_str(), &dzg2, "dzg2/D");
        m_RecStepTree->Branch(G4String("photon1PhR").c_str(), &theCRData.photon1_phantom_Rayleigh, "photon1PhR/I");
        m_RecStepTree->Branch(G4String("photon2PhR").c_str(), &theCRData.photon2_phantom_Rayleigh, "photon2PhR/I");
        m_RecStepTree->Branch(G4String("photon1PhC").c_str(), &theCRData.photon1_phantom_compton, "photon1PhC/I");
        m_RecStepTree->Branch(G4String("photon2PhC").c_str(), &theCRData.photon2_phantom_compton, "photon2PhC/I");
        m_RecStepTree->Branch(G4String("ComptVol1").c_str(), &theCRData.theComptonVolumeName1, "ComptVol1/C");
        m_RecStepTree->Branch(G4String("ComptVol2").c_str(), &theCRData.theComptonVolumeName2, "ComptVol2/C");
        m_RecStepTree->Branch(G4String("RaylVol1").c_str(), &theCRData.theRayleighVolumeName1, "RaylVol1/C");
        m_RecStepTree->Branch(G4String("RaylVol2").c_str(), &theCRData.theRayleighVolumeName2, "RaylVol2/C");
        m_RecStepTree->Branch(G4String("EventID").c_str(), &m_RSEventID, "EventID/I");
        m_RecStepTree->Branch(G4String("RunID").c_str(), &m_RSRunID, "RunID/I");
        ////////////////////////
    }

}
//--------------------------------------------------------------------------



//--------------------------------------------------------------------------
void GateToRoot::RecordEndOfAcquisition() {
    //GateMessage("Output", 5, " GateToRoot::RecordEndOfAcquisition -- begin.\n";);



    //=================  cluster  ===============================================
    //we store the latest event ID at end of time split
    //we have to consider the case that virtualStop == end of Timeslice
    //in this case we leave latesteventID=0
    G4double virtualStop = GateApplicationMgr::GetInstance()->GetVirtualTimeStop();
    G4double timeStart = GateApplicationMgr::GetInstance()->GetTimeStart();
    G4double timeSlice = GateApplicationMgr::GetInstance()->GetTimeSlice();

    //! We need to fill this histogram with the latest eventID of the simulation.
    //! This will be used when merging all root output in cluster mode.
    if (virtualStop != -1. && fmod((virtualStop - timeStart), timeSlice) > 1e-10) {
        TH1D *latest_histo;
        G4String hist_name;
        hist_name = "latest_event_ID";
        if ((latest_histo = (TH1D *) m_working_root_directory->GetList()->FindObject(hist_name)) != NULL) {
            latest_histo->Fill(latestEventID - 1.0);
        } else {
            G4cerr
                    << "GateToRoot::RecordEndOfAcquisition(): Failed to access to 'latest_event_ID' histogram to fill it !\n";
        }
    }
    //=============================================================================

    // Here we store the total number of primaries in the TH1D histo
    TH1D *primaries_histo;
    G4String hist_name = "total_nb_primaries";
    if ((primaries_histo = (TH1D *) m_working_root_directory->GetList()->FindObject(hist_name)) != NULL) {
        primaries_histo->Fill(nbPrimaries);
    } else {
        G4cerr
                << "GateToRoot::RecordEndOfAcquisition(): Failed to access to 'total_nb_primaries' histogram to fill it !\n";
    }

    // Store the data for pet analysis

    // Remove 1 because the increment was before the end
    latestEventID = latestEventID - 1;
    // get the time in second
    mTimeStop = GateApplicationMgr::GetInstance()->GetTimeStop() / second;
    mTimeStart = GateApplicationMgr::GetInstance()->GetTimeStart() / second;
    // all variables linked in the "pet_data" tree (define below) will be written
    auto t = (TTree *) m_working_root_directory->GetList()->FindObject("pet_data");
    t->Fill();


    /* PY Descourt 08/09/2009 */
    GateSteppingAction *myAction = ((GateSteppingAction *) (GateRunManager::GetRunManager()->GetUserSteppingAction()));
    TrackingMode theMode = myAction->GetMode();
    if (theMode == TrackingMode::kTracker) {
        G4cout << " ----- ROOT FILE DATA INFORMATIONS ----- \n";
        if (tracksTuple != 0) { tracksTuple->Print(); }
        if (m_RecStepTree != 0) { m_RecStepTree->Print(); }
        ///// IMPORTANT NOTE
        /////    in case we have a lot of data being written, Root automatically create new files when file size exceeds 1.9 GBytes and closes files on the fly
        //// we need first to get the right pointer on the last file with the TTree method TTree::GetCurrentFile() which returns a pointer to the opened current Root File
        ///  which is not the one we intstantiated if more than one Root File has been written
        ////
        m_hfile = tracksTuple->GetCurrentFile();
        G4cout << " GateToRoot::RecordEndOfAcquisition() : Tracker MODE  ::::::::    current Root Tracks Data File  = "
               << m_hfile << " named " << m_hfile->GetName() << Gateendl;
        //  if (m_verboseLevel > 0)
        G4cout << "GateToRoot: ROOT: files writing...\n";
        m_hfile->Write(0,TObject::kOverwrite);
        //  if (m_verboseLevel > 0)
        G4cout << "GateToRoot: ROOT: files closing...\n";
        if (m_hfile->IsOpen()) { m_hfile->Close(); }
    }

    if ((theMode == TrackingMode::kBoth) || (theMode == TrackingMode::kDetector)) {
        //!    IMPORTANT NOTE
        //!    in case we have a lot of data being written, Root automatically
        //!    create new files when file size exceeds 1.9 GBytes and closes files on the fly
        //!    we need first to get the right pointer on the last file with the
        //!    TTree method TTree::GetCurrentFile() which returns a pointer to the opened
        //!    current Root File
        //!  which is not the one we intstantiated if more than one Root File has been written

        m_hfile = m_treeHit->GetCurrentFile();

        if (nVerboseLevel > 0)
            G4cout << "GateToRoot: ROOT: files writing...\n";
        //GateMessage("Output", 1, " GateToRoot: ROOT: files writing...\n";);
        m_hfile->Write(0,TObject::kOverwrite);

        if (nVerboseLevel > 0)
            G4cout << "GateToRoot: ROOT: files closing...\n";
        //GateMessage("Output", 1, " GateToRoot: ROOT: files closing...\n";);
        m_hfile->Close();
    }
    /* PY Descourt 08/09/2009 */

}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRoot::RecordBeginOfRun(const G4Run *) {
    //reset latestEventID
    latestEventID = 0;
    // save Rndm status
    /*if (m_saveRndmFlag)
      { CLHEP::HepRandom::showEngineStatus();
      CLHEP::HepRandom::saveEngineStatus("beginOfRun.rndm");
      FILE* rndmFile = fopen ( "endOfRun.rndm" , "r");
      if (rndmFile!=NULL) {
      fclose(rndmFile);
      CLHEP::HepRandom::restoreEngineStatus("endOfRun.rndm");
      }
      }*/

    if (nVerboseLevel > 2)
        G4cout << "GateToRoot::RecordBeginOfRun\n";
    //  Book();
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRoot::RecordEndOfRun(const G4Run *) {
    //! if the flag is not set to >0, don't record the event
    //  m_recordFlag = 0;
    // save Rndm status
    /* if (m_saveRndmFlag)
       {
       CLHEP::HepRandom::showEngineStatus();
       CLHEP::HepRandom::saveEngineStatus("endOfRun.rndm");
       }*/
    if (nVerboseLevel > 2)
        G4cout << "GateToRoot::RecordEndOfRun\n";

    nbPrimaries -= 1.; // Number of primaries increase too much at each end of run !


}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRoot::RecordBeginOfEvent(const G4Event *evt) {


    //  GateMessage("Output", 5 , " GateToRoot::RecordBeginOfEvent -- begin\n";);

    if (nVerboseLevel > 2)
        G4cout << "GateToRoot::RecordBeginOfEvent\n";

    m_hitBuffer.Clear();

    for (size_t i = 0; i < m_outputChannelList.size(); ++i)
        m_outputChannelList[i]->Clear();

    /*PY Descourt 08/09/2009 */

    theCRData.photon1_phantom_Rayleigh = 0;
    theCRData.photon2_phantom_Rayleigh = 0;
    theCRData.photon1_phantom_compton = 0;
    theCRData.photon2_phantom_compton = 0;
    strcpy(theCRData.theComptonVolumeName1, G4String("NULL").c_str());
    strcpy(theCRData.theComptonVolumeName2, G4String("NULL").c_str());
    strcpy(theCRData.theRayleighVolumeName1, G4String("NULL").c_str());
    strcpy(theCRData.theRayleighVolumeName2, G4String("NULL").c_str());
    m_ionDecayPos = G4ThreeVector(0., 0., 0.);
    m_positronGenerationPos = G4ThreeVector(0., 0., 0.);
    m_positronAnnihilPos = G4ThreeVector(0., 0., 0.);
    dxg1 = 0.;
    dyg1 = 0.;
    dzg1 = 0.;
    dxg2 = 0.;
    dyg2 = 0.;
    dzg2 = 0.;

    theCRData.photon1_phantom_Rayleigh = 0;
    theCRData.photon2_phantom_Rayleigh = 0;
    theCRData.photon1_phantom_compton = 0;
    theCRData.photon2_phantom_compton = 0;
    strcpy(theCRData.theComptonVolumeName1, G4String("NULL").c_str());
    strcpy(theCRData.theComptonVolumeName2, G4String("NULL").c_str());
    strcpy(theCRData.theRayleighVolumeName1, G4String("NULL").c_str());
    strcpy(theCRData.theRayleighVolumeName2, G4String("NULL").c_str());


    TrackingMode theMode = ((GateSteppingAction *) (GateRunManager::GetRunManager()->GetUserSteppingAction()))->GetMode();
    if ((theMode == TrackingMode::kDetector) && (evt->GetNumberOfPrimaryVertex() > 0)) {

        // we read the RecStep and number of rayleigh & compton scatterings from the RecStep Data Root file
        if (fSkipRecStepData == 0) GetCurrentRecStepData(evt);
        else {
            if (nVerboseLevel > 3)
                G4cout << "GateToRoot::RecordBeginOfEvent   WARNING  fSkipRecStepData = " << fSkipRecStepData
                       << Gateendl;
            theCRData = theCRData_copy;
            m_positronKinEnergy = m_positronKinEnergy_copy;
            m_ionDecayPos = m_ionDecayPos_copy;
            m_positronGenerationPos = m_positronGenerationPos_copy;
            m_positronAnnihilPos = m_positronAnnihilPos_copy;
            dxg1 = dxg1_copy;
            dyg1 = dyg1_copy;
            dzg1 = dzg1_copy;
            dxg2 = dxg2_copy;
            dyg2 = dyg2_copy;
            dzg2 = dzg2_copy;
            fSkipRecStepData = 0;
            //G4cout << "GateToRoot::RecordBeginOfEvent \n";
            //PrintRecStep();
        }

        // PrintRecStep();
    }

    /*PY Descourt 08/09/2009 */

    //  GateMessage("Output", 5, " GateToRoot::RecordBeginOfEvent -- end\n";);

}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRoot::RecordEndOfEvent(const G4Event *event) {

    // GateMessage("Output", 5 , " GateToRoot::RecordEndOfEvent -- begin\n";);


    GateSteppingAction *myAction = ((GateSteppingAction *) (GateRunManager::GetRunManager()->GetUserSteppingAction()));
    TrackingMode theMode = myAction->GetMode();
    if (theMode == TrackingMode::kTracker)return;

    nbPrimaries += 1.;
    latestEventID += 1.;

    GateCrystalHitsCollection *CHC = GetOutputMgr()->GetCrystalHitCollection();

    if (CHC) {

        // Hits loop

        G4int NbHits = CHC->entries();

        for (G4int iHit = 0; iHit < NbHits; iHit++) {

            GateCrystalHit *aHit = (*CHC)[iHit];
            G4String processName = aHit->GetProcess();
            G4int PDGEncoding = aHit->GetPDGEncoding();

            if (nVerboseLevel > 2)
                G4cout
                        << "GateToRoot::RecordEndOfEvent : CrystalHitsCollection: processName : <" << processName
                        << ">    Particls PDG code : " << PDGEncoding << Gateendl;

            if (aHit->GoodForAnalysis()) {
                m_hitBuffer.Fill(aHit);
                if (nVerboseLevel > 1)
                    G4cout << "GateToRoot::RecordEndOfEvent : m_treeHit->Fill\n";


                if (m_rootHitFlag) m_treeHit->Fill();
            }
        }


        if (m_recordFlag > 0) {
            G4double eventTime = (GateSourceMgr::GetInstance())->GetTime();
            TH1F *hist;
            G4String hist_name;
            hist_name = "Positron_Kinetic_Energy_MeV";
            if ((hist = (TH1F *) m_working_root_directory->GetList()->FindObject(hist_name)) != NULL) {

                hist->Fill(m_positronKinEnergy / MeV);
            } else {
                if (nVerboseLevel > 0)
                    G4cout
                            << "GateToRoot: ROOT: Cannot find histo" << hist_name << Gateendl;
            }
            hist_name = "Ion_decay_time_s";
            hist = NULL;
            if ((hist = (TH1F *) m_working_root_directory->GetList()->FindObject(hist_name)) != NULL) {
                hist->Fill(eventTime / s);
            } else {
                if (nVerboseLevel > 0)
                    G4cout
                            << "GateToRoot:  ROOT: Cannot find histo" << hist_name << Gateendl;
            }
            G4ThreeVector posAnnihilDist = m_positronAnnihilPos - m_positronGenerationPos;
            hist_name = "Positron_annihil_distance_mm";
            hist = NULL;
            if ((hist = (TH1F *) m_working_root_directory->GetList()->FindObject(hist_name)) != NULL) {
                hist->Fill(posAnnihilDist.mag() / mm);
            } else {
                if (nVerboseLevel > 0)
                    G4cout
                            << "GateToRoot:  ROOT: Cannot find histo" << hist_name << Gateendl;
            }

            // Histo of acolinearity angle distribution

            G4double dev = (dxg1 * dxg2 + dyg1 * dyg2 + dzg1 * dzg2) /
                           ((sqrt(dxg1 * dxg1 + dyg1 * dyg1 + dzg1 * dzg1)) *
                            (sqrt(dxg2 * dxg2 + dyg2 * dyg2 + dzg2 * dzg2)));
            if (dzg1 > dzg2) { dev = rad2deg(acos(-dev)); }
            else { dev = rad2deg(acos(dev)) - 180; }

            if (std::isnan(dev)) dev = 0.;

            // G4cout<< " dev = " << dev << Gateendl;

            hist_name = "Acolinea_Angle_Distribution_deg";
            hist = NULL;
            if ((hist = (TH1F *) m_working_root_directory->GetList()->FindObject(hist_name)) != NULL) {
                hist->Fill(dev);
            } else {
                //if (nVerboseLevel > 0)
                G4cout << "GateToRoot:  ROOT: Cannot find histo " << hist_name << Gateendl;
            }

            TNtuple *ntuple;
            G4String ntuple_name = "Gate";
            if ((ntuple = (TNtuple *) m_working_root_directory->GetList()->FindObject(ntuple_name)) == NULL) {
                if (nVerboseLevel > 0)
                    G4cout
                            << "GateToRoot: ROOT: Cannot find ntuple " << ntuple_name << Gateendl;
            } else {
                //! better than the simple eventID, but still not enough: it's valid only for
                //! the single run and not for the application
                G4int iEvent = ((GatePrimaryGeneratorAction *) GateRunManager::GetRunManager()->
                        GetUserPrimaryGeneratorAction())->GetEventNumber();
                if (m_rootNtupleFlag)
                    ntuple->Fill(iEvent,
                                 eventTime / s,
                                 m_positronKinEnergy / MeV,
                                 posAnnihilDist.mag() / mm);
            }

        }

    }

    RecordDigitizer(event);

    // v. cuplov - optical photons
    RecordOpticalData(event);
    // v. cuplov - optical photons

    // GateMessage("Output", 5, " GateToRoot::RecordEndOfEvent -- end\n";);

}

//--------------------------------------------------------------------------

// v.cuplov - optical photon: Record OpticalPhoton Data
void GateToRoot::RecordOpticalData(const G4Event *event) {
    G4TrajectoryContainer *trajectoryContainer = event->GetTrajectoryContainer();
    if (trajectoryContainer) m_trajectoryNavigator->SetTrajectoryContainer(trajectoryContainer);

    GateCrystalHitsCollection *CHC = GetOutputMgr()->GetCrystalHitCollection();
    GatePhantomHitsCollection *PHC = GetOutputMgr()->GetPhantomHitCollection();

    // Initialization of variables:
    //   nPhantomOpticalRayleigh = 0;
    //   nPhantomOpticalMie = 0;
    //   nPhantomOpticalAbsorption = 0;
    //   nCrystalOpticalRayleigh = 0;
    //   nCrystalOpticalMie = 0;
    //   nCrystalOpticalAbsorption = 0;

    nScintillation = 0;
    nCrystalOpticalWLS = 0;
    nPhantomOpticalWLS = 0;
    NumCrystalWLS = 0;
    NumPhantomWLS = 0;

    // Looking at Phantom Hit Collection:
    if (PHC) {

        G4int NpHits = PHC->entries();
        strcpy(NameOfProcessInPhantom, "");

        for (G4int iPHit = 0; iPHit < NpHits; iPHit++) {
            GatePhantomHit *pHit = (*PHC)[iPHit];
            G4String processName = (*PHC)[iPHit]->GetProcess();

            if (pHit->GoodForAnalysis() && pHit->GetPDGEncoding() == -22)// looking at optical photons only
            {
                strcpy(NameOfProcessInPhantom, pHit->GetProcess().c_str());

                //                   if (processName.find("OpRayleigh") != G4String::npos)  nPhantomOpticalRayleigh++;
                //                   if (processName.find("OpticalMie") != G4String::npos)  nPhantomOpticalMie++;
                //                   if (processName.find("OpticalAbsorption") != G4String::npos) {
                //                          nPhantomOpticalAbsorption++;
                //                         PhantomAbsorbedPhotonHitPos_X = (*PHC)[iPHit]->GetPos().x();
                //                          PhantomAbsorbedPhotonHitPos_Y = (*PHC)[iPHit]->GetPos().y();
                //                          PhantomAbsorbedPhotonHitPos_Z = (*PHC)[iPHit]->GetPos().z();
                //                   }

                if (processName.find("OpticalWLS") != G4String::npos) {
                    nPhantomOpticalWLS++;      // Fluorescence counting
                    PhantomWLSPos_X = (*PHC)[iPHit]->GetPos().x();
                    PhantomWLSPos_Y = (*PHC)[iPHit]->GetPos().y();
                    PhantomWLSPos_Z = (*PHC)[iPHit]->GetPos().z();
                }

                PhantomLastHitPos_X = (*PHC)[iPHit]->GetPos().x();
                PhantomLastHitPos_Y = (*PHC)[iPHit]->GetPos().y();
                PhantomLastHitPos_Z = (*PHC)[iPHit]->GetPos().z();
                PhantomLastHitEnergy = (*PHC)[iPHit]->GetEdep();
            }  // end GoodForAnalysis() and optical photon
        } // end loop over phantom hits
    } // end if PHC


    // Looking at Crystal Hits Collection:
    if (CHC) {

        G4int NbHits = CHC->entries();
        strcpy(NameOfProcessInCrystal, "");

        for (G4int iHit = 0; iHit < NbHits; iHit++) {
            GateCrystalHit *aHit = (*CHC)[iHit];
            G4String processName = aHit->GetProcess();

            //              if (aHit->GoodForAnalysis() && aHit-> GetPDGEncoding()==0) // looking at optical photons only
            if (aHit->GoodForAnalysis()) {
                strcpy(NameOfProcessInCrystal, aHit->GetProcess().c_str());

                if (processName.find("Scintillation") != G4String::npos) nScintillation++;

                if (aHit->GetPDGEncoding() == -22)  // looking at optical photons only
                {
                    if (processName.find("OpticalWLS") != G4String::npos) nCrystalOpticalWLS++;
                    //               		if (processName.find("OpRayleigh") != G4String::npos)  nCrystalOpticalRayleigh++;
                    //               		if (processName.find("OpticalMie") != G4String::npos)  nCrystalOpticalMie++;
                    //               		if (processName.find("OpticalAbsorption") != G4String::npos) {
                    //                              nCrystalOpticalAbsorption++;
                    //                              CrystalAbsorbedPhotonHitPos_X = (*CHC)[iHit]->GetGlobalPos().x();
                    //                              CrystalAbsorbedPhotonHitPos_Y = (*CHC)[iHit]->GetGlobalPos().y();
                    //                              CrystalAbsorbedPhotonHitPos_Z = (*CHC)[iHit]->GetGlobalPos().z();
                    //                  	}

                    CrystalLastHitPos_X = (*CHC)[iHit]->GetGlobalPos().x();
                    CrystalLastHitPos_Y = (*CHC)[iHit]->GetGlobalPos().y();
                    CrystalLastHitPos_Z = (*CHC)[iHit]->GetGlobalPos().z();
                    CrystalLastHitEnergy = (*CHC)[iHit]->GetEdep();
                }
            } // end GoodForAnalysis()
        } // end loop over crystal hits
    } // end if CHC

    // counting the number of Wave Length Shifting = Fluorescence:
    if (nCrystalOpticalWLS > 0) NumCrystalWLS++;
    if (nPhantomOpticalWLS > 0) NumPhantomWLS++;

    if (m_rootOpticalFlag && trajectoryContainer) { OpticalTree->Fill(); }

}


//--------------------------------------------------------------------------
void GateToRoot::RecordDigitizer(const G4Event *) {
    if (nVerboseLevel > 2)
        G4cout << "GateToRoot::RecordDigitizer\n";

    // Digitizer information

    for (size_t i = 0; i < m_outputChannelList.size(); ++i)
        m_outputChannelList[i]->RecordDigitizer();

}
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
void GateToRoot::RecordStepWithVolume(const GateVVolume *, const G4Step *aStep) {

    // v. cuplov - optical photon momentum direction
    G4ParticleDefinition *partDef = aStep->GetTrack()->GetDefinition();
    if (partDef == G4OpticalPhoton::OpticalPhotonDefinition()) {
        G4ThreeVector momentumDirection = aStep->GetTrack()->GetMomentumDirection();
        MomentumDirectionx = momentumDirection.x();
        MomentumDirectiony = momentumDirection.y();
        MomentumDirectionz = momentumDirection.z();
    }
    // v. cuplov - optical photon momentum direction

    if (m_recordFlag > 0) {
        //GateMessage("OutputMgr", 5, " GateToRoot::RecordStep -- begin \n";);

        if (nVerboseLevel > 2)
            G4cout << "GateToRoot::RecordStep\n";

        G4ParticleDefinition *partDef = aStep->GetTrack()->GetDefinition();

        if (partDef == G4Positron::PositronDefinition()) {
            if (nVerboseLevel > 1) G4cout << "GateToRoot: Positron \n";
            G4String procName;
            const G4VProcess *process;
            process = aStep->GetPreStepPoint()->GetProcessDefinedStep();
            if (process != NULL) {
                procName = process->GetProcessName();
            } else {
                procName = "";
            }
            // to be changed with: s->GetTrack()->GetCurrentStepNumber() == 1 after some check
            if (procName == "") {
                m_positronKinEnergy = aStep->GetPreStepPoint()->GetKineticEnergy();
                m_positronGenerationPos = aStep->GetPreStepPoint()->GetPosition();
                if (nVerboseLevel > 1)
                    G4cout << "GateToRoot: Process empty."
                           << " m_positronKinEnergy " << m_positronKinEnergy << Gateendl;
            }
            procName = aStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
            if (procName == "annihil") {
                m_positronAnnihilPos = aStep->GetPostStepPoint()->GetPosition();
            }
        } else if (partDef != G4GenericIon::GenericIonDefinition()) {
            G4String procName;
            // could be a secondary ion in the radioactive chain
            if (aStep->GetTrack()->GetParentID() == 0) {
                const G4VProcess *process(aStep->GetPostStepPoint()->GetProcessDefinedStep()); // RTA
                if (process && process->GetProcessName() == "RadioactiveDecay")
                    m_ionDecayPos = aStep->GetPostStepPoint()->GetPosition(); // RTA
            }
        }

        // Calcul of momentum direction for each gamma  -> To plot the angle distribution -> BenchmarkPET

        G4ThreeVector momentumDirection = aStep->GetTrack()->GetVertexMomentumDirection();

        if (partDef == G4Gamma::GammaDefinition()) {

            // We don't take into account the bremstralhung production

            G4String procName;
            const G4VProcess *process;
            process = aStep->GetTrack()->GetCreatorProcess();
            if (process != NULL) {
                procName = process->GetProcessName();
            } else {
                procName = "";
            }
            if (aStep->GetTrack()->GetTrackID() == 2 && procName == "annihil") {

                dxg1 = momentumDirection.x();
                dyg1 = momentumDirection.y();
                dzg1 = momentumDirection.z();

            }

            if (aStep->GetTrack()->GetTrackID() == 3 && procName == "annihil") {
                dxg2 = momentumDirection.x();
                dyg2 = momentumDirection.y();
                dzg2 = momentumDirection.z();
            }
        }

        //GateMessage("OutputMgr", 5, " GateToRoot::RecordStep -- end \n";);
    }

}


//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRoot::Reset() {
    if (m_recordFlag > 0) {

        TH1F *hist;
        G4String hist_name;
        hist_name = "Positron_Kinetic_Energy_MeV";
        if ((hist = (TH1F *) m_working_root_directory->GetList()->FindObject(hist_name)) != NULL) {
            hist->Reset();
        }
        hist_name = "Ion_decay_time_s";
        if ((hist = (TH1F *) m_working_root_directory->GetList()->FindObject(hist_name)) != NULL) {
            hist->Reset();
        }
        hist_name = "Positron_annihil_distance_mm";
        if ((hist = (TH1F *) m_working_root_directory->GetList()->FindObject(hist_name)) != NULL) {
            hist->Reset();
        }
    }
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRoot::Store() {
    // store histograms
    // maybe move here the file.write

}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRoot::RecordVoxels(GateVGeometryVoxelStore *voxelStore) {
    if (m_recordFlag > 0) {

        if (nVerboseLevel > 2)
            G4cout << "GateToRoot::RecordVoxels\n";

        G4String voxelFileName = m_fileName + "Voxels.root";
        TFile *voxelsFile = new TFile(voxelFileName.c_str(), "RECREATE", "ROOT file with voxel info");


        TH3F *histVoxels;
        G4String hist_name;
        G4String hist_title;
        hist_name = "Voxel_density";
        hist_title = "Voxel density (g/cm3)";

        G4int nx = voxelStore->GetVoxelNx();
        G4int ny = voxelStore->GetVoxelNy();
        G4int nz = voxelStore->GetVoxelNz();
        histVoxels = new TH3F(hist_name, hist_title, nx, 0., float(nx), ny, 0., float(ny), nz, 0., float(nz));
        TH2F **histVoxelSlices = new TH2F *[nz];
        for (G4int iz = 0; iz < nz; iz++) {
            char chnum[20];
            sprintf(chnum, "%04d", iz);
            G4String snum(chnum);
            hist_name = "Voxel_density_slice_" + snum;
            hist_title = "Voxel density (g/cm3) slice " + snum;
            histVoxelSlices[iz] = new TH2F(hist_name, hist_title, nx, 0., float(nx), ny, 0., float(ny));
            for (G4int iy = 0; iy < ny; iy++) {
                for (G4int ix = 0; ix < nx; ix++) {
                    Float_t density = voxelStore->GetVoxelMaterial(ix, iy, iz)->GetDensity() / (gram / cm3);
                    //	G4cout << "Material: " << voxelStore->GetVoxelMaterial(ix,iy,iz)->GetName() << "  density: " << density << Gateendl;
                    histVoxels->Fill(ix, iy, iz, density);
                    histVoxelSlices[iz]->Fill(ix, iy, density);
                }
            }
        }

        voxelsFile->Write(0,TObject::kOverwrite);
        voxelsFile->Close();

        if (m_hfile) m_hfile->cd();
    }
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRoot::RegisterNewSingleDigiCollection(const G4String &aCollectionName, G4bool outputFlag) {
    SingleOutputChannel *singleOutputChannel =
            new SingleOutputChannel(aCollectionName, outputFlag);
    m_outputChannelList.push_back(singleOutputChannel);

    //G4cout << " GateToRoot::RegisterNewSingleDigiCollection outputFlag = " <<outputFlag<< Gateendl;
    m_rootMessenger->CreateNewOutputChannelCommand(singleOutputChannel);
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRoot::RegisterNewCoincidenceDigiCollection(const G4String &aCollectionName, G4bool outputFlag) {
    CoincidenceOutputChannel *coincidenceOutputChannel =
            new CoincidenceOutputChannel(aCollectionName, outputFlag);
    m_outputChannelList.push_back(coincidenceOutputChannel);
    m_rootMessenger->CreateNewOutputChannelCommand(coincidenceOutputChannel);
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRoot::SingleOutputChannel::RecordDigitizer() {
    G4DigiManager *fDM = G4DigiManager::GetDMpointer();

    if (m_collectionID < 0)
        m_collectionID = fDM->GetDigiCollectionID(m_collectionName);
    const GateSingleDigiCollection *SDC =
            (GateSingleDigiCollection *) (fDM->GetDigiCollection(m_collectionID));

    if (!SDC) {
        //GateMessage("OutputMgr", 5, " There is no SDC collection\n";);
        if (nVerboseLevel > 0)
            G4cout << "[GateToRoot::SingleOutputChannel::RecordDigitizer]:"
                   << " digi collection '" << m_collectionName << "' not found\n";
    } else {
        // Digi loop
        //GateMessage("OutputMgr", 5, " There is SDC collection. \n";);
        if (nVerboseLevel > 0)
            G4cout << "[GateToRoot::SingleOutputChannel::RecordDigitizer]: Total Digits: "
                   << SDC->entries() << Gateendl;

        //  GateMessage("OutputMgr", 5, " Single collection m_outputFlag = " << m_outputFlag << Gateendl;);

        if (m_outputFlag) {
            G4int n_digi = SDC->entries();
            //GateMessage("OutputMgr", 5, " Single collection m_outputFlag = " << m_outputFlag << Gateendl;);
            for (G4int iDigi = 0; iDigi < n_digi; iDigi++) {
                m_buffer.Fill((*SDC)[iDigi]);
                m_tree->Fill();
            }
        }
    }
}
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
void GateToRoot::CoincidenceOutputChannel::RecordDigitizer() {
    //GateMessage("OutputMgr", 5, " GateToRoot::CoincidenceOutputChannel::RecordDigitizer -- begin\n";);
    G4DigiManager *fDM = G4DigiManager::GetDMpointer();
    if (m_collectionID < 0)
        m_collectionID = fDM->GetDigiCollectionID(m_collectionName);
    GateCoincidenceDigiCollection *CDC =
            (GateCoincidenceDigiCollection *) (fDM->GetDigiCollection(m_collectionID));

    if (!CDC) {
        //GateMessage("OutputMgr", 5, " There is no CDC collection.\n";);
        if (nVerboseLevel > 0)
            G4cout << "[GateToRoot::CoincidenceOutputChannel::RecordDigitizer]:"
                   << " digi collection '" << m_collectionName << "' not found\n";
    } else {

        //GateMessage("OutputMgr", 5, " There is CDC collection. \n";);
        // Digi loop
        if (nVerboseLevel > 0)
            G4cout << "[GateToRoot::CoincidenceOutputChannel::RecordDigitizer]: Total Digits: "
                   << CDC->entries() << Gateendl;

        //GateMessage("OutputMgr", 5, " Coincidence collection m_outputFlag = " << m_outputFlag << Gateendl;);

        if (m_outputFlag) {
            G4int n_digi = CDC->entries();
            for (G4int iDigi = 0; iDigi < n_digi; iDigi++) {
                m_buffer.Fill((*CDC)[iDigi]);
                m_tree->Fill();
            }
        }
    }

    //GateMessage("OutputMgr", 5, " GateToRoot::CoincidenceOutputChannel::RecordDigitizer -- end\n";);
}

void GateToRoot::CloseTracksRootFile() {
    if (m_TracksFile != 0) {
        m_RecStepTree->GetEntry(m_currentRSData);
        last_RSEventID = m_RSEventID;
        if (nVerboseLevel > 3) {
            G4cout << "GateToRoot::CloseTracksRootFile() ::: Closing Tracks Data Root File " << fTracksFN << " ... ";
            G4cout << " GateToRoot::CloseTracksRootFile() --- m_currentRSData = " << m_currentRSData
                   << "   m_RecStepTree->GetEntries() = " << m_RecStepTree->GetEntries() << Gateendl;
            G4cout << " GateToRoot::CloseTracksRootFile() --- last_RSEventID = " << last_RSEventID << Gateendl;
        }
        dxg1_copy = dxg1;
        dyg1_copy = dyg1;
        dzg1_copy = dzg1;
        dxg2_copy = dxg2;
        dyg2_copy = dyg2;
        dzg2_copy = dzg2;

        m_positronKinEnergy_copy = m_positronKinEnergy;
        m_ionDecayPos_copy = m_ionDecayPos;
        m_positronGenerationPos_copy = m_positronGenerationPos;
        m_positronAnnihilPos_copy = m_positronAnnihilPos;
        theCRData_copy = theCRData;
        //G4cout << " GateToRoot::CloseTracksRootFile()\n";
        //PrintRecStep();

        if (m_TracksFile->IsOpen()) { m_TracksFile->Close(); }
    }
    m_EOF = 1;
    m_TracksFile = 0;
    if (nVerboseLevel > 3)G4cout << "done\n";
}

G4int GateToRoot::CheckEOF() {
    return m_EOF;
}

void GateToRoot::PrintRecStep() {
    G4cout << "Ion Decay Position = " << m_ionDecayPos << Gateendl;
    G4cout << "positron Generation Position = " << m_positronGenerationPos << Gateendl;
    G4cout << "positron Annihilation Position = " << m_positronAnnihilPos << Gateendl;
    G4cout << "positron Kinetic Energy        = " << m_positronKinEnergy << Gateendl;
    G4cout << "dxg1 = " << dxg1 << Gateendl;
    G4cout << "dyg1 = " << dyg1 << Gateendl;
    G4cout << "dzg1 = " << dzg1 << Gateendl;
    G4cout << "dxg2 = " << dxg2 << Gateendl;
    G4cout << "dyg2 = " << dyg2 << Gateendl;
    G4cout << "dzg2 = " << dzg2 << Gateendl;
    G4cout << "photon1_phantom_Rayleigh = " << theCRData.photon2_phantom_Rayleigh << Gateendl;
    G4cout << "photon2_phantom_Rayleigh = " << theCRData.photon2_phantom_Rayleigh << Gateendl;
    G4cout << "photon1_phantom_compton = " << theCRData.photon1_phantom_compton << Gateendl;
    G4cout << "photon2_phantom_compton  = " << theCRData.photon2_phantom_compton << Gateendl;
    G4cout << "theComptonVolumeName1   = " << theCRData.theComptonVolumeName1 << Gateendl;
    G4cout << "theComptonVolumeName2   = " << theCRData.theComptonVolumeName2 << Gateendl;
    G4cout << "theRayleighVolumeName1   = " << theCRData.theRayleighVolumeName1 << Gateendl;
    G4cout << "theRayleighVolumeName2   = " << theCRData.theRayleighVolumeName2 << Gateendl;
    G4cout << "--------------------- COPY------------------------------\n";
    G4cout << "Ion Decay Position = " << m_ionDecayPos_copy << Gateendl;
    G4cout << "positron Generation Position = " << m_positronGenerationPos_copy << Gateendl;
    G4cout << "positron Annihilation Position = " << m_positronAnnihilPos_copy << Gateendl;
    G4cout << "positron Kinetic Energy        = " << m_positronKinEnergy_copy << Gateendl;
    G4cout << "dxg1 = " << dxg1_copy << Gateendl;
    G4cout << "dyg1 = " << dyg1_copy << Gateendl;
    G4cout << "dzg1 = " << dzg1_copy << Gateendl;
    G4cout << "dxg2 = " << dxg2_copy << Gateendl;
    G4cout << "dyg2 = " << dyg2_copy << Gateendl;
    G4cout << "dzg2 = " << dzg2_copy << Gateendl;
    G4cout << "photon1_phantom_Rayleigh = " << theCRData_copy.photon2_phantom_Rayleigh << Gateendl;
    G4cout << "photon2_phantom_Rayleigh = " << theCRData_copy.photon2_phantom_Rayleigh << Gateendl;
    G4cout << "photon1_phantom_compton = " << theCRData_copy.photon1_phantom_compton << Gateendl;
    G4cout << "photon2_phantom_compton  = " << theCRData_copy.photon2_phantom_compton << Gateendl;
    G4cout << "theComptonVolumeName1   = " << theCRData_copy.theComptonVolumeName1 << Gateendl;
    G4cout << "theComptonVolumeName2   = " << theCRData_copy.theComptonVolumeName2 << Gateendl;
    G4cout << "theRayleighVolumeName1   = " << theCRData_copy.theRayleighVolumeName1 << Gateendl;
    G4cout << "theRayleighVolumeName2   = " << theCRData_copy.theRayleighVolumeName2 << Gateendl;
}

/// OPEN ROOT TRACKS DATA FILE IN READ MODE
void GateToRoot::OpenTracksFile() {

    G4int lastEventID = -1;
    G4String previousFN = fTracksFN;


    GateSteppingAction *myAction = ((GateSteppingAction *) (GateRunManager::GetRunManager()->GetUserSteppingAction()));

    G4int currentN = myAction->GetcurrentN();


    G4int NbOfFiles = myAction->GetNfiles();

    //if ( m_verboseLevel > 3 )
    G4cout << (NbOfFiles - currentN) << " File(s) remain to be opened in Detector Mode :\n";

    if (currentN == 0) {
        fTracksFN = m_fileName + "_TrackerData.root";
    }

    G4String aFile;
    for (int i = currentN; i < NbOfFiles; i++) {
        if (i == 0) { aFile = m_fileName + "_TrackerData.root"; }
        else {
            std::stringstream s;
            s << i;
            aFile = m_fileName + "_TrackerData_" + s.str() + ".root";
        }
        //if ( m_verboseLevel > 3 )
        G4cout << "---- " << aFile << Gateendl;
    }

    if (currentN > 0) {
        std::stringstream s; // convert currentN into string
        s << currentN;
        fTracksFN = m_fileName + "_TrackerData_" + s.str() + ".root";
        lastEventID = EventID;
        if (nVerboseLevel > 3) G4cout << "  LAST EVENT ID " << lastEventID << Gateendl;
        CloseTracksRootFile();
    }

    if (m_TracksFile != 0) { delete m_TracksFile; }

    m_TracksFile = new TFile(fTracksFN.c_str(), "READ", "ROOT file with Tracker Data");

    //if ( m_verboseLevel > 3 )
    G4cout << "GateToRoot::OpenTracksFile() :::: Opening Tracks Data Root File " << fTracksFN << Gateendl;

    // Check that we succeeded in opening the file

    if (!m_TracksFile) {
        G4String msg = "Could not instantiate the pointer to  output ROOT file '" + fTracksFN + "'";
        G4Exception("GateToRoot::OpenTracksFile", "OpenTracksFile", FatalException, msg);
    }
    if (!(m_TracksFile->IsOpen())) {
        G4String msg = "Could not instantiate the pointer to  output ROOT file '" + fTracksFN + "'";
        G4Exception("GateToRoot::OpenTracksFile", "OpenTracksFile", FatalException, msg);
    }
    tracksTuple = (TTree *) (m_TracksFile->Get(G4String("PhTracksData").c_str()));

    //       tracksTuple->Print();

    tracksTuple->SetBranchAddress(G4String("RunID").c_str(), &RunID);
    tracksTuple->SetBranchAddress(G4String("TrackID").c_str(), &TrackID);
    tracksTuple->SetBranchAddress(G4String("ParentID").c_str(), &ParentID);
    tracksTuple->SetBranchAddress(G4String("Pos_x").c_str(), &posx);
    tracksTuple->SetBranchAddress(G4String("Pos_y").c_str(), &posy);
    tracksTuple->SetBranchAddress(G4String("Pos_z").c_str(), &posz);
    tracksTuple->SetBranchAddress(G4String("LTime").c_str(), &LTime);
    tracksTuple->SetBranchAddress(G4String("GTime").c_str(), &GTime);
    tracksTuple->SetBranchAddress(G4String("PTime").c_str(), &PTime);
    tracksTuple->SetBranchAddress(G4String("MDirection_x").c_str(), &MDirectionx);
    tracksTuple->SetBranchAddress(G4String("MDirection_y").c_str(), &MDirectiony);
    tracksTuple->SetBranchAddress(G4String("MDirection_z").c_str(), &MDirectionz);
    tracksTuple->SetBranchAddress(G4String("Momentum_x").c_str(), &Momentumx);
    tracksTuple->SetBranchAddress(G4String("Momentum_y").c_str(), &Momentumy);
    tracksTuple->SetBranchAddress(G4String("Momentum_z").c_str(), &Momentumz);
    tracksTuple->SetBranchAddress(G4String("Energy").c_str(), &Energy);
    tracksTuple->SetBranchAddress(G4String("Wavelength").c_str(), &Wavelength);  // v. cuplov wavelength
    tracksTuple->SetBranchAddress(G4String("Kinenergy").c_str(), &KinEnergy);
    tracksTuple->SetBranchAddress(G4String("Velocity").c_str(), &Velocity);
    tracksTuple->SetBranchAddress(G4String("Vertexposition_x").c_str(), &VertexPositionx);
    tracksTuple->SetBranchAddress(G4String("Vertexposition_y").c_str(), &VertexPositiony);
    tracksTuple->SetBranchAddress(G4String("Vertexposition_z").c_str(), &VertexPositionz);
    tracksTuple->SetBranchAddress(G4String("Vertexmomentumdirection_x").c_str(), &VtxMomDirx);
    tracksTuple->SetBranchAddress(G4String("Vertexmomentumdirection_y").c_str(), &VtxMomDiry);
    tracksTuple->SetBranchAddress(G4String("Vertexmomentumdirection_z").c_str(), &VtxMomDirz);
    tracksTuple->SetBranchAddress(G4String("VertexKineticEnergy").c_str(), &VertexKineticEnergy);
    tracksTuple->SetBranchAddress(G4String("Polarization_x").c_str(), &Polarizationx);
    tracksTuple->SetBranchAddress(G4String("Polarization_y").c_str(), &Polarizationy);
    tracksTuple->SetBranchAddress(G4String("Polarization_z").c_str(), &Polarizationz);
    tracksTuple->SetBranchAddress(G4String("Weight").c_str(), &Weight);
    tracksTuple->SetBranchAddress(G4String("EventID").c_str(), &EventID);
    tracksTuple->SetBranchAddress(G4String("EventTime").c_str(), &m_EventTime);
    tracksTuple->SetBranchAddress(G4String("PDGCode").c_str(), &PDGCode);
    tracksTuple->SetBranchAddress(G4String("SourceID").c_str(), &m_sourceID);
    tracksTuple->SetBranchAddress(G4String("WasKilled").c_str(), &m_wasKilled);
    tracksTuple->SetBranchAddress(G4String("ProcessName").c_str(), &m_processName);
    tracksTuple->SetBranchAddress(G4String("PPName").c_str(), &m_parentparticleName);
    tracksTuple->SetBranchAddress(G4String("LogAtVertex").c_str(), &m_volumeName);

    if (currentN > 0) {
        // store the content of the RecStep data

        /// check that the first event ID entry is the same as the one of the preceding file
        // if not we should close this file and open it next time we get here
        tracksTuple->GetEntry(0);
        if (EventID != lastEventID) { // if ( m_verboseLevel > 3 ) {
            G4cout << "  GateToRoot::OpenTracksFile()  ::  last Event ID was " << lastEventID
                   << "  ---  current one read from last Tracks Root File " << fTracksFN << " is " << EventID
                   << Gateendl;
            G4cout << " SAVING RecStep Data to be used in GateToRoot::RecordBeginOfEvent \n";
            G4cout << " GateToRoot::OpenTracksFile()  ::  LAST event ID read from RecStep Data in file " << previousFN
                   << " is " << m_RSEventID << Gateendl;
            G4cout << " GateToRoot::OpenTracksFile()  ::  LAST Event ID read from Tracks Data in previous file "
                   << previousFN << " is " << lastEventID << Gateendl;
            //}
            if (lastEventID == last_RSEventID)fSkipRecStepData = 1;
        }
    }

    m_RecStepTree = (TTree *) (m_TracksFile->Get(G4String("RecStepData").c_str()));
    m_RecStepTree->SetBranchAddress(G4String("EventID").c_str(), &m_RSEventID);
    m_RecStepTree->SetBranchAddress(G4String("IonDecayPos").c_str(), &m_ionDecayPos);
    m_RecStepTree->SetBranchAddress(G4String("PositronGenerationPos").c_str(), &m_positronGenerationPos);
    m_RecStepTree->SetBranchAddress(G4String("PositronAnnihilPos").c_str(), &m_positronAnnihilPos);
    m_RecStepTree->SetBranchAddress(G4String("PositronKinEnergy").c_str(), &m_positronKinEnergy);
    m_RecStepTree->SetBranchAddress(G4String("dxg1").c_str(), &dxg1);
    m_RecStepTree->SetBranchAddress(G4String("dyg1").c_str(), &dyg1);
    m_RecStepTree->SetBranchAddress(G4String("dzg1").c_str(), &dzg1);
    m_RecStepTree->SetBranchAddress(G4String("dxg2").c_str(), &dxg2);
    m_RecStepTree->SetBranchAddress(G4String("dyg2").c_str(), &dyg2);
    m_RecStepTree->SetBranchAddress(G4String("dzg2").c_str(), &dzg2);
    m_RecStepTree->SetBranchAddress(G4String("photon1PhR").c_str(), &theCRData.photon1_phantom_Rayleigh);
    m_RecStepTree->SetBranchAddress(G4String("photon2PhR").c_str(), &theCRData.photon2_phantom_Rayleigh);
    m_RecStepTree->SetBranchAddress(G4String("photon1PhC").c_str(), &theCRData.photon1_phantom_compton);
    m_RecStepTree->SetBranchAddress(G4String("photon2PhC").c_str(), &theCRData.photon2_phantom_compton);
    m_RecStepTree->SetBranchAddress(G4String("ComptVol1").c_str(), &theCRData.theComptonVolumeName1);
    m_RecStepTree->SetBranchAddress(G4String("ComptVol2").c_str(), &theCRData.theComptonVolumeName2);
    m_RecStepTree->SetBranchAddress(G4String("RaylVol1").c_str(), &theCRData.theRayleighVolumeName1);
    m_RecStepTree->SetBranchAddress(G4String("RaylVol2").c_str(), &theCRData.theRayleighVolumeName2);
    m_RecStepTree->SetBranchAddress(G4String("RunID").c_str(), &m_RSRunID);

    //  rewind counters

    m_EOF = 0;
    m_currentTracksData = 0;
    m_currentRSData = 0;

}


void GateToRoot::RecordPHData(ComptonRayleighData aCRData) {
    theCRData.photon1_phantom_Rayleigh = aCRData.photon1_phantom_Rayleigh;
    theCRData.photon2_phantom_Rayleigh = aCRData.photon1_phantom_Rayleigh;
    theCRData.photon1_phantom_compton = aCRData.photon1_phantom_compton;
    theCRData.photon2_phantom_compton = aCRData.photon2_phantom_compton;
    strcpy(theCRData.theComptonVolumeName1, aCRData.theComptonVolumeName1);
    strcpy(theCRData.theComptonVolumeName2, aCRData.theComptonVolumeName2);
    strcpy(theCRData.theRayleighVolumeName1, aCRData.theRayleighVolumeName1);
    strcpy(theCRData.theRayleighVolumeName2, aCRData.theRayleighVolumeName2);
}

void GateToRoot::GetPHData(ComptonRayleighData &aCRData) {
    aCRData.photon1_phantom_Rayleigh = theCRData.photon1_phantom_Rayleigh;
    aCRData.photon2_phantom_Rayleigh = theCRData.photon1_phantom_Rayleigh;
    aCRData.photon1_phantom_compton = theCRData.photon1_phantom_compton;
    aCRData.photon2_phantom_compton = theCRData.photon2_phantom_compton;
    strcpy(aCRData.theComptonVolumeName1, theCRData.theComptonVolumeName1);
    strcpy(aCRData.theComptonVolumeName2, theCRData.theComptonVolumeName2);
    strcpy(aCRData.theRayleighVolumeName1, theCRData.theRayleighVolumeName1);
    strcpy(aCRData.theRayleighVolumeName2, theCRData.theRayleighVolumeName2);

}

void GateToRoot::GetCurrentRecStepData(const G4Event *evt) {
    if (m_currentRSData == m_RecStepTree->GetEntries()) return;
    m_RecStepTree->GetEntry(m_currentRSData);

    //G4cout << " GateToRoot::GetCurrentRecStepData \n";
    //PrintRecStep();

    if (m_RSEventID != evt->GetEventID()) {
        const G4Run *currentRun = GateRunManager::GetRunManager()->GetCurrentRun();
        G4int RunID = currentRun->GetRunID();
        G4cout << " GateToRoot::GetCurrentRecStepData :::: current Run ID " << RunID << "    current RecStep File "
               << m_RecStepTree->GetCurrentFile()->GetName() << Gateendl;
        G4cout << " GateToRoot::GetCurrentRecStepData :::: m_currentTracksData = " << m_currentTracksData
               << "     tracksTuple->GetEntries()   " << tracksTuple->GetEntries() << Gateendl;
        G4cout << " GateToRoot::GetCurrentRecStepData :::: current event ID read from RecStep File " << m_RSEventID
               << "     current event ID " << evt->GetEventID() << Gateendl;
        G4cout << " GateToRoot::GetCurrentRecStepData :::: m_currentRSData = " << m_currentRSData
               << "    m_RecStepTree->GetEntries()  " << m_RecStepTree->GetEntries() << Gateendl;
        G4Exception("GateToRoot::GetCurrentRecStepData", "GetCurrentRecStepData", FatalException, "Aborting ...");
    }
    //G4cout << " GateToRoot::GetCurrentRecStepData :::: m_currentRSData = "<<m_currentRSData<<"    m_RecStepTree->GetEntries()  "<<m_RecStepTree->GetEntries()<< Gateendl;
    m_currentRSData++;
}


GateTrack *GateToRoot::GetCurrentTracksData() {
    GateSteppingAction *myAction = ((GateSteppingAction *) (GateRunManager::GetRunManager()->GetUserSteppingAction()));
    if (m_currentTracksData == tracksTuple->GetEntries()) // check if we are done
    {
        m_EOF = 1;
        G4int GoOn = myAction->SeekNewFile(true);
        if (GoOn == 0) { return 0; }
    }
    if (myAction->NoMoreTracksData() == 1 && m_EOF == 1) { return 0; }

    G4int nbytes = tracksTuple->GetEntry(m_currentTracksData);
    if (nVerboseLevel > 1) {
        G4cout << "GateToRoot::GetCurrentTracksData() :::  Reading buffer of size " << nbytes << " bytes in "
               << m_fileName + "_TrackerData.root\n";
        G4cout << "RunID " << RunID << "   EventID " << EventID << "         Position "
               << G4ThreeVector(posx, posy, posz) << Gateendl;
    }
    if (m_currentGTrack != 0) {
        G4ThreeVector tmp;
        m_currentGTrack->SetRunID(RunID);
        m_currentGTrack->SetTrackID(TrackID);
        m_currentGTrack->SetParentID(ParentID);
        m_currentGTrack->SetPosition(G4ThreeVector(posx, posy, posz));
        m_currentGTrack->SetLocalTime(LTime);
        m_currentGTrack->SetGlobalTime(GTime);
        m_currentGTrack->SetProperTime(PTime);
        m_currentGTrack->SetMomentumDirection(G4ThreeVector(MDirectionx, MDirectiony, MDirectionz));
        m_currentGTrack->SetMomentum(G4ThreeVector(Momentumx, Momentumy, Momentumz));
        m_currentGTrack->SetTotalEnergy(Energy);
        m_currentGTrack->SetKineticEnergy(KinEnergy);
        m_currentGTrack->SetVelocity(Velocity);
        tmp = G4ThreeVector(VertexPositionx, VertexPositiony, VertexPositionz);
        m_currentGTrack->SetVertexPosition(tmp);
        m_currentGTrack->SetVertexMomentumDirection(G4ThreeVector(VtxMomDirx, VtxMomDiry, VtxMomDirz));
        m_currentGTrack->SetVertexKineticEnergy(VertexKineticEnergy);
        m_currentGTrack->SetPolarization(G4ThreeVector(Polarizationx, Polarizationy, Polarizationz));
        m_currentGTrack->SetWeight(Weight);
        m_currentGTrack->SetEventID(EventID);
        m_currentGTrack->SetWasKilled(m_wasKilled);

        m_currentGTrack->SetPDGCode(PDGCode);
        m_currentGTrack->SetSourceID(m_sourceID);

        m_currentGTrack->SetProcessName(m_processName); // name of the process
        m_currentGTrack->SetParentParticleName(
                m_parentparticleName); // name of the particle which created this particle

        m_currentGTrack->SetVertexVolumeName(m_volumeName);
        m_currentGTrack->SetTime(m_EventTime);
        G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
        G4ParticleDefinition *pd = particleTable->FindParticle(PDGCode);
        if (pd == NULL) {
            G4Exception("GateToRoot:: GetCurrentTracksData", "GetCurrentTracksData", FatalException,
                        "ERROR PDGCode of the particle  is not defined. \n");
        }
        m_particleName = (G4String) (pd->GetParticleName());
        m_currentGTrack->SetParticleName(m_particleName);
    }
    if (nVerboseLevel > 1) {
        G4cout << " RETRIEVING Current Gate Track Informations : \n";
        m_currentGTrack->Print();
    }
    return m_currentGTrack;
}

void GateToRoot::RecordRecStepData(const G4Event *evt) {
    //G4cout << " GateToRoot::RecordRecStepData : recording RecStep Data to ROOT file \n";
    m_RSEventID = evt->GetEventID();
    m_RSRunID = GateRunManager::GetRunManager()->GetCurrentRun()->GetRunID();
    m_RecStepTree->Fill();
    //PrintRecStep();
    //G4cout << " GateToRoot::RecordRecStepData : runID " << m_RSRunID << "  eventID "<< m_RSEventID  << Gateendl;
    //G4cout << " GateToRoot::RecordRecStepData : total " << m_RecStepTree->GetEntries() << Gateendl;
    //G4cout << " GateToRoot::RecordRecStepData : time  " << (GateSourceMgr::GetInstance()->GetTime())/s << Gateendl;

}

void GateToRoot::RecordTracks(GateSteppingAction *mySteppingAction) {

    std::vector<GateTrack *> *PPTrackVector = mySteppingAction->GetPPTrackVector();

    if (mySteppingAction->GetTxtOn() == 1) {
        std::ofstream outFile;
        outFile.open("PostStepInfo.txt", std::ios::app);
        outFile << " number of Primary particles " << PPTrackVector->size() << "\n";
    }

    std::vector<GateTrack *>::iterator iter;

    for (iter = PPTrackVector->begin(); iter != PPTrackVector->end(); iter++) {

        if (nVerboseLevel > 0) {
            G4cout << " Writing to Root File \n";
            (*iter)->Print();
        }

        RunID = (*iter)->GetRunID();
        EventID = (*iter)->GetEventID();
        TrackID = (*iter)->GetTrackID();
        ParentID = (*iter)->GetParentID();
        posx = (*iter)->GetPosition().x();
        posy = (*iter)->GetPosition().y();
        posz = (*iter)->GetPosition().z();
        LTime = (*iter)->GetLocalTime();
        GTime = (*iter)->GetGlobalTime();

        // get current time from of the current event

        Polarizationx = (*iter)->GetPolarization().x();
        Polarizationy = (*iter)->GetPolarization().y();
        Polarizationz = (*iter)->GetPolarization().z();

        PTime = (*iter)->GetProperTime();
        MDirectionx = (*iter)->GetMomentumDirection().x();
        MDirectiony = (*iter)->GetMomentumDirection().y();
        MDirectionz = (*iter)->GetMomentumDirection().z();

        Momentumx = (*iter)->GetMomentum().x();
        Momentumy = (*iter)->GetMomentum().y();
        Momentumz = (*iter)->GetMomentum().z();
        Energy = (*iter)->GetTotalEnergy();

        // v. cuplov wavelength
        if (Energy != 0.) {
            //			Wavelength =  1239.61/Energy ;
            Wavelength = c_light * h_Planck / Energy / nanometer; //==> gives nm
        } else { Wavelength = -1.; }

        KinEnergy = (*iter)->GetKineticEnergy();
        Velocity = (*iter)->GetVelocity();

        VertexPositionx = (*iter)->GetVertexPosition().x();
        VertexPositiony = (*iter)->GetVertexPosition().y();
        VertexPositionz = (*iter)->GetVertexPosition().z();
        VtxMomDirx = (*iter)->GetVertexMomentumDirection().x();
        VtxMomDiry = (*iter)->GetVertexMomentumDirection().y();
        VtxMomDirz = (*iter)->GetVertexMomentumDirection().z();
        VertexKineticEnergy = (*iter)->GetVertexKineticEnergy();
        PDGCode = (*iter)->GetPDGCode();
        m_sourceID = (*iter)->GetSourceID();

        m_wasKilled = (*iter)->GetWasKilled();

        strcpy(m_processName, (*iter)->GetProcessName().c_str());
        strcpy(m_parentparticleName, (*iter)->GetParentParticleName().c_str());
        strcpy(m_volumeName, (*iter)->GetVertexVolumeName().c_str());
        //G4cout << "GateToRoot::RecordTracks ::: process Name " << m_processName << Gateendl;


        //                G4String aString;
        //                aString          = (*iter)->GetVolAtVertex();
        //		m_volumeName  = (char* ) (aString.c_str() );
        ////
        // write data to Root File
        //

        m_EventTime = (*iter)->GetTime();

        //G4cout  << "GateToRoot::RecordTracks event Time "  << m_EventTime/s <<   "     EventID " <<  EventID << Gateendl;
        //G4cout  << "GateToRoot::RecordTracks source ID " << m_sourceID << Gateendl;

        G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();

        G4ParticleDefinition *pd = particleTable->FindParticle(PDGCode);

        if (pd == NULL) {
            if (PDGCode == 0) m_particleName = G4String("GenericIon");
            else
                G4Exception("GateToRoot::RecordTracks", "RecordTracks", FatalException,
                            "ERROR PDGCode of the particle  is not defined. \n");
        } else m_particleName = (G4String) (pd->GetParticleName());

        tracksTuple->Fill();
        delete (*iter);

        //G4cout << " GateToRoot::RecordTracks particle name " << m_particleName  << "   parent particle name " << m_parentparticleName<< Gateendl;


        if (mySteppingAction->GetTxtOn() == 1) {
            std::ofstream outFile;
            outFile.open("PostStepInfo.txt", std::ios::app);
            outFile << " Event ID " << EventID << "    Particle " << m_particleName << "\n";
            outFile << "Position " << G4ThreeVector(posx, posy, posz) << "    Track     ID " << TrackID
                    << "      Parent ID " << ParentID << "\n";
            outFile << " Local Time " << LTime << "  Global Time  " << GTime << "      Proper Time " << PTime <<
                    "      Momentum Direction " << G4ThreeVector(MDirectionx, MDirectiony, MDirectionz)
                    << "      Momentum " << G4ThreeVector(Momentumx, Momentumy, Momentumz) << "\n";
            outFile << "Energy " << Energy << "              Kinetic Energy " << KinEnergy
                    << "                       Velocity " << Velocity << "\n";
            outFile << " ========================================================\n";
        }

    }

    PPTrackVector->clear();

}

/*PY Descourt 08/09/2009 */

//--------------------------------------------------------------------------
#endif
