/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*  Optical Photons: V. Cuplov -  2012
         - New function RecordOpticalData(event).
         - New ntuple for optical photon data is defined in GateToRoot class (previously was in GateFastAnalysis)
         - Revision v6.2   2012/07/09  by vesna.cuplov@gmail.com
           output ROOT file is dedicated to optical photons
         - Revision v6.2   2012/07/24  by vesna.cuplov@gmail.com
           Unique output file with Gate default trees (Hits,Singles,Coincidences...) + OpticalData Tree.
         - Revision v6.2 2012/08/06  Added optical photon momentum direction (x,y,z) in tree.
         - Revision 2012/09/17  /gate/output/root/setRootOpticalFlag functionality added.
           Set the flag for Optical ROOT output.
         - Revision 2012/11/14  - added new leaves: position (x,y,z) of fluorescent (OpticalWLS process) hits
 				- Scintillation counter bug-fixed
*/

#ifndef GateToRoot_H
#define GateToRoot_H

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TBranch.h"

#include "globals.hh"
#include <fstream>

#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"

#include "GateRootDefs.hh"
#include "GateVOutputModule.hh"

//OK GND 2022
#include "GateDigitizerMgr.hh"

/* PY Descourt 08/09/2009 */
#include "GateActions.hh"
#include "GateTrack.hh"

class GateToRootMessenger;

class GateVVolume;

// v. cuplov - optical photons
class GateTrajectoryNavigator;
// v. cuplov - optical photons

class ComptonRayleighData {
public:
    G4int photon1_phantom_Rayleigh, photon2_phantom_Rayleigh;
    G4int photon1_phantom_compton, photon2_phantom_compton;
    Char_t theComptonVolumeName1[60], theComptonVolumeName2[60], theRayleighVolumeName1[60], theRayleighVolumeName2[60];

    ComptonRayleighData(ComptonRayleighData &);

    ComptonRayleighData &operator=(const ComptonRayleighData &);

    ComptonRayleighData();
};

//--------------------------------------------------------------------------
class GateToRoot : public GateVOutputModule {
public:

    GateToRoot(const G4String &name, GateOutputMgr *outputMgr, DigiMode digiMode);

    virtual ~GateToRoot();

    const G4String &GiveNameOfFile();

    void RecordBeginOfAcquisition();

    void RecordEndOfAcquisition();

    void RecordBeginOfRun(const G4Run *);

    void RecordEndOfRun(const G4Run *);

    void RecordBeginOfEvent(const G4Event *);

    void RecordEndOfEvent(const G4Event *);

    void RecordStepWithVolume(const GateVVolume *v, const G4Step *);

    //! saves the geometry voxel information
    void RecordVoxels(GateVGeometryVoxelStore *);

    void RecordDigitizer(const G4Event *);

// v. cuplov - optical photons
    void RecordOpticalData(const G4Event *event);
// v. cuplov - optical photons

    void RecordVoxels(const G4Step *);

    void BookBeginOfAquisition();

    void BookBeginOfRun();

    void Store();

    void Reset();

    G4int GetRecordFlag() { return m_recordFlag; };

    void SetRecordFlag(G4int flag) { m_recordFlag = flag; };

    /* PY Descourt 08/09/2009 */
    void OpenTracksFile();

    //! saves the geometry voxel information
    void RecordTracks(GateSteppingAction *);

    void RecordRecStepData(const G4Event *evt);

    G4int CheckEOF();

    GateTrack *GetCurrentTracksData();

    void GetCurrentRecStepData(const G4Event *);

    G4int GetHeadNo() { return m_currentTracksData; };

    void ReadForward() { m_currentTracksData++; };

    void ReadBackward() {
        if (tracksTuple != 0) {
            if (m_currentTracksData > 0) { m_currentTracksData--; }
        }
    };

    void CloseTracksRootFile();

    void RecordPHData(ComptonRayleighData aCRData);

    void GetPHData(ComptonRayleighData &aCRData);
    /* PY Descourt 08/09/2009 */

    //--------------------------------------------------------------------------
    class VOutputChannel {
    public:
        inline VOutputChannel(const G4String &aCollectionName, G4bool outputFlag, G4bool CCFlag)
                : nVerboseLevel(0),
                  m_outputFlag(outputFlag),
				  m_CCFlag(CCFlag),
                  m_collectionName(aCollectionName),
                  m_collectionID(-1),
				  m_signlesCommands(0){}

        virtual inline ~VOutputChannel() {}

        virtual void Clear() = 0;

        virtual void RecordDigitizer() = 0;

        virtual void Book() = 0;

        inline void SetOutputFlag(G4bool flag) { m_outputFlag = flag; };

        inline void SetCCFlag(G4bool val){m_CCFlag=val;};
        inline G4bool GetCCFlag(){return m_CCFlag;};


        inline void AddSinglesCommand() { m_signlesCommands++; };


        inline void SetVerboseLevel(G4int val) { nVerboseLevel = val; };

        G4int nVerboseLevel;
        G4bool m_outputFlag;
        G4bool m_CCFlag;

        G4String m_collectionName;
        G4int m_collectionID;
        G4int m_signlesCommands;

    };


    //--------------------------------------------------------------------------
    class SingleOutputChannel : public VOutputChannel {
    public:
        inline SingleOutputChannel(const G4String &aCollectionName, G4bool outputFlag)
                : VOutputChannel(aCollectionName, outputFlag, false),
                  m_tree(0)
        		{ m_buffer.Clear();     			}

        virtual inline ~SingleOutputChannel() {}

        inline void Clear() { m_buffer.Clear(); }

        inline void Book() {
            m_collectionID = -1;
            //OK GND 2022 multiSD backward compatibility
            GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

            if (m_outputFlag) {
            	if( digitizerMgr->m_SDlist.size()==1 )
            	{
            		if(m_signlesCommands==0)
            		{
            		std::string tmp_str = m_collectionName.substr(0, m_collectionName.find("_"));
            		m_tree = new GateSingleTree(tmp_str);
            		}
            		else
            			m_tree = new GateSingleTree(m_collectionName);
            	}
            	else
            		m_tree = new GateSingleTree(m_collectionName);

            	m_buffer.SetCCFlag(GetCCFlag());
            	m_tree->Init(m_buffer);
            }
        }

        void RecordDigitizer();

        GateRootSingleBuffer m_buffer;
        GateSingleTree *m_tree;
    };


    //--------------------------------------------------------------------------
    class CoincidenceOutputChannel : public VOutputChannel {
    public:
        inline CoincidenceOutputChannel(const G4String &aCollectionName, G4bool outputFlag)
                : VOutputChannel(aCollectionName, outputFlag, false),
                  m_tree(0) { m_buffer.Clear(); }

        virtual inline ~CoincidenceOutputChannel() {}

        inline void Clear() { m_buffer.Clear(); }

        inline void Book() {
        	 m_collectionID = -1;
            if (m_outputFlag) {
                m_tree = new GateCoincTree(m_collectionName);
                m_tree->Init(m_buffer);
            }
        }

        void RecordDigitizer();

        GateRootCoincBuffer m_buffer;
        GateCoincTree *m_tree;
    };



    //! flag to decide if it writes or not Hits, Singles and Digis to the ROOT file


    G4int GetSDlistSize() { return m_SDlistSize; };
    void SetSDlistSize(G4int size) {m_SDlistSize = size; };

    G4bool GetRootHitFlag() { return m_rootHitFlag; };

    void SetRootHitFlag(G4bool flag) { m_rootHitFlag = flag; };

    void SetRootCCFlag(G4bool flag) { m_rootCCFlag = flag; };
    G4bool GetRootCCFlag() {return  m_rootCCFlag; };

    void SetRootCCSourceParentIDSpecificationFlag(G4bool flag) { m_rootCCSourceParentIDSpecificationFlag = flag; };
    G4bool GetRootCCSourceParentIDSpecificationFlag() {return  m_rootCCSourceParentIDSpecificationFlag; };


    G4bool GetRootNtupleFlag() { return m_rootNtupleFlag; };

    void SetRootNtupleFlag(G4bool flag) { m_rootNtupleFlag = flag; };

    G4bool GetSaveRndmFlag() { return m_saveRndmFlag; };

    void SetSaveRndmFlag(G4bool flag) { m_saveRndmFlag = flag; };

    G4bool GetRootOpticalFlag() { return m_rootOpticalFlag; };

    void SetRootOpticalFlag(G4bool flag) { m_rootOpticalFlag = flag; };


    //! Get the output file name
    const G4String &GetFileName() { return m_fileName; };

// v. cuplov - m_fileName from SetFileName is defined without ".root"
// Additionnal root files names will be of the form GateOutPut_additionnalName.root.
// In the previous version of the code, file names would appear as GateOutPut.root_additionnalName.root
//! Set the output file name
//  void   SetFileName(const G4String aName)   { m_fileName = aName + ".root"; };
    void SetFileName(const G4String aName) { m_fileName = aName; };
// v. cuplov

    //! Get the output file path
    const char *GetFilePath() { return m_fileName.c_str(); };

    void SetRndmFreq(G4int val) { saveRndm = val; }

    G4int GetRndmFreq() { return saveRndm; }

    void RegisterNewSingleDigiCollection(const G4String &aCollectionName, G4bool outputFlag);

    void RegisterNewCoincidenceDigiCollection(const G4String &aCollectionName, G4bool outputFlag);

    void PrintRecStep();

    void SetVerboseLevel(G4int val) {
        GateVOutputModule::SetVerboseLevel(val);
        for (size_t i = 0; i < m_outputChannelList.size(); ++i)
            m_outputChannelList[i]->SetVerboseLevel(val);
    };

private:

    G4ThreeVector m_ionDecayPos;
    G4ThreeVector m_positronGenerationPos;
    G4ThreeVector m_positronAnnihilPos;

    G4double dxg1, dyg1, dzg1, dxg2, dyg2, dzg2;

    G4int saveRndm;

    G4double latestEventID; // Used by the gjs an d gjm programs (cluster mode)
    G4double nbPrimaries;
    G4double mTimeStop;
    G4double mTimeStart;

    G4double m_positronKinEnergy;
    G4int m_recordFlag;

    TFile *m_hfile; // the file for histograms, tree ...

    //OK GND 2022
    //GateHitTree *m_treeHit; // the tree for hit quantities
    //for multiple SDs
    std::vector<GateHitTree *> m_treesHit; // the tree for hit quantities
    //Number of SD is saved in the following variable for not calling at each event for hits GateDigitizerMgr::GetInstance()
    G4int m_SDlistSize;

    TH1D *m_total_nb_primaries_hist; //histogram of total_nb_primaries
    TH1D *m_latest_event_ID_hist;
    TDirectory *m_working_root_directory;

    // OK GND 2022
    //GateRootHitBuffer m_hitBuffer;
    std::vector<GateRootHitBuffer> m_hitBuffers; // the tree for hit quantities

// v. cuplov - optical photons
    GateTrajectoryNavigator *m_trajectoryNavigator;
    //TTree *OpticalTree; // new tree
    std::vector<TTree*> m_OpticalTrees; // new tree
    Char_t NameOfProcessInCrystal[40];
    Char_t NameOfProcessInPhantom[40];
//  G4int nPhantomOpticalRayleigh;
//  G4int nPhantomOpticalMie;
//  G4int nPhantomOpticalAbsorption;
//  G4int nCrystalOpticalRayleigh;
//  G4int nCrystalOpticalMie;
//  G4int nCrystalOpticalAbsorption;
    G4int nScintillation, nCrystalOpticalWLS, nPhantomOpticalWLS;
    G4int NumCrystalWLS, NumPhantomWLS;
    G4double CrystalLastHitPos_X, CrystalLastHitPos_Y, CrystalLastHitPos_Z, CrystalLastHitEnergy;
//  G4double CrystalAbsorbedPhotonHitPos_X, CrystalAbsorbedPhotonHitPos_Y,CrystalAbsorbedPhotonHitPos_Z;
    G4double PhantomLastHitPos_X, PhantomLastHitPos_Y, PhantomLastHitPos_Z, PhantomLastHitEnergy;
//  G4double PhantomAbsorbedPhotonHitPos_X, PhantomAbsorbedPhotonHitPos_Y,PhantomAbsorbedPhotonHitPos_Z;
    G4double PhantomWLSPos_X, PhantomWLSPos_Y, PhantomWLSPos_Z;
    G4double MomentumDirectionx, MomentumDirectiony, MomentumDirectionz;
// v. cuplov - optical photons

    G4bool m_rootHitFlag;
    G4bool m_rootCCFlag;
    G4bool m_rootCCSourceParentIDSpecificationFlag;
    G4bool m_rootNtupleFlag;
    G4bool m_saveRndmFlag;
    G4bool m_rootOpticalFlag = false;

    G4String m_fileName;

    GateToRootMessenger *m_rootMessenger;

    std::vector<VOutputChannel *> m_outputChannelList;

    /* PY Descourt 08/09/2009 */
    G4double m_positronKinEnergy_copy;
    G4ThreeVector m_ionDecayPos_copy;
    G4ThreeVector m_positronGenerationPos_copy;
    G4ThreeVector m_positronAnnihilPos_copy;
    G4double dxg1_copy, dyg1_copy, dzg1_copy, dxg2_copy, dyg2_copy, dzg2_copy;
    G4int m_RSEventID, m_RSRunID;

    /// ROOT FILE & DATAS FOR TRACKS INFOS
    G4int m_currentTracksData; // stores the current index of the Tracks Data when reading Root file in detector mode
    G4int m_currentRSData; // stores the current index of the Record Stepping Data when reading Root file in detector mode
    GateTrack *m_currentGTrack;  // in detector mode contains all current Track Data
    G4String fTracksFN;
    TFile *m_TracksFile;
    TTree *tracksTuple;
    TTree *m_RecStepTree;

    G4int fSkipRecStepData;
    Long64_t last_RSEventID;

//
//// data from GateAnalysis::RecordEndOfEvent() stored in  the RecStepData Root File
//
    ComptonRayleighData theCRData;
    ComptonRayleighData theCRData_copy;

    G4int m_EOF;  // 1 if tracks root file end of file is reached


    G4int TrackID, ParentID, RunID, EventID;
    G4int PDGCode, m_sourceID;
    G4int m_wasKilled;
    G4double posx, posy, posz, MDirectionx, MDirectiony, MDirectionz, Momentumx, Momentumy, Momentumz;
    G4double VertexPositionx, VertexPositiony, VertexPositionz, VtxMomDirx, VtxMomDiry, VtxMomDirz, Polarizationx, Polarizationy, Polarizationz;
    G4double LTime, GTime, PTime, Energy, KinEnergy, Velocity, VertexKineticEnergy, Weight;
    G4double Wavelength;
    G4double m_EventTime;
    G4String m_particleName;
    Char_t m_volumeName[40], m_processName[40], m_parentparticleName[40];
    /* PY Descourt 08/09/2009 */
};
//--------------------------------------------------------------------------

#endif
#endif
