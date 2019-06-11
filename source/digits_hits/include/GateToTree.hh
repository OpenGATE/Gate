/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


//
// Created by mdupont on 17/05/19.
//

#ifndef GATE_GATETOTREE_HH
#define GATE_GATETOTREE_HH

#include "GateVOutputModule.hh"
#include "GateTreeFileManager.hh"
#include "G4SystemOfUnits.hh"
//#include <vector>
#include <unordered_map>

class GateToTreeMessenger;

class GateCoincidenceDigi;

class GateToTree : public GateVOutputModule
{

public:
  GateToTree(const G4String &name, GateOutputMgr *outputMgr, DigiMode digiMode);

  void RecordBeginOfAcquisition() override;

  virtual ~GateToTree();

  void RecordEndOfAcquisition() override;

  void RecordBeginOfRun(const G4Run *run) override;

  void RecordEndOfRun(const G4Run *run) override;

  void RecordBeginOfEvent(const G4Event *event) override;

  void RecordEndOfEvent(const G4Event *event) override;

  void RecordStepWithVolume(const GateVVolume *v, const G4Step *step) override;

  void RecordVoxels(GateVGeometryVoxelStore *store) override;
  void RegisterNewCoincidenceDigiCollection(const G4String &string, G4bool aBool) override;
  void RegisterNewSingleDigiCollection(const G4String &string, G4bool aBool) override;

  const G4String &GiveNameOfFile() override;
  void addFileName(const G4String &s);

  static void SetOutputIDName(G4int id_system, const char * anOutputIDName, size_t depth);

private:

  void retrieve(GateCoincidenceDigi* aDigi, G4int side, G4int system_id);

  template <typename T>
  void retrieve(T* p, G4int system_id) //p == GateCrystalHit, GateSingleDigi, or &GatePulses
  {
      m_runID = p->GetRunID();
      m_eventID[0] = p->GetEventID();
      m_sourceID[0] = p->GetSourceID();
      m_sourcePosX[0] = p->GetSourcePosition().x() / mm;
      m_sourcePosY[0] = p->GetSourcePosition().y() / mm;
      m_sourcePosZ[0] = p->GetSourcePosition().z() / mm;

      m_posX[0] = p->GetGlobalPos().x() / mm;
      m_posY[0] = p->GetGlobalPos().y() / mm;
      m_posZ[0] = p->GetGlobalPos().z() / mm;
      m_time[0] = p->GetTime() / s;

      m_axialPos = p->GetScannerPos().z() / mm;
      m_rotationAngle = p->GetScannerRotAngle() / degree;

      m_comptonVolumeName[0] = p->GetComptonVolumeName();
      m_RayleighVolumeName[0] = p->GetRayleighVolumeName();

      m_nPhantomCompton[0] = p->GetNPhantomCompton();
      m_nCrystalCompton[0] = p->GetNCrystalCompton();
      m_nPhantomRayleigh[0] = p->GetNPhantomRayleigh();
      m_nCrystalRayleigh[0] = p->GetNCrystalRayleigh();

      for(auto depth = 0; depth < MAX_DEPTH_SYSTEM; ++depth)
          m_outputID[0][system_id][depth] = p->GetComponentID(depth);


  }

  GateToTreeMessenger *m_messenger;
  GateOutputTreeFileManager m_manager_hits;
//  std::vector<GateOutputTreeFileManager> m_vmanager_singles;
  std::unordered_map<std::string, GateOutputTreeFileManager> m_mmanager_singles;
  std::unordered_map<std::string, G4int> m_singles_to_collectionID;

  std::unordered_map<std::string, GateOutputTreeFileManager> m_mmanager_coincidences;
  std::unordered_map<std::string, G4int> m_coincidences_to_collectionID;

  std::vector<std::string> m_listOfFileName;
  std::vector<std::string> m_listOfSinglesCollection;
  std::vector<std::string> m_listOfCoincidencesCollection;

  G4bool m_hits_enabled;
public:
  G4bool getHitsEnabled() const;
  void setHitsEnabled(G4bool mHitsEnabled);
  void addCollection(const std::string &str); //called by messenger

private:


  G4String m_uselessFileName; //only for GiveNameOfFile which return a reference..



  G4int m_PDGEncoding;
  G4int m_trackID;
  G4int m_parentID;
  G4double m_trackLocalTime;
  G4double m_time[2];
  G4double m_edep[2];
  G4double m_stepLength;
  G4double m_trackLength;


  G4int m_runID;
  G4int m_eventID[2];
  G4int m_sourceID[2];
  G4int m_primaryID;


  G4double m_localPosX;
  G4double m_localPosY;
  G4double m_localPosZ;

  G4double m_posX[2];
  G4double m_posY[2];
  G4double m_posZ[2];

  G4double m_momDirX;
  G4double m_momDirY;
  G4double m_momDirZ;

  G4double m_axialPos;
  G4double m_rotationAngle;


  std::string m_processName;
  std::string m_comptonVolumeName[2];
  std::string m_RayleighVolumeName[2];



  G4double m_sourcePosX[2];
  G4double m_sourcePosY[2];
  G4double m_sourcePosZ[2];

  G4int m_nPhantomCompton[2];
  G4int m_nCrystalCompton[2];
  G4int m_nPhantomRayleigh[2];
  G4int m_nCrystalRayleigh[2];

  G4double m_sinogramTheta;
  G4double m_sinogramS;



  static const auto VOLUMEID_SIZE = 10;
//  static const auto OUTPUTID_SIZE = 6;

  G4int m_volumeID[VOLUMEID_SIZE];
  G4int m_photonID;
//  G4int m_outpuID[OUTPUTID_SIZE];



 public:


  static const auto MAX_NB_SYSTEM = 32;
  static const auto MAX_DEPTH_SYSTEM = 6;
  static const auto MAX_OUTPUTIDNAME_SIZE = 32;

  static char m_outputIDName[MAX_NB_SYSTEM][MAX_DEPTH_SYSTEM][MAX_OUTPUTIDNAME_SIZE];
  static G4int m_max_depth_system[MAX_NB_SYSTEM];

 private:
  G4int m_outputID[2][MAX_NB_SYSTEM][MAX_DEPTH_SYSTEM];






};


#endif //GATE_GATETOTREE_HH
