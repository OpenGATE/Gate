/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

//
// Created by mdupont on 17/05/19.
//

#include "GateToTree.hh"
#include "GateToTree.hh"

#include <cassert>

#include "GateToTreeMessenger.hh"

#include "GateOutputMgr.hh"
#include "GateSystemListManager.hh"
#include "GateMiscFunctions.hh"
#include "G4DigiManager.hh"


char GateToTree::m_outputIDName[GateToTree::MAX_NB_SYSTEM][GateToTree::MAX_DEPTH_SYSTEM][GateToTree::MAX_OUTPUTIDNAME_SIZE];
G4int GateToTree::m_max_depth_system[GateToTree::MAX_NB_SYSTEM];
const auto MAX_NB_CHARACTER = 32;


SaveDataParam::SaveDataParam()
{
  m_save = true;
}

bool SaveDataParam::toSave() const
{
  return m_save;
}

void SaveDataParam::setToSave(bool mSave)
{
  m_save = mSave;
}


GateToTree::GateToTree(const G4String &name, GateOutputMgr *outputMgr, DigiMode digiMode) :
    GateVOutputModule(name, outputMgr, digiMode)
{
    m_hitsParams_to_write.emplace("PDGEncoding", SaveDataParam());
    m_hitsParams_to_write.emplace("trackID", SaveDataParam());
    m_hitsParams_to_write.emplace("parentID", SaveDataParam());
    m_hitsParams_to_write.emplace("trackLocalTime", SaveDataParam());
    m_hitsParams_to_write.emplace("trackLocalTime", SaveDataParam());
    m_hitsParams_to_write.emplace("time", SaveDataParam());
    m_hitsParams_to_write.emplace("runID", SaveDataParam());
    m_hitsParams_to_write.emplace("eventID", SaveDataParam());
    m_hitsParams_to_write.emplace("sourceID", SaveDataParam());
    m_hitsParams_to_write.emplace("primaryID", SaveDataParam());
    m_hitsParams_to_write.emplace("posX", SaveDataParam());
    m_hitsParams_to_write.emplace("posY", SaveDataParam());
    m_hitsParams_to_write.emplace("posZ", SaveDataParam());
    m_hitsParams_to_write.emplace("localPosX", SaveDataParam());
    m_hitsParams_to_write.emplace("localPosY", SaveDataParam());
    m_hitsParams_to_write.emplace("localPosZ", SaveDataParam());
    m_hitsParams_to_write.emplace("momDirX", SaveDataParam());
    m_hitsParams_to_write.emplace("momDirY", SaveDataParam());
    m_hitsParams_to_write.emplace("momDirZ", SaveDataParam());
    m_hitsParams_to_write.emplace("edep", SaveDataParam());
    m_hitsParams_to_write.emplace("stepLength", SaveDataParam());
    m_hitsParams_to_write.emplace("trackLength", SaveDataParam());
    m_hitsParams_to_write.emplace("rotationAngle", SaveDataParam());
    m_hitsParams_to_write.emplace("axialPos", SaveDataParam());
    m_hitsParams_to_write.emplace("processName", SaveDataParam());
    m_hitsParams_to_write.emplace("comptVolName", SaveDataParam());
    m_hitsParams_to_write.emplace("RayleighVolName", SaveDataParam());
    m_hitsParams_to_write.emplace("sourcePosX", SaveDataParam());
    m_hitsParams_to_write.emplace("sourcePosY", SaveDataParam());
    m_hitsParams_to_write.emplace("sourcePosZ", SaveDataParam());
    m_hitsParams_to_write.emplace("nPhantomCompton", SaveDataParam());
    m_hitsParams_to_write.emplace("nCrystalCompton", SaveDataParam());
    m_hitsParams_to_write.emplace("nPhantomRayleigh", SaveDataParam());
    m_hitsParams_to_write.emplace("nCrystalRayleigh", SaveDataParam());
    m_hitsParams_to_write.emplace("photonID", SaveDataParam());
    m_hitsParams_to_write.emplace("volumeIDs", SaveDataParam());
    m_hitsParams_to_write.emplace("componentsIDs", SaveDataParam());

    m_singlesParams_to_write.emplace("runID", SaveDataParam());
    m_singlesParams_to_write.emplace("eventID", SaveDataParam());
    m_singlesParams_to_write.emplace("sourceID", SaveDataParam());
    m_singlesParams_to_write.emplace("sourcePosX", SaveDataParam());
    m_singlesParams_to_write.emplace("sourcePosY", SaveDataParam());
    m_singlesParams_to_write.emplace("sourcePosZ", SaveDataParam());
    m_singlesParams_to_write.emplace("globalPosX", SaveDataParam());
    m_singlesParams_to_write.emplace("globalPosY", SaveDataParam());
    m_singlesParams_to_write.emplace("globalPosZ", SaveDataParam());
    m_singlesParams_to_write.emplace("time", SaveDataParam());
    m_singlesParams_to_write.emplace("energy", SaveDataParam());
    m_singlesParams_to_write.emplace("comptonPhantom", SaveDataParam());
    m_singlesParams_to_write.emplace("comptonCrystal", SaveDataParam());
    m_singlesParams_to_write.emplace("RayleighPhantom", SaveDataParam());
    m_singlesParams_to_write.emplace("RayleighCrystal", SaveDataParam());
    m_singlesParams_to_write.emplace("comptVolName", SaveDataParam());
    m_singlesParams_to_write.emplace("RayleighVolName", SaveDataParam());
    m_singlesParams_to_write.emplace("rotationAngle", SaveDataParam());
    m_singlesParams_to_write.emplace("axialPos", SaveDataParam());
    m_singlesParams_to_write.emplace("componentsIDs", SaveDataParam());

    m_coincidencesParams_to_write.emplace("runID", SaveDataParam());
    m_coincidencesParams_to_write.emplace("eventID", SaveDataParam());
    m_coincidencesParams_to_write.emplace("sourceID", SaveDataParam());
    m_coincidencesParams_to_write.emplace("sourcePosX", SaveDataParam());
    m_coincidencesParams_to_write.emplace("sourcePosY", SaveDataParam());
    m_coincidencesParams_to_write.emplace("sourcePosZ", SaveDataParam());
    m_coincidencesParams_to_write.emplace("rotationAngle", SaveDataParam());
    m_coincidencesParams_to_write.emplace("axialPos", SaveDataParam());
    m_coincidencesParams_to_write.emplace("globalPosX", SaveDataParam());
    m_coincidencesParams_to_write.emplace("globalPosY", SaveDataParam());
    m_coincidencesParams_to_write.emplace("globalPosZ", SaveDataParam());
    m_coincidencesParams_to_write.emplace("time", SaveDataParam());
    m_coincidencesParams_to_write.emplace("energy", SaveDataParam());
    m_coincidencesParams_to_write.emplace("comptVolName", SaveDataParam());
    m_coincidencesParams_to_write.emplace("RayleighVolName", SaveDataParam());
    m_coincidencesParams_to_write.emplace("comptonPhantom", SaveDataParam());
    m_coincidencesParams_to_write.emplace("comptonCrystal", SaveDataParam());
    m_coincidencesParams_to_write.emplace("RayleighPhantom", SaveDataParam());
    m_coincidencesParams_to_write.emplace("RayleighCrystal", SaveDataParam());
    m_coincidencesParams_to_write.emplace("sinogramTheta", SaveDataParam());
    m_coincidencesParams_to_write.emplace("sinogramS", SaveDataParam());
    m_coincidencesParams_to_write.emplace("componentsIDs", SaveDataParam());


    m_messenger = new GateToTreeMessenger(this);
    m_hits_enabled = false;
}

void GateToTree::RecordBeginOfAcquisition()
{
    if(!this->IsEnabled())
      return;

    if(m_hits_enabled)
    {
      for(auto &&fileName: m_listOfFileName)
      {
        auto extension = getExtension(fileName);
        auto name = removeExtension(fileName);

        G4String hits_filename = name + ".hits." + extension;
        m_manager_hits.add_file(hits_filename, extension);
      }

      if(m_hitsParams_to_write.at("PDGEncoding").toSave())
        m_manager_hits.write_variable("PDGEncoding", &m_PDGEncoding);

      if(m_hitsParams_to_write.at("trackID").toSave())
        m_manager_hits.write_variable("trackID", &m_trackID);

      if(m_hitsParams_to_write.at("parentID").toSave())
        m_manager_hits.write_variable("parentID", &m_parentID);

      if(m_hitsParams_to_write.at("trackLocalTime").toSave())
        m_manager_hits.write_variable("trackLocalTime", &m_trackLocalTime);

      if(m_hitsParams_to_write.at("time").toSave())
        m_manager_hits.write_variable("time", &m_time[0]);

      if(m_hitsParams_to_write.at("runID").toSave())
        m_manager_hits.write_variable("runID", &m_runID);

      if(m_hitsParams_to_write.at("eventID").toSave())
        m_manager_hits.write_variable("eventID", &m_eventID[0]);

      if(m_hitsParams_to_write.at("sourceID").toSave())
        m_manager_hits.write_variable("sourceID", &m_sourceID[0]);

      if(m_hitsParams_to_write.at("primaryID").toSave())
        m_manager_hits.write_variable("primaryID", &m_primaryID);

      if(m_hitsParams_to_write.at("posX").toSave())
        m_manager_hits.write_variable("posX", &m_posX[0]);

      if(m_hitsParams_to_write.at("posY").toSave())
        m_manager_hits.write_variable("posY", &m_posY[0]);

      if(m_hitsParams_to_write.at("posZ").toSave())
        m_manager_hits.write_variable("posZ", &m_posZ[0]);

      if(m_hitsParams_to_write.at("localPosX").toSave())
        m_manager_hits.write_variable("localPosX", &m_localPosX);

      if(m_hitsParams_to_write.at("localPosY").toSave())
        m_manager_hits.write_variable("localPosY", &m_localPosY);

      if(m_hitsParams_to_write.at("localPosZ").toSave())
        m_manager_hits.write_variable("localPosZ", &m_localPosZ);

      if(m_hitsParams_to_write.at("momDirX").toSave())
        m_manager_hits.write_variable("momDirX", &m_momDirX);

      if(m_hitsParams_to_write.at("momDirY").toSave())
        m_manager_hits.write_variable("momDirY", &m_momDirY);

      if(m_hitsParams_to_write.at("momDirZ").toSave())
        m_manager_hits.write_variable("momDirZ", &m_momDirZ);

      if(m_hitsParams_to_write.at("edep").toSave())
        m_manager_hits.write_variable("edep", &m_edep[0]);

      if(m_hitsParams_to_write.at("stepLength").toSave())
        m_manager_hits.write_variable("stepLength", &m_stepLength);

      if(m_hitsParams_to_write.at("trackLength").toSave())
        m_manager_hits.write_variable("trackLength", &m_trackLength);

      if(m_hitsParams_to_write.at("rotationAngle").toSave())
        m_manager_hits.write_variable("rotationAngle", &m_rotationAngle);

      if(m_hitsParams_to_write.at("axialPos").toSave())
        m_manager_hits.write_variable("axialPos", &m_axialPos);

      if(m_hitsParams_to_write.at("processName").toSave())
        m_manager_hits.write_variable("processName", &m_processName, MAX_NB_CHARACTER);

      if(m_hitsParams_to_write.at("comptVolName").toSave())
        m_manager_hits.write_variable("comptVolName", &m_comptonVolumeName[0], MAX_NB_CHARACTER);

      if(m_hitsParams_to_write.at("RayleighVolName").toSave())
        m_manager_hits.write_variable("RayleighVolName", &m_RayleighVolumeName[0], MAX_NB_CHARACTER);


      if(m_hitsParams_to_write.at("volumeIDs").toSave())
      {
        for(auto i = 0; i < VOLUMEID_SIZE; ++i)
        {
          std::stringstream ss;
          ss << "volumeID[" << i << "]";
          m_manager_hits.write_variable(ss.str(), &m_volumeID[i]);
        }
      }



      if(m_hitsParams_to_write.at("sourcePosX").toSave())
        m_manager_hits.write_variable("sourcePosX", &m_sourcePosX[0]);
      if(m_hitsParams_to_write.at("sourcePosY").toSave())
        m_manager_hits.write_variable("sourcePosY", &m_sourcePosY[0]);
      if(m_hitsParams_to_write.at("sourcePosZ").toSave())
        m_manager_hits.write_variable("sourcePosZ", &m_sourcePosZ[0]);

      if(m_hitsParams_to_write.at("nPhantomCompton").toSave())
        m_manager_hits.write_variable("nPhantomCompton", &m_nPhantomCompton[0]);

      if(m_hitsParams_to_write.at("nCrystalCompton").toSave())
        m_manager_hits.write_variable("nCrystalCompton", &m_nCrystalCompton[0]);

      if(m_hitsParams_to_write.at("nPhantomRayleigh").toSave())
        m_manager_hits.write_variable("nPhantomRayleigh", &m_nPhantomRayleigh[0]);

      if(m_hitsParams_to_write.at("nCrystalRayleigh").toSave())
        m_manager_hits.write_variable("nCrystalRayleigh", &m_nCrystalRayleigh[0]);


      if(m_hitsParams_to_write.at("componentsIDs").toSave())
      {
        for(auto k = 0; k < GateSystemListManager::GetInstance()->size(); ++k)
        {
          for(auto depth = 0; depth < m_max_depth_system[k]; ++depth)
          {
            std::stringstream ss;
            ss << m_outputIDName[k][depth];
            m_manager_hits.write_variable(ss.str(), &m_outputID[0][k][depth]);
          }
        }
      }


      if(m_hitsParams_to_write.at("photonID").toSave())
        m_manager_hits.write_variable("photonID", &m_photonID);
      m_manager_hits.write_header();
    }

    for(auto &&m: m_mmanager_singles)
    {

        auto &mm = m.second;

        if(m_singlesParams_to_write.at("runID").toSave())
          mm.write_variable("runID", &m_runID);

        if(m_singlesParams_to_write.at("eventID").toSave())
          mm.write_variable("eventID", &m_eventID[0]);

        if(m_singlesParams_to_write.at("sourceID").toSave())
          mm.write_variable("sourceID", &m_sourceID[0]);

        if(m_singlesParams_to_write.at("sourcePosX").toSave())
          mm.write_variable("sourcePosX", &m_sourcePosX[0]);
        if(m_singlesParams_to_write.at("sourcePosY").toSave())
          mm.write_variable("sourcePosY", &m_sourcePosY[0]);
        if(m_singlesParams_to_write.at("sourcePosZ").toSave())
          mm.write_variable("sourcePosZ", &m_sourcePosZ[0]);

        if(m_singlesParams_to_write.at("globalPosX").toSave())
          mm.write_variable("globalPosX", &m_posX[0]);
        if(m_singlesParams_to_write.at("globalPosY").toSave())
          mm.write_variable("globalPosY", &m_posY[0]);
        if(m_singlesParams_to_write.at("globalPosZ").toSave())
          mm.write_variable("globalPosZ", &m_posZ[0]);

        if(m_singlesParams_to_write.at("componentsIDs").toSave())
        {
          for(auto k = 0; k < GateSystemListManager::GetInstance()->size(); ++k)
          {
            for(auto depth = 0; depth < m_max_depth_system[k]; ++depth)
            {
              std::stringstream ss;
              ss << m_outputIDName[k][depth];
              mm.write_variable(ss.str(), &m_outputID[0][k][depth]);
            }
          }
        }


        if(m_singlesParams_to_write.at("time").toSave())
          mm.write_variable("time", &m_time[0]);
        if(m_singlesParams_to_write.at("energy").toSave())
          mm.write_variable("energy", &m_edep[0]);

        if(m_singlesParams_to_write.at("comptonPhantom").toSave())
          mm.write_variable("comptonPhantom", &m_nPhantomCompton[0]);
        if(m_singlesParams_to_write.at("comptonCrystal").toSave())
          mm.write_variable("comptonCrystal", &m_nCrystalCompton[0]);
        if(m_singlesParams_to_write.at("RayleighPhantom").toSave())
          mm.write_variable("RayleighPhantom", &m_nPhantomRayleigh[0]);
        if(m_singlesParams_to_write.at("RayleighCrystal").toSave())
          mm.write_variable("RayleighCrystal", &m_nCrystalRayleigh[0]);

        if(m_singlesParams_to_write.at("comptVolName").toSave())
          mm.write_variable("comptVolName", &m_comptonVolumeName[0], MAX_NB_CHARACTER);
        if(m_singlesParams_to_write.at("RayleighVolName").toSave())
          mm.write_variable("RayleighVolName", &m_RayleighVolumeName[0], MAX_NB_CHARACTER);

        if(m_singlesParams_to_write.at("rotationAngle").toSave())
          mm.write_variable("rotationAngle", &m_rotationAngle);
        if(m_singlesParams_to_write.at("axialPos").toSave())
          mm.write_variable("axialPos", &m_axialPos);
        mm.write_header();
    }

    for(auto &&m: m_mmanager_coincidences)
    {

        auto &mm = m.second;
        if(m_coincidencesParams_to_write.at("runID").toSave())
          mm.write_variable("runID", &m_runID);

        if(m_coincidencesParams_to_write.at("eventID").toSave())
        {
          mm.write_variable("eventID1", &m_eventID[0]);
          mm.write_variable("eventID2", &m_eventID[1]);
        }

        if(m_coincidencesParams_to_write.at("sourceID").toSave())
        {
          mm.write_variable("sourceID1", &m_sourceID[0]);
          mm.write_variable("sourceID2", &m_sourceID[1]);
        }

        if(m_coincidencesParams_to_write.at("sourcePosX").toSave())
        {
          mm.write_variable("sourcePosX1", &m_sourcePosX[0]);
          mm.write_variable("sourcePosX2", &m_sourcePosX[1]);
        }
        if(m_coincidencesParams_to_write.at("sourcePosY").toSave())
        {
          mm.write_variable("sourcePosY1", &m_sourcePosY[0]);
          mm.write_variable("sourcePosY2", &m_sourcePosY[1]);
        }

        if(m_coincidencesParams_to_write.at("sourcePosZ").toSave())
        {
          mm.write_variable("sourcePosZ1", &m_sourcePosZ[0]);
          mm.write_variable("sourcePosZ2", &m_sourcePosZ[1]);
        }


      if(m_coincidencesParams_to_write.at("rotationAngle").toSave())
        mm.write_variable("rotationAngle", &m_rotationAngle);

      if(m_coincidencesParams_to_write.at("axialPos").toSave())
        mm.write_variable("axialPos", &m_axialPos);

      if(m_coincidencesParams_to_write.at("globalPosX").toSave())
      {
        mm.write_variable("globalPosX1", &m_posX[0]);
        mm.write_variable("globalPosX2", &m_posX[1]);
      }

      if(m_coincidencesParams_to_write.at("globalPosY").toSave())
      {
        mm.write_variable("globalPosY1", &m_posY[0]);
        mm.write_variable("globalPosY2", &m_posY[1]);
      }

      if(m_coincidencesParams_to_write.at("globalPosZ").toSave())
      {
        mm.write_variable("globalPosZ1", &m_posZ[0]);
        mm.write_variable("globalPosZ2", &m_posZ[1]);
      }

      if(m_coincidencesParams_to_write.at("time").toSave())
      {
        mm.write_variable("time1", &m_time[0]);
        mm.write_variable("time2", &m_time[1]);
      }

      if(m_coincidencesParams_to_write.at("energy").toSave())
      {
        mm.write_variable("energy1", &m_edep[0]);
        mm.write_variable("energy2", &m_edep[1]);
      }

      if(m_coincidencesParams_to_write.at("comptVolName").toSave())
      {
        mm.write_variable("comptVolName1", &m_comptonVolumeName[0], MAX_NB_CHARACTER);
        mm.write_variable("comptVolName2", &m_comptonVolumeName[1], MAX_NB_CHARACTER);
      }

      if(m_coincidencesParams_to_write.at("RayleighVolName").toSave())
      {
        mm.write_variable("RayleighVolName1", &m_RayleighVolumeName[0], MAX_NB_CHARACTER);
        mm.write_variable("RayleighVolName2", &m_RayleighVolumeName[1], MAX_NB_CHARACTER);
      }

      if(m_coincidencesParams_to_write.at("comptonPhantom").toSave())
      {
        mm.write_variable("comptonPhantom1", &m_nPhantomCompton[0]);
        mm.write_variable("comptonPhantom2", &m_nPhantomCompton[1]);
      }
      if(m_coincidencesParams_to_write.at("comptonCrystal").toSave())
      {
        mm.write_variable("comptonCrystal1", &m_nCrystalCompton[0]);
        mm.write_variable("comptonCrystal2", &m_nCrystalCompton[1]);
      }
      if(m_coincidencesParams_to_write.at("RayleighPhantom").toSave())
      {
        mm.write_variable("RayleighPhantom1", &m_nPhantomRayleigh[0]);
        mm.write_variable("RayleighPhantom2", &m_nPhantomRayleigh[1]);
      }
      if(m_coincidencesParams_to_write.at("RayleighCrystal").toSave())
      {
        mm.write_variable("RayleighCrystal1", &m_nCrystalRayleigh[0]);
        mm.write_variable("RayleighCrystal2", &m_nCrystalRayleigh[1]);
      }

      if(m_coincidencesParams_to_write.at("componentsIDs").toSave())
      {
        for(auto side = 1; side <= 2; ++side)
        {
          for(auto k = 0; k < GateSystemListManager::GetInstance()->size(); ++k)
          {
            for(auto depth = 0; depth < m_max_depth_system[k]; ++depth)
            {
              std::stringstream ss;
              ss << m_outputIDName[k][depth] << side;
              mm.write_variable(ss.str(), &m_outputID[side - 1][k][depth]);
            }
          }
        }
      }

      if(m_coincidencesParams_to_write.at("sinogramTheta").toSave())
        mm.write_variable("sinogramTheta", &m_sinogramTheta);

      if(m_coincidencesParams_to_write.at("sinogramS").toSave())
        mm.write_variable("sinogramS", &m_sinogramS);
      mm.write_header();
    }


}

void GateToTree::RecordEndOfAcquisition()
{
  m_manager_hits.close();
  for(auto &&m: m_mmanager_singles)
    m.second.close();

  for(auto &&m: m_mmanager_coincidences)
    m.second.close();

}

void GateToTree::RecordBeginOfRun(const G4Run *run)
{

}

void GateToTree::RecordEndOfRun(const G4Run *run)
{

}

void GateToTree::RecordBeginOfEvent(const G4Event *event)
{

}

void GateToTree::RecordEndOfEvent(const G4Event *event)
{
  auto CHC = this->GetOutputMgr()->GetCrystalHitCollection();
  if(!CHC)
  {
    std::cout <<
              "GateToTree::RecordEndOfEvent : GateCrystalHitCollection not found"
              << Gateendl;
    return;
  }

//    auto writeComptonVolumeName = m_hitsParams_to_write.at("comptVolName").toSave();
//    auto writeRayleighVolName = m_hitsParams_to_write.at("RayleighVolName").toSave();
//    auto writeVolumeIDs = m_hitsParams_to_write.at("volumeIDs").toSave();

  for( G4int iHit = 0; iHit < CHC->entries(); ++iHit )
  {
    auto hit = (*CHC)[ iHit ];
    if(!hit->GoodForAnalysis())
      continue;

    auto system_id = hit->GetSystemID();
    retrieve(hit, system_id);

    if(m_hitsParams_to_write.at("componentsIDs").toSave())
    {
      for(auto depth = 0; depth < MAX_DEPTH_SYSTEM; ++depth)
        m_outputID[0][system_id][depth] = hit->GetComponentID(depth);
    }

//        if(writeComptonVolumeName)
    m_comptonVolumeName[0] = hit->GetComptonVolumeName();

//        if(writeRayleighVolName)
    m_RayleighVolumeName[0] = hit->GetRayleighVolumeName();

    m_primaryID = hit->GetPrimaryID();

    m_localPosX = hit->GetLocalPos().x() / mm;
    m_localPosY = hit->GetLocalPos().y() / mm;
    m_localPosZ = hit->GetLocalPos().z() / mm;

    m_momDirX = hit->GetMomentumDir().x();
    m_momDirY = hit->GetMomentumDir().y();
    m_momDirZ = hit->GetMomentumDir().z();

    m_PDGEncoding = hit->GetPDGEncoding();
    m_trackID = hit->GetTrackID();
    m_parentID = hit->GetParentID();

    m_trackLocalTime = hit->GetTrackLocalTime() / second;
    m_edep[0] = hit->GetEdep() / MeV;
    m_stepLength = hit->GetStepLength() / mm;
    m_trackLength = hit->GetTrackLength() / mm;

    m_processName = hit->GetProcess();

    if(m_hitsParams_to_write.at("volumeIDs").toSave())
      hit->GetVolumeID().StoreDaughterIDs(m_volumeID, VOLUMEID_SIZE);

    m_photonID = hit->GetPhotonID();
    m_manager_hits.fill();
  }

  auto fDM = G4DigiManager::GetDMpointer();

  if(!m_singles_to_collectionID.size())
  {
    for(auto&& m: m_mmanager_singles)
    {
      auto collectionID = fDM->GetDigiCollectionID(m.first);
      m_singles_to_collectionID.emplace(m.first, collectionID);
    }
  }



  for(auto&& m: m_mmanager_singles)
  {
//        auto collectionID = fDM->GetDigiCollectionID(m.first);
    auto collectionID = m_singles_to_collectionID.at(m.first);
//        auto SDC = static_cast<const GateSingleDigiCollection*>(fDM->GetDigiCollection(collectionID));
    const GateSingleDigiCollection * SDC =
        (GateSingleDigiCollection*) (fDM->GetDigiCollection( collectionID ));

    if(!SDC)
    {
//            G4cout << "GateToTree::RecordEndOfEvent no collection = " << m.first << "\n";
      continue;
    }

    assert(SDC);
    auto v = SDC->GetVector();
    assert(v);

    G4int system_id = -1;


    for(auto &&digi: *v)
    {
      if(system_id == -1)
      {
        auto mother = static_cast<const GateCrystalHit*>(digi->GetPulse().GetMother());
        assert(mother);
        system_id = mother->GetSystemID();
      }

      retrieve(digi, system_id);
      m_comptonVolumeName[0] = digi->GetComptonVolumeName();
      m_RayleighVolumeName[0] = digi->GetRayleighVolumeName();
      if(m_singlesParams_to_write.at("componentsIDs").toSave())
      {
        for(auto depth = 0; depth < MAX_DEPTH_SYSTEM; ++depth)
          m_outputID[0][system_id][depth] = digi->GetComponentID(depth);
      }

      m_edep[0] = digi->GetEnergy() / MeV;
      m.second.fill();
    }

  }

  if(!m_coincidences_to_collectionID.size())
  {
    for(auto&& m: m_mmanager_coincidences)
    {
      auto collectionID = fDM->GetDigiCollectionID(m.first);
      m_coincidences_to_collectionID.emplace(m.first, collectionID);
    }
  }

  for(auto&& m: m_mmanager_coincidences)
  {
    auto collectionID = m_coincidences_to_collectionID.at(m.first);
    auto SDC = static_cast<const GateCoincidenceDigiCollection*>(fDM->GetDigiCollection(collectionID));

    if(!SDC)
    {
      continue;
    }

    assert(SDC);
    auto v = SDC->GetVector();
    assert(v);

    G4int system_id = -1;


    for(auto &&digi: *v)
    {
      if(system_id == -1)
      {
        auto mother = static_cast<const GateCrystalHit*>(digi->GetPulse(0).GetMother());
        assert(mother);
        system_id = mother->GetSystemID();
      }

      const auto &pulse = digi->GetPulse(0);
      retrieve(&pulse, system_id);
      m_comptonVolumeName[0] = pulse.GetComptonVolumeName();
      m_RayleighVolumeName[0] = pulse.GetRayleighVolumeName();


      this->retrieve(digi, 0, system_id);
      this->retrieve(digi, 1, system_id);


      m_sinogramTheta = atan2(m_posX[0]-m_posX[1], m_posY[0]-m_posY[1]);


      G4double denom = (m_posY[0]-m_posY[1]) * (m_posY[0]-m_posY[1]) +
                       (m_posX[1]-m_posX[0]) * (m_posX[1]-m_posX[0]);

      if (denom!=0.) {
        denom = sqrt(denom);
        m_sinogramS = ( m_posX[0] * (m_posY[0]-m_posY[1]) +
                        m_posY[0] * (m_posX[1]-m_posX[0])  )
                      / denom;
      } else {
        m_sinogramS = 0.;
      }


      if (m_sinogramTheta < 0.0) {
        m_sinogramTheta = m_sinogramTheta + pi;
        m_sinogramS = -m_sinogramS;
      }

      m.second.fill();
    }
  }
}

void GateToTree::retrieve(GateCoincidenceDigi *aDigi, G4int side, G4int system_id)
{
  const auto &pulse = aDigi->GetPulse(side);
  m_eventID[side] = pulse.GetEventID();
  m_sourceID[side] = pulse.GetSourceID();
//    m_sourceID[side] = digi->GetSourceID();
  m_sourcePosX[side] = pulse.GetSourcePosition().x() / mm;
  m_sourcePosY[side] = pulse.GetSourcePosition().y() / mm;
  m_sourcePosZ[side] = pulse.GetSourcePosition().z() / mm;

  m_posX[side] = pulse.GetGlobalPos().x() / mm;
  m_posY[side] = pulse.GetGlobalPos().y() / mm;
  m_posZ[side] = pulse.GetGlobalPos().z() / mm;

  m_time[side] = pulse.GetTime() / s;
  m_edep[side] = pulse.GetEnergy() / MeV;

  m_comptonVolumeName[side] = pulse.GetComptonVolumeName();
  m_RayleighVolumeName[side] = pulse.GetRayleighVolumeName();

  m_nPhantomCompton[side] = pulse.GetNPhantomCompton();
  m_nCrystalCompton[side] = pulse.GetNCrystalCompton();
  m_nPhantomRayleigh[side] = pulse.GetNPhantomRayleigh();
  m_nCrystalRayleigh[side] = pulse.GetNCrystalRayleigh();

  if(m_coincidencesParams_to_write.at("componentsIDs").toSave())
  {
    for(auto depth = 0; depth < MAX_DEPTH_SYSTEM; ++depth)
      m_outputID[side][system_id][depth] = pulse.GetComponentID(depth);
  }
}



void GateToTree::RecordStepWithVolume(const GateVVolume *v, const G4Step *step)
{

}

void GateToTree::RecordVoxels(GateVGeometryVoxelStore *store)
{

}

const G4String &GateToTree::GiveNameOfFile()
{
//    return m_fileName;
  if(!m_listOfFileName.empty())
  {
    m_uselessFileName = "leave me";
    return m_uselessFileName;
  }

  else
  {
    m_uselessFileName = " ";
    return m_uselessFileName;
  }

}

GateToTree::~GateToTree()
{
  delete m_messenger;
}
void GateToTree::RegisterNewCoincidenceDigiCollection(const G4String &string, G4bool aBool)
{
  if(!aBool)
    return;

  m_listOfCoincidencesCollection.push_back(string);
}

void GateToTree::RegisterNewSingleDigiCollection(const G4String &string, G4bool aBool)
{
  if(!aBool)
    return;
  m_listOfSinglesCollection.push_back(string);
}

void GateToTree::SetOutputIDName(G4int id_system, const char *anOutputIDName, size_t depth)
{
//  G4cout << "GateToTree::SetOutputIDName, id_system = '" << id_system
//         << "' anOutputIDName = '" << anOutputIDName
//         << "' depth = '" << depth
//         << "'\n";

  assert(id_system < MAX_NB_SYSTEM);
  assert(depth < MAX_DEPTH_SYSTEM);

  m_max_depth_system[id_system] = depth + 1;

  strncpy(m_outputIDName[id_system][depth], anOutputIDName, MAX_OUTPUTIDNAME_SIZE);
}

void GateToTree::addFileName(const G4String &s)
{
  m_listOfFileName.push_back(s);
}

G4bool GateToTree::getHitsEnabled() const
{
  return m_hits_enabled;
}

void GateToTree::setHitsEnabled(G4bool mHitsEnabled)
{
  m_hits_enabled = mHitsEnabled;
}

void GateToTree::addCollection(const std::string &str)
{

  G4String possibleValues = "";

  for(auto &&s: m_listOfSinglesCollection)
  {

    possibleValues += "'" + s + "' (type:singles);";

    if(s == str)
    {
      GateOutputTreeFileManager m;
      for(auto &&fileName: m_listOfFileName)
      {
        auto extension = getExtension(fileName);
        auto name = removeExtension(fileName);

        G4String n_fileName = name + "." + str + "." + extension;
        m.add_file(n_fileName, extension);

      }
      m_mmanager_singles.emplace(str, std::move(m));

      return;

    }
  }


  for(auto &&s: m_listOfCoincidencesCollection)
  {
    possibleValues += "'" + s + "' (type:coincidences);";
    if(s == str)
    {
      GateOutputTreeFileManager m;
      for(auto &&fileName: m_listOfFileName)
      {
        auto extension = getExtension(fileName);
        auto name = removeExtension(fileName);

        G4String n_fileName = name + "." + str + "." + extension;
        m.add_file(n_fileName, extension);
      }
      m_mmanager_coincidences.emplace(str, std::move(m));
      return;

    }
  }

  GateError("No collection named " << str << " possible valeues are " << possibleValues);
}

std::unordered_map<std::string, SaveDataParam> &GateToTree::getHitsParamsToWrite()
{
  return m_hitsParams_to_write;
}

std::unordered_map<std::string, SaveDataParam> &GateToTree::getSinglesParamsToWrite()
{
  return m_singlesParams_to_write;
}

std::unordered_map<std::string, SaveDataParam> &GateToTree::getCoincidencesParamsToWrite()
{
  return m_coincidencesParams_to_write;
}


