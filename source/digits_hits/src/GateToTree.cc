/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

//
// Created by mdupont on 17/05/19.

//    - 2023/02/22 PDGcode for optical photon is changed from 0 to -22
//  OK GND 2022 TODO: adaptation for multiSD for OpticalOutput is not finished. Stop because of question: do we really need it ?
//    let a side for the moment (in case if needed uncomment lines 1050, 1052, 1053 in Book() method
//

#include "GateToTree.hh"

#include <cassert>


#include "G4Event.hh"
#include "G4Step.hh"
#include "G4OpticalPhoton.hh"

#include "GateToTreeMessenger.hh"

#include "GateOutputMgr.hh"
#include "GateSystemListManager.hh"
#include "GateVSystem.hh"
#include "GateMiscFunctions.hh"
#include "G4DigiManager.hh"
#include "GateDigitizerMgr.hh"

char GateToTree::m_outputIDName[GateToTree::MAX_NB_SYSTEM][GateToTree::MAX_DEPTH_SYSTEM][GateToTree::MAX_OUTPUTIDNAME_SIZE];
bool GateToTree::m_outputIDHasName[GateToTree::MAX_NB_SYSTEM][GateToTree::MAX_DEPTH_SYSTEM];
G4int GateToTree::m_max_depth_system[GateToTree::MAX_NB_SYSTEM];
const auto MAX_NB_CHARACTER = 32;


SaveDataParam::SaveDataParam() {
    m_save = true;
}

bool SaveDataParam::toSave() const {
    return m_save;
}

void SaveDataParam::setToSave(bool mSave) {
    m_save = mSave;
}


GateToTree::GateToTree(const G4String &name, GateOutputMgr *outputMgr, DigiMode digiMode) :
        GateVOutputModule(name, outputMgr, digiMode) {


    for (auto nb_system = 0; nb_system < MAX_NB_SYSTEM; ++nb_system) {
        for (auto depth = 0; depth < MAX_DEPTH_SYSTEM; ++depth) {
            m_outputIDHasName[nb_system][depth] = false;
        }
    }


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
    m_hitsParams_to_write.emplace("systemID", SaveDataParam());
    m_hitsParams_to_write.emplace("sourceType", SaveDataParam());
    m_hitsParams_to_write.emplace("decayType", SaveDataParam());
    m_hitsParams_to_write.emplace("gammaType", SaveDataParam());
    //CC
    m_hitsParams_to_write.emplace("sourceEnergy",SaveDataParam());
    m_hitsParams_to_write.emplace("sourcePDG",SaveDataParam());
    m_hitsParams_to_write.emplace("nCrystalConv",SaveDataParam());
    m_hitsParams_to_write.emplace("nCrystalCompt",SaveDataParam());
    m_hitsParams_to_write.emplace("nCrystalRayl",SaveDataParam());
    m_hitsParams_to_write.emplace("energyFinal",SaveDataParam());
    m_hitsParams_to_write.emplace("energyIniT",SaveDataParam());
    m_hitsParams_to_write.emplace("postStepProcess",SaveDataParam());


    m_opticalParams_to_write.emplace("NumScintillation", SaveDataParam());
    m_opticalParams_to_write.emplace("NumCrystalWLS", SaveDataParam());
    m_opticalParams_to_write.emplace("NumPhantomWLS", SaveDataParam());
    m_opticalParams_to_write.emplace("CrystalLastHitPos_X", SaveDataParam());
    m_opticalParams_to_write.emplace("CrystalLastHitPos_Y", SaveDataParam());
    m_opticalParams_to_write.emplace("CrystalLastHitPos_Z", SaveDataParam());
    m_opticalParams_to_write.emplace("CrystalLastHitEnergy", SaveDataParam());
    m_opticalParams_to_write.emplace("PhantomLastHitPos_X", SaveDataParam());
    m_opticalParams_to_write.emplace("PhantomLastHitPos_Y", SaveDataParam());
    m_opticalParams_to_write.emplace("PhantomLastHitPos_Z", SaveDataParam());
    m_opticalParams_to_write.emplace("PhantomLastHitEnergy", SaveDataParam());
    m_opticalParams_to_write.emplace("PhantomWLSPos_X", SaveDataParam());
    m_opticalParams_to_write.emplace("PhantomWLSPos_Y", SaveDataParam());
    m_opticalParams_to_write.emplace("PhantomWLSPos_Z", SaveDataParam());
    m_opticalParams_to_write.emplace("PhantomProcessName", SaveDataParam());
    m_opticalParams_to_write.emplace("CrystalProcessName", SaveDataParam());
    m_opticalParams_to_write.emplace("MomentumDirectionx", SaveDataParam());
    m_opticalParams_to_write.emplace("MomentumDirectiony", SaveDataParam());
    m_opticalParams_to_write.emplace("MomentumDirectionz", SaveDataParam());

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
    m_singlesParams_to_write.emplace("systemID", SaveDataParam());
    //CC
    m_singlesParams_to_write.emplace("sourceEnergy",SaveDataParam());
    m_singlesParams_to_write.emplace("sourcePDG",SaveDataParam());
    m_singlesParams_to_write.emplace("nCrystalConv",SaveDataParam());
    m_singlesParams_to_write.emplace("nCrystalCompt",SaveDataParam());
    m_singlesParams_to_write.emplace("nCrystalRayl",SaveDataParam());
    m_singlesParams_to_write.emplace("energyFinal",SaveDataParam());
    m_singlesParams_to_write.emplace("energyIni",SaveDataParam());
    m_singlesParams_to_write.emplace("localPosX", SaveDataParam());
    m_singlesParams_to_write.emplace("localPosY", SaveDataParam());
    m_singlesParams_to_write.emplace("localPosZ", SaveDataParam());

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
    m_coincidencesParams_to_write.emplace("systemID", SaveDataParam());

    m_messenger = new GateToTreeMessenger(this);
    m_hits_enabled = false;
}

void GateToTree::RecordBeginOfAcquisition() {
    if (!this->IsEnabled())
        return;

   /* if (m_hits_enabled) {
        for (auto &&fileName: m_listOfFileName) {
            auto extension = getExtension(fileName);
            auto name = removeExtension(fileName);

            G4String hits_filename = name + ".hits." + extension;
            m_manager_hits.add_file(hits_filename, extension);
        }
        */
        //OK GND 2022
	for (auto &&m: m_mmanager_hits)
	{

		auto &mm = m.second;

        if (m_hitsParams_to_write.at("PDGEncoding").toSave())
            mm.write_variable("PDGEncoding", &m_PDGEncoding);

        if (m_hitsParams_to_write.at("trackID").toSave())
            mm.write_variable("trackID", &m_trackID);

        if (m_hitsParams_to_write.at("parentID").toSave())
            mm.write_variable("parentID", &m_parentID);

        if (m_hitsParams_to_write.at("trackLocalTime").toSave())
            mm.write_variable("trackLocalTime", &m_trackLocalTime);

        if (m_hitsParams_to_write.at("time").toSave())
            mm.write_variable("time", &m_time[0]);

        if (m_hitsParams_to_write.at("runID").toSave())
            mm.write_variable("runID", &m_runID);

        if (m_hitsParams_to_write.at("eventID").toSave())
            mm.write_variable("eventID", &m_eventID[0]);

        if (m_hitsParams_to_write.at("sourceID").toSave())
            mm.write_variable("sourceID", &m_sourceID[0]);

        if (m_hitsParams_to_write.at("primaryID").toSave())
            mm.write_variable("primaryID", &m_primaryID);

        if (m_hitsParams_to_write.at("posX").toSave())
            mm.write_variable("posX", &m_posX[0]);

        if (m_hitsParams_to_write.at("posY").toSave())
            mm.write_variable("posY", &m_posY[0]);

        if (m_hitsParams_to_write.at("posZ").toSave())
            mm.write_variable("posZ", &m_posZ[0]);

        if (m_hitsParams_to_write.at("localPosX").toSave())
            mm.write_variable("localPosX", &m_localPosX);

        if (m_hitsParams_to_write.at("localPosY").toSave())
            mm.write_variable("localPosY", &m_localPosY);

        if (m_hitsParams_to_write.at("localPosZ").toSave())
            mm.write_variable("localPosZ", &m_localPosZ);

        if (m_hitsParams_to_write.at("momDirX").toSave())
            mm.write_variable("momDirX", &m_momDirX);

        if (m_hitsParams_to_write.at("momDirY").toSave())
            mm.write_variable("momDirY", &m_momDirY);

        if (m_hitsParams_to_write.at("momDirZ").toSave())
            mm.write_variable("momDirZ", &m_momDirZ);

        if (m_hitsParams_to_write.at("edep").toSave())
            mm.write_variable("edep", &m_edep[0]);

        if (m_hitsParams_to_write.at("stepLength").toSave())
            mm.write_variable("stepLength", &m_stepLength);

        if (m_hitsParams_to_write.at("trackLength").toSave())
            mm.write_variable("trackLength", &m_trackLength);

        if (m_hitsParams_to_write.at("rotationAngle").toSave())
            mm.write_variable("rotationAngle", &m_rotationAngle);

        if (m_hitsParams_to_write.at("axialPos").toSave())
            mm.write_variable("axialPos", &m_axialPos);

        if (m_hitsParams_to_write.at("processName").toSave())
            mm.write_variable("processName", &m_processName, MAX_NB_CHARACTER);

        if (m_hitsParams_to_write.at("comptVolName").toSave())
            mm.write_variable("comptVolName", &m_comptonVolumeName[0], MAX_NB_CHARACTER);

        if (m_hitsParams_to_write.at("RayleighVolName").toSave())
            mm.write_variable("RayleighVolName", &m_RayleighVolumeName[0], MAX_NB_CHARACTER);

        if (m_hitsParams_to_write.at("volumeIDs").toSave()) {
            for (auto i = 0; i < VOLUMEID_SIZE; ++i) {
                std::stringstream ss;
                ss << "volumeID[" << i << "]";
                mm.write_variable(ss.str(), &m_volumeID[i]);
            }
        }

        if (m_hitsParams_to_write.at("sourcePosX").toSave())
            mm.write_variable("sourcePosX", &m_sourcePosX[0]);
        if (m_hitsParams_to_write.at("sourcePosY").toSave())
            mm.write_variable("sourcePosY", &m_sourcePosY[0]);
        if (m_hitsParams_to_write.at("sourcePosZ").toSave())
            mm.write_variable("sourcePosZ", &m_sourcePosZ[0]);

        if (m_hitsParams_to_write.at("nPhantomCompton").toSave())
            mm.write_variable("nPhantomCompton", &m_nPhantomCompton[0]);

        if (m_hitsParams_to_write.at("nCrystalCompton").toSave())
            mm.write_variable("nCrystalCompton", &m_nCrystalCompton[0]);

        if (m_hitsParams_to_write.at("nPhantomRayleigh").toSave())
            mm.write_variable("nPhantomRayleigh", &m_nPhantomRayleigh[0]);

        if (m_hitsParams_to_write.at("nCrystalRayleigh").toSave())
            mm.write_variable("nCrystalRayleigh", &m_nCrystalRayleigh[0]);


        if (m_hitsParams_to_write.at("componentsIDs").toSave()) {

            if (GateSystemListManager::GetInstance()->size() == 1) {
                int k = 0;
                for (auto depth = 0; depth < m_max_depth_system[k]; ++depth) {
                    std::stringstream ss;
                    ss << m_outputIDName[k][depth];
                    mm.write_variable(ss.str(), &m_outputID[0][k][depth]);
                }
            } else {
                for (unsigned int k = 0; k < GateSystemListManager::GetInstance()->size(); ++k) {
                    auto system = GateSystemListManager::GetInstance()->GetSystem(k);

                    for (auto depth = 0; depth < m_max_depth_system[k]; ++depth) {
                        if (!m_outputIDHasName[k][depth])
                            continue;
                        std::stringstream ss;
                        ss << system->GetOwnName() << "/" << m_outputIDName[k][depth];
                        mm.write_variable(ss.str(), &m_outputID[0][k][depth]);
                    }
                }
            }
        }

        if (m_hitsParams_to_write.at("photonID").toSave())
            mm.write_variable("photonID", &m_photonID);

        if (m_hitsParams_to_write.at("systemID").toSave() && GateSystemListManager::GetInstance()->size() > 1)
            mm.write_variable("systemID", &m_systemID);

        if (m_hitsParams_to_write.at("sourceType").toSave())
            mm.write_variable("sourceType", &m_sourceType);

        if (m_hitsParams_to_write.at("decayType").toSave())
            mm.write_variable("decayType", &m_decayType);

        if (m_hitsParams_to_write.at("gammaType").toSave())
            mm.write_variable("gammaType", &m_gammaType);

        if(m_cc_enabled)
        {
		   if (m_hitsParams_to_write.at("sourceEnergy").toSave())
			   mm.write_variable("sourceEnergy", &m_sourceEnergy );
		   if (m_hitsParams_to_write.at("sourcePDG").toSave())
			   mm.write_variable("sourcePDG", &m_sourcePDG);
		   if (m_hitsParams_to_write.at("nCrystalConv").toSave())
			   mm.write_variable("nCrystalConv", &m_nCrystalConv);
		   if (m_hitsParams_to_write.at("nCrystalCompt").toSave())
			   mm.write_variable("nCrystalCompt", &m_nCrystalCompt);
		   if (m_hitsParams_to_write.at("nCrystalRayl").toSave())
			   mm.write_variable("nCrystalRayl", &m_nCrystalRayl);

		   if (m_hitsParams_to_write.at("energyFinal").toSave())
			   mm.write_variable("energyFin", &m_energyFin);
		   if (m_hitsParams_to_write.at("energyIniT").toSave())
			   mm.write_variable("energyIniT", &m_energyIniT);
		   if (m_hitsParams_to_write.at("postStepProcess").toSave())
			   mm.write_variable("postStepProcess", &m_postStepProcess, MAX_NB_CHARACTER);
        }


        mm.write_header();
    }


   /* if (m_opticalData_enabled) {
        for (auto &&fileName: m_listOfFileName) {
            auto extension = getExtension(fileName);
            auto name = removeExtension(fileName);

            G4String hits_filename = name + ".optical." + extension;
            m_manager_optical.add_file(hits_filename, extension);
        }
*/
	//OK GND 2022
    for (auto &&m: m_mmanager_optical)
 	 {

	 auto &mm = m.second;
        if (m_opticalParams_to_write.at("NumScintillation").toSave())
            mm.write_variable("NumScintillation", &m_nScintillation);

        if (m_opticalParams_to_write.at("NumCrystalWLS").toSave())
            mm.write_variable("NumCrystalWLS", &m_NumCrystalWLS);
        if (m_opticalParams_to_write.at("NumPhantomWLS").toSave())
            mm.write_variable("NumPhantomWLS", &m_NumPhantomWLS);

        if (m_opticalParams_to_write.at("CrystalLastHitPos_X").toSave())
            mm.write_variable("CrystalLastHitPos_X", &m_CrystalLastHitPos_X);
        if (m_opticalParams_to_write.at("CrystalLastHitPos_Y").toSave())
            mm.write_variable("CrystalLastHitPos_Y", &m_CrystalLastHitPos_Y);
        if (m_opticalParams_to_write.at("CrystalLastHitPos_Z").toSave())
            mm.write_variable("CrystalLastHitPos_Z", &m_CrystalLastHitPos_Z);
        if (m_opticalParams_to_write.at("CrystalLastHitEnergy").toSave())
            mm.write_variable("CrystalLastHitEnergy", &m_CrystalLastHitEnergy);

        if (m_opticalParams_to_write.at("PhantomLastHitPos_X").toSave())
            mm.write_variable("PhantomLastHitPos_X", &m_PhantomLastHitPos_X);
        if (m_opticalParams_to_write.at("PhantomLastHitPos_Y").toSave())
            mm.write_variable("PhantomLastHitPos_Y", &m_PhantomLastHitPos_Y);
        if (m_opticalParams_to_write.at("PhantomLastHitPos_Z").toSave())
            mm.write_variable("PhantomLastHitPos_Z", &m_PhantomLastHitPos_Z);
        if (m_opticalParams_to_write.at("PhantomLastHitEnergy").toSave())
            mm.write_variable("PhantomLastHitEnergy", &m_PhantomLastHitEnergy);

        if (m_opticalParams_to_write.at("PhantomWLSPos_X").toSave())
            mm.write_variable("PhantomWLSPos_X", &m_PhantomWLSPos_X);
        if (m_opticalParams_to_write.at("PhantomWLSPos_Y").toSave())
            mm.write_variable("PhantomWLSPos_Y", &m_PhantomWLSPos_Y);
        if (m_opticalParams_to_write.at("PhantomWLSPos_Z").toSave())
            mm.write_variable("PhantomWLSPos_Z", &m_PhantomWLSPos_Z);

        if (m_opticalParams_to_write.at("PhantomProcessName").toSave())
            mm.write_variable("PhantomProcessName", &m_NameOfProcessInPhantom, MAX_NB_CHARACTER);
        if (m_opticalParams_to_write.at("CrystalProcessName").toSave())
            mm.write_variable("CrystalProcessName", &m_NameOfProcessInCrystal, MAX_NB_CHARACTER);

        if (m_opticalParams_to_write.at("MomentumDirectionx").toSave())
            mm.write_variable("MomentumDirectionx", &m_MomentumDirectionx);
        if (m_opticalParams_to_write.at("MomentumDirectiony").toSave())
            mm.write_variable("MomentumDirectiony", &m_MomentumDirectiony);
        if (m_opticalParams_to_write.at("MomentumDirectionz").toSave())
            mm.write_variable("MomentumDirectionz", &m_MomentumDirectionz);

        mm.write_header();

    }

    for (auto &&m: m_mmanager_singles) {

        auto &mm = m.second;

        if (m_singlesParams_to_write.at("runID").toSave())
            mm.write_variable("runID", &m_runID);

        if (m_singlesParams_to_write.at("eventID").toSave())
            mm.write_variable("eventID", &m_eventID[0]);

        if (m_singlesParams_to_write.at("sourceID").toSave())
            mm.write_variable("sourceID", &m_sourceID[0]);

        if (m_singlesParams_to_write.at("sourcePosX").toSave())
            mm.write_variable("sourcePosX", &m_sourcePosX[0]);
        if (m_singlesParams_to_write.at("sourcePosY").toSave())
            mm.write_variable("sourcePosY", &m_sourcePosY[0]);
        if (m_singlesParams_to_write.at("sourcePosZ").toSave())
            mm.write_variable("sourcePosZ", &m_sourcePosZ[0]);

        if (m_singlesParams_to_write.at("globalPosX").toSave())
            mm.write_variable("globalPosX", &m_posX[0]);
        if (m_singlesParams_to_write.at("globalPosY").toSave())
            mm.write_variable("globalPosY", &m_posY[0]);
        if (m_singlesParams_to_write.at("globalPosZ").toSave())
            mm.write_variable("globalPosZ", &m_posZ[0]);

        if (m_singlesParams_to_write.at("componentsIDs").toSave()) {
            if (GateSystemListManager::GetInstance()->size() == 1) {
                int k = 0;
                for (auto depth = 0; depth < m_max_depth_system[k]; ++depth) {
                    std::stringstream ss;
                    ss << m_outputIDName[k][depth];
                    mm.write_variable(ss.str(), &m_outputID[0][k][depth]);
                }
            } else {
                for (unsigned int k = 0; k < GateSystemListManager::GetInstance()->size(); ++k) {
                    auto system = GateSystemListManager::GetInstance()->GetSystem(k);
                    for (auto depth = 0; depth < m_max_depth_system[k]; ++depth) {
                        if (!m_outputIDHasName[k][depth])
                            continue;
                        std::stringstream ss;
                        ss << system->GetOwnName() << "/" << m_outputIDName[k][depth];
                        mm.write_variable(ss.str(), &m_outputID[0][k][depth]);
                    }
                }
            }
        }


        if (m_singlesParams_to_write.at("time").toSave())
            mm.write_variable("time", &m_time[0]);
        if (m_singlesParams_to_write.at("energy").toSave())
            mm.write_variable("energy", &m_edep[0]);

        if (m_singlesParams_to_write.at("comptonPhantom").toSave())
            mm.write_variable("comptonPhantom", &m_nPhantomCompton[0]);
        if (m_singlesParams_to_write.at("comptonCrystal").toSave())
            mm.write_variable("comptonCrystal", &m_nCrystalCompton[0]);
        if (m_singlesParams_to_write.at("RayleighPhantom").toSave())
            mm.write_variable("RayleighPhantom", &m_nPhantomRayleigh[0]);
        if (m_singlesParams_to_write.at("RayleighCrystal").toSave())
            mm.write_variable("RayleighCrystal", &m_nCrystalRayleigh[0]);

        if (m_singlesParams_to_write.at("comptVolName").toSave())
            mm.write_variable("comptVolName", &m_comptonVolumeName[0], MAX_NB_CHARACTER);
        if (m_singlesParams_to_write.at("RayleighVolName").toSave())
            mm.write_variable("RayleighVolName", &m_RayleighVolumeName[0], MAX_NB_CHARACTER);

        if (m_singlesParams_to_write.at("rotationAngle").toSave())
            mm.write_variable("rotationAngle", &m_rotationAngle);
        if (m_singlesParams_to_write.at("axialPos").toSave())
            mm.write_variable("axialPos", &m_axialPos);

        if (m_singlesParams_to_write.at("systemID").toSave() && GateSystemListManager::GetInstance()->size() > 1)
            mm.write_variable("systemID", &m_systemID);

        if(m_cc_enabled)
           {
        	if (m_singlesParams_to_write.at("sourceEnergy").toSave())
			   mm.write_variable("sourceEnergy", &m_sourceEnergy );
        	if (m_singlesParams_to_write.at("sourcePDG").toSave())
			   mm.write_variable("sourcePDG", &m_sourcePDG);
        	if (m_singlesParams_to_write.at("nCrystalConv").toSave())
			   mm.write_variable("nCrystalConv", &m_nCrystalConv);
        	if (m_singlesParams_to_write.at("nCrystalCompt").toSave())
			   mm.write_variable("nCrystalCompt", &m_nCrystalCompt);
        	if (m_singlesParams_to_write.at("nCrystalRayl").toSave())
			   mm.write_variable("nCrystalRayl", &m_nCrystalRayl);

        	if (m_singlesParams_to_write.at("energyFinal").toSave())
			   mm.write_variable("energyFin", &m_energyFin);
        	if (m_singlesParams_to_write.at("energyIni").toSave())
			   mm.write_variable("energyIni", &m_energyIni);

        	if (m_singlesParams_to_write.at("localPosX").toSave())
        		mm.write_variable("localPosX", &m_localPosX);
        	if (m_singlesParams_to_write.at("localPosY").toSave())
        		mm.write_variable("localPosY", &m_localPosY);
        	if (m_singlesParams_to_write.at("localPosZ").toSave())
        		mm.write_variable("localPosZ", &m_localPosZ);

           }

        mm.write_header();
    }

    for (auto &&m: m_mmanager_coincidences) {

        auto &mm = m.second;
        if (m_coincidencesParams_to_write.at("runID").toSave())
            mm.write_variable("runID", &m_runID);

        if (m_coincidencesParams_to_write.at("eventID").toSave()) {
            mm.write_variable("eventID1", &m_eventID[0]);
            mm.write_variable("eventID2", &m_eventID[1]);
        }

        if (m_coincidencesParams_to_write.at("sourceID").toSave()) {
            mm.write_variable("sourceID1", &m_sourceID[0]);
            mm.write_variable("sourceID2", &m_sourceID[1]);
        }

        if (m_coincidencesParams_to_write.at("sourcePosX").toSave()) {
            mm.write_variable("sourcePosX1", &m_sourcePosX[0]);
            mm.write_variable("sourcePosX2", &m_sourcePosX[1]);
        }
        if (m_coincidencesParams_to_write.at("sourcePosY").toSave()) {
            mm.write_variable("sourcePosY1", &m_sourcePosY[0]);
            mm.write_variable("sourcePosY2", &m_sourcePosY[1]);
        }

        if (m_coincidencesParams_to_write.at("sourcePosZ").toSave()) {
            mm.write_variable("sourcePosZ1", &m_sourcePosZ[0]);
            mm.write_variable("sourcePosZ2", &m_sourcePosZ[1]);
        }


        if (m_coincidencesParams_to_write.at("rotationAngle").toSave())
            mm.write_variable("rotationAngle", &m_rotationAngle);

        if (m_coincidencesParams_to_write.at("axialPos").toSave())
            mm.write_variable("axialPos", &m_axialPos);

        if (m_coincidencesParams_to_write.at("globalPosX").toSave()) {
            mm.write_variable("globalPosX1", &m_posX[0]);
            mm.write_variable("globalPosX2", &m_posX[1]);
        }

        if (m_coincidencesParams_to_write.at("globalPosY").toSave()) {
            mm.write_variable("globalPosY1", &m_posY[0]);
            mm.write_variable("globalPosY2", &m_posY[1]);
        }

        if (m_coincidencesParams_to_write.at("globalPosZ").toSave()) {
            mm.write_variable("globalPosZ1", &m_posZ[0]);
            mm.write_variable("globalPosZ2", &m_posZ[1]);
        }

        if (m_coincidencesParams_to_write.at("time").toSave()) {
            mm.write_variable("time1", &m_time[0]);
            mm.write_variable("time2", &m_time[1]);
        }

        if (m_coincidencesParams_to_write.at("energy").toSave()) {
            mm.write_variable("energy1", &m_edep[0]);
            mm.write_variable("energy2", &m_edep[1]);
        }

        if (m_coincidencesParams_to_write.at("comptVolName").toSave()) {
            mm.write_variable("comptVolName1", &m_comptonVolumeName[0], MAX_NB_CHARACTER);
            mm.write_variable("comptVolName2", &m_comptonVolumeName[1], MAX_NB_CHARACTER);
        }

        if (m_coincidencesParams_to_write.at("RayleighVolName").toSave()) {
            mm.write_variable("RayleighVolName1", &m_RayleighVolumeName[0], MAX_NB_CHARACTER);
            mm.write_variable("RayleighVolName2", &m_RayleighVolumeName[1], MAX_NB_CHARACTER);
        }

        if (m_coincidencesParams_to_write.at("comptonPhantom").toSave()) {
            mm.write_variable("comptonPhantom1", &m_nPhantomCompton[0]);
            mm.write_variable("comptonPhantom2", &m_nPhantomCompton[1]);
        }
        if (m_coincidencesParams_to_write.at("comptonCrystal").toSave()) {
            mm.write_variable("comptonCrystal1", &m_nCrystalCompton[0]);
            mm.write_variable("comptonCrystal2", &m_nCrystalCompton[1]);
        }
        if (m_coincidencesParams_to_write.at("RayleighPhantom").toSave()) {
            mm.write_variable("RayleighPhantom1", &m_nPhantomRayleigh[0]);
            mm.write_variable("RayleighPhantom2", &m_nPhantomRayleigh[1]);
        }
        if (m_coincidencesParams_to_write.at("RayleighCrystal").toSave()) {
            mm.write_variable("RayleighCrystal1", &m_nCrystalRayleigh[0]);
            mm.write_variable("RayleighCrystal2", &m_nCrystalRayleigh[1]);
        }


        if (m_coincidencesParams_to_write.at("componentsIDs").toSave()) {
            if (GateSystemListManager::GetInstance()->size() == 1) {
                int k = 0;
                for (auto side = 1; side <= 2; ++side) {
                    for (auto depth = 0; depth < m_max_depth_system[k]; ++depth) {
                        std::stringstream ss;
                        ss << m_outputIDName[k][depth] << side;
                        mm.write_variable(ss.str(), &m_outputID[side - 1][k][depth]);
                    }
                }

            } else {
                for (auto side = 1; side <= 2; ++side) {
                    for (unsigned int k = 0; k < GateSystemListManager::GetInstance()->size(); ++k) {
                        auto system = GateSystemListManager::GetInstance()->GetSystem(k);
                        for (auto depth = 0; depth < m_max_depth_system[k]; ++depth) {
                            if (!m_outputIDHasName[k][depth])
                                continue;
                            std::stringstream ss;
                            ss << system->GetOwnName() << "/" << m_outputIDName[k][depth] << side;
                            mm.write_variable(ss.str(), &m_outputID[side - 1][k][depth]);
                        }
                    }
                }
            }
        }

        if (m_coincidencesParams_to_write.at("systemID").toSave() && GateSystemListManager::GetInstance()->size() > 1)
            mm.write_variable("systemID", &m_systemID);

        if (m_coincidencesParams_to_write.at("sinogramTheta").toSave())
            mm.write_variable("sinogramTheta", &m_sinogramTheta);

        if (m_coincidencesParams_to_write.at("sinogramS").toSave())
            mm.write_variable("sinogramS", &m_sinogramS);
        mm.write_header();
    }


}

void GateToTree::RecordEndOfAcquisition() {
    //m_manager_hits.close();
    //m_manager_optical.close();

	//OK GND 2022

	for (auto &&m: m_mmanager_hits)
            	m.second.close();

	for (auto &&m: m_mmanager_optical)
	            	m.second.close();

	for (auto &&m: m_mmanager_singles)
        m.second.close();

    for (auto &&m: m_mmanager_coincidences)
        m.second.close();

}

void GateToTree::RecordBeginOfRun(const G4Run *run) {
    UNUSED(run);
}

void GateToTree::RecordEndOfRun(const G4Run *run) {
    UNUSED(run);
}

void GateToTree::RecordBeginOfEvent(const G4Event *event) {
    UNUSED(event);
}

void GateToTree::RecordEndOfEvent(const G4Event *event) {

	//OK GND 2022
	auto fDM = G4DigiManager::GetDMpointer();

	 if (!m_hits_to_collectionID.size()) {
	        for (auto &&m: m_mmanager_hits) {
	            auto str = m.first+"Collection";
	            //str.erase(remove_if(str.begin(), str.end(), '_'), str.end());
	            auto collectionID = fDM->GetHitsCollectionID(str);
	            m_hits_to_collectionID.emplace(m.first, collectionID);
	        }
	    }

	 if (!m_optical_to_collectionID.size()) {
	        for (auto &&m:m_mmanager_optical) {
	            auto str = m.first+"Collection";
	            //str.erase(remove_if(str.begin(), str.end(), '_'), str.end());
	            auto collectionID = fDM->GetHitsCollectionID(str);
	            m_optical_to_collectionID.emplace(m.first, collectionID);
	        }
	    }



	 for (auto &&m: m_mmanager_hits)
	 {
		 auto collectionID = m_hits_to_collectionID.at(m.first);
		 const GateHitsCollection *CHC =
				 (GateHitsCollection *) (fDM->GetHitsCollection(collectionID));

		 if (!CHC) {
			std::cout <<
					  "GateToTree::RecordEndOfEvent : GateHitCollection not found"
					  << Gateendl;
			continue;
		}

		 assert(CHC);
		 auto v = CHC->GetVector();
		 assert(v);

	//    auto writeComptonVolumeName = m_hitsParams_to_write.at("comptVolName").toSave();
	//    auto writeRayleighVolName = m_hitsParams_to_write.at("RayleighVolName").toSave();
	//    auto writeVolumeIDs = m_hitsParams_to_write.at("volumeIDs").toSave();

		m_systemID = -1;
	   // for (unsigned int iHit = 0; iHit < CHC->entries(); ++iHit) {
		//    auto hit = (*CHC)[iHit];
		for (auto &&hit: *v)
		{
			if (!hit->GoodForAnalysis())
				continue;

			m_systemID = hit->GetSystemID();
			retrieve(hit, m_systemID);

			if (m_hitsParams_to_write.at("componentsIDs").toSave()) {
				for (auto depth = 0; depth < MAX_DEPTH_SYSTEM; ++depth)
					m_outputID[0][m_systemID][depth] = hit->GetComponentID(depth);
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

			if (m_hitsParams_to_write.at("volumeIDs").toSave())
				hit->GetVolumeID().StoreDaughterIDs(m_volumeID, VOLUMEID_SIZE);

			m_photonID = hit->GetPhotonID();

			m_sourceType = hit->GetSourceType();
			m_decayType = hit->GetDecayType();
			m_gammaType = hit->GetGammaType();

			if(m_cc_enabled)
			{
				m_sourceEnergy=hit->GetSourceEnergy();
				m_sourcePDG=hit->GetSourcePDG();
				m_nCrystalConv=hit->GetNCrystalConv();
				m_nCrystalCompt=hit->GetNCrystalCompton();
				m_nCrystalRayl=hit->GetNCrystalRayleigh();

				m_postStepProcess=hit->GetPostStepProcess();
				m_energyFin=hit->GetEnergyFin();
				m_energyIniT=hit->GetEnergyIniTrack();
			}

			 m.second.fill();
			//m_manager_hits.fill();
		}
	 }
    //auto fDM = G4DigiManager::GetDMpointer();

    if (!m_singles_to_collectionID.size()) {
        for (auto &&m: m_mmanager_singles) {
           // auto collectionID = fDM->GetDigiCollectionID(m.first);
        	// OK GND 2022
        	auto collectionID = GetCollectionID(m.first);
        	m_singles_to_collectionID.emplace(m.first, collectionID);
        }
    }

    for (auto &&m: m_mmanager_singles) {
//        auto collectionID = fDM->GetDigiCollectionID(m.first);
        auto collectionID = m_singles_to_collectionID.at(m.first);
//        auto SDC = static_cast<const GateDigiCollection*>(fDM->GetDigiCollection(collectionID));
        const GateDigiCollection *SDC =
                        (GateDigiCollection *) (fDM->GetDigiCollection(collectionID));

        if (!SDC) {
//            G4cout << "GateToTree::RecordEndOfEvent no collection = " << m.first << "\n";
            continue;
        }

        assert(SDC);
        auto v = SDC->GetVector();
        assert(v);

        m_systemID = -1;


        for (auto &&digi: *v) {
            if (m_systemID == -1) {
                auto mother = static_cast<const GateHit *>(digi->GetMother());
                if (!mother) {
                    // for example pulse created by GateNoise.cc (l67).
                    m_systemID = 0;
                } else {
                    m_systemID = mother->GetSystemID();
                }
            }

            retrieve(digi, m_systemID);
            m_comptonVolumeName[0] = digi->GetComptonVolumeName();
            m_RayleighVolumeName[0] = digi->GetRayleighVolumeName();
            if (m_singlesParams_to_write.at("componentsIDs").toSave()) {
                for (auto depth = 0; depth < MAX_DEPTH_SYSTEM; ++depth)
                    m_outputID[0][m_systemID][depth] = digi->GetComponentID(depth);
            }

            m_edep[0] = digi->GetEnergy() / MeV;
            m.second.fill();
        }

    }

    if (!m_coincidences_to_collectionID.size()) {
        for (auto &&m: m_mmanager_coincidences) {
            auto collectionID = fDM->GetDigiCollectionID(m.first);
            m_coincidences_to_collectionID.emplace(m.first, collectionID);
        }
    }

    for (auto &&m: m_mmanager_coincidences) {
        auto collectionID = m_coincidences_to_collectionID.at(m.first);
        auto SDC = static_cast<const GateCoincidenceDigiCollection *>(fDM->GetDigiCollection(collectionID));

        if (!SDC) {
            continue;
        }

        assert(SDC);
        auto v = SDC->GetVector();
        assert(v);

        m_systemID = -1;


        for (auto &&coin_digi: *v) {
            if (m_systemID == -1) {
                auto mother = static_cast<const GateHit *>(coin_digi->GetDigi(0)->GetMother());
                if (!mother) {
                    // for example pulse created by GateNoise.cc (l67).
                    m_systemID = 0;
                } else {
                    m_systemID = mother->GetSystemID();
                }
            }

            const auto digi = coin_digi->GetDigi(0);
            retrieve(digi, m_systemID);
            m_comptonVolumeName[0] = digi->GetComptonVolumeName();
            m_RayleighVolumeName[0] = digi->GetRayleighVolumeName();


            this->retrieve(coin_digi, 0, m_systemID);
            this->retrieve(coin_digi, 1, m_systemID);


            m_sinogramTheta = atan2(m_posX[0] - m_posX[1], m_posY[0] - m_posY[1]);


            G4double denom = (m_posY[0] - m_posY[1]) * (m_posY[0] - m_posY[1]) +
                             (m_posX[1] - m_posX[0]) * (m_posX[1] - m_posX[0]);

            if (denom != 0.) {
                denom = sqrt(denom);
                m_sinogramS = (m_posX[0] * (m_posY[0] - m_posY[1]) +
                               m_posY[0] * (m_posX[1] - m_posX[0]))
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

    RecordOpticalData(event);
}

void GateToTree::retrieve(GateCoincidenceDigi *aDigi, G4int side, G4int system_id) {
	const auto &signleDigi = aDigi->GetDigi(side);
    m_eventID[side] = signleDigi->GetEventID();
    m_sourceID[side] = signleDigi->GetSourceID();
 //    m_sourceID[side] = digi->GetSourceID();
     m_sourcePosX[side] = signleDigi->GetSourcePosition().x() / mm;
     m_sourcePosY[side] = signleDigi->GetSourcePosition().y() / mm;
     m_sourcePosZ[side] = signleDigi->GetSourcePosition().z() / mm;

     m_posX[side] = signleDigi->GetGlobalPos().x() / mm;
     m_posY[side] = signleDigi->GetGlobalPos().y() / mm;
     m_posZ[side] = signleDigi->GetGlobalPos().z() / mm;

     m_time[side] = signleDigi->GetTime() / s;
     m_edep[side] = signleDigi->GetEnergy() / MeV;

     m_comptonVolumeName[side] = signleDigi->GetComptonVolumeName();
     m_RayleighVolumeName[side] = signleDigi->GetRayleighVolumeName();

     m_nPhantomCompton[side] = signleDigi->GetNPhantomCompton();
     m_nCrystalCompton[side] = signleDigi->GetNCrystalCompton();
     m_nPhantomRayleigh[side] = signleDigi->GetNPhantomRayleigh();
     m_nCrystalRayleigh[side] = signleDigi->GetNCrystalRayleigh();

     if (m_coincidencesParams_to_write.at("componentsIDs").toSave()) {
         for (auto depth = 0; depth < MAX_DEPTH_SYSTEM; ++depth)
             m_outputID[side][system_id][depth] = signleDigi->GetComponentID(depth);
     }
}

void GateToTree::RecordStepWithVolume(const GateVVolume *v, const G4Step *aStep) {
    UNUSED(v);
    auto partDef = aStep->GetTrack()->GetDefinition();
    if (partDef == G4OpticalPhoton::OpticalPhotonDefinition()) {
        G4ThreeVector momentumDirection = aStep->GetTrack()->GetMomentumDirection();
        m_MomentumDirectionx = momentumDirection.x();
        m_MomentumDirectiony = momentumDirection.y();
        m_MomentumDirectionz = momentumDirection.z();
    }

}

void GateToTree::RecordVoxels(GateVGeometryVoxelStore *store) {
    UNUSED(store);
}

const G4String &GateToTree::GiveNameOfFile() {
//    return m_fileName;
    if (!m_listOfFileName.empty()) {
        m_uselessFileName = "leave me";
        return m_uselessFileName;
    } else {
        m_uselessFileName = " ";
        return m_uselessFileName;
    }

}

GateToTree::~GateToTree() {
    delete m_messenger;
}

void GateToTree::RegisterNewCoincidenceDigiCollection(const G4String &string, G4bool aBool) {
	 //OK GND 2022 TODO uncomment??
	//if (!aBool)
     //   return;

    m_listOfCoincidencesCollection.push_back(string);
}

void GateToTree::RegisterNewSingleDigiCollection(const G4String &string, G4bool aBool) {
	 //OK GND 2022 TODO uncomment ??
	//if (!aBool)
     //   return;
    m_listOfSinglesCollection.push_back(string);
}

void GateToTree::RegisterNewHitsCollection(const G4String &string, G4bool aBool) {
    //OK GND 2022 TODO uncomment ??
	//if (!aBool)
     //   return;
    m_listOfHitsCollection.push_back(string);
}

void GateToTree::SetOutputIDName(G4int id_system, const char *anOutputIDName, size_t depth) {
    /*G4cout << "GateToTree::SetOutputIDName, id_system = '" << id_system
           << "' anOutputIDName = '" << anOutputIDName
           << "' depth = '" << depth
           << "'\n";
           */

    assert(id_system < MAX_NB_SYSTEM);
    assert(depth < MAX_DEPTH_SYSTEM);


    m_max_depth_system[id_system] = depth + 1;

    strncpy(m_outputIDName[id_system][depth], anOutputIDName, MAX_OUTPUTIDNAME_SIZE);
    m_outputIDHasName[id_system][depth] = true;
}

void GateToTree::addFileName(const G4String &s) {
    m_listOfFileName.push_back(s);
}

G4bool GateToTree::getHitsEnabled() const {
    return m_hits_enabled;
}

void GateToTree::setHitsEnabled(G4bool mHitsEnabled) {
    m_hits_enabled = mHitsEnabled;

    //OK GND 2022 multiSD
      GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

    	  for (size_t i=0; i< digitizerMgr->m_SDlist.size();i++)
    	  {
    		  addHitsCollection(digitizerMgr->m_SDlist[i]->GetName());
    	  }



}

//OK GND 2022
void GateToTree::addHitsCollection(const std::string &str) {
	GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

    G4String possibleValues = "";

    for (auto &&s: m_listOfHitsCollection) {

        possibleValues += "'" + s + "' (type:hits);";

        auto str_tmp=str+"Collection";
        if (s == str_tmp) {
            GateOutputTreeFileManager m;
            for (auto &&fileName: m_listOfFileName) {
                auto extension = getExtension(fileName);
                auto name = removeExtension(fileName);
                G4String n_fileName;
                if (digitizerMgr->m_SDlist.size() ==1 )
                	 n_fileName = name + ".hits." + extension;
                else
                	 n_fileName = name + ".hits_" + str + "." + extension;
                m.add_file(n_fileName, extension);

            }
            m_mmanager_hits.emplace(str, std::move(m));
         return;

        }
    }
}


//OK GND 2022
void GateToTree::addOpticalCollection(const std::string &str) {
  //	GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

    G4String possibleValues = "";

    for (auto &&s: m_listOfHitsCollection) {

        possibleValues += "'" + s + "' (type:hits);";

        auto str_tmp=str+"Collection";
        if (s == str_tmp) {
            GateOutputTreeFileManager m;
            for (auto &&fileName: m_listOfFileName) {
                auto extension = getExtension(fileName);
                auto name = removeExtension(fileName);
                G4String n_fileName;
               // if (digitizerMgr->m_SDlist.size() ==1 )
                	n_fileName = name + ".optical." + extension;
                //else
                //	n_fileName = name + ".optical_" + str + "." + extension;
                m.add_file(n_fileName, extension);

            }
            m_mmanager_optical.emplace(str, std::move(m));
         return;

        }
    }
}




void GateToTree::addCollection(const std::string &str) {
	GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

    G4String possibleValues = "";

    for (auto &&s: m_listOfSinglesCollection) {

        possibleValues += "'" + s + "' (type:singles);";

        if (s == str) {
            GateOutputTreeFileManager m;
            for (auto &&fileName: m_listOfFileName) {
                auto extension = getExtension(fileName);
                auto name = removeExtension(fileName);
                G4String n_fileName;
                if (digitizerMgr->m_SDlist.size() ==1 )
                {
                	std::string tmp_str = str.substr(0, str.find("_"));
                	n_fileName = name + "."+tmp_str+"." + extension;
                }
                else
                	n_fileName = name + "." + str + "." + extension;
                m.add_file(n_fileName, extension);

            }
            m_mmanager_singles.emplace(str, std::move(m));

            return;

        }
    }


    for (auto &&s: m_listOfCoincidencesCollection) {
        possibleValues += "'" + s + "' (type:coincidences);";
        if (s == str) {
            GateOutputTreeFileManager m;
            for (auto &&fileName: m_listOfFileName) {
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

std::unordered_map<std::string, SaveDataParam> &GateToTree::getHitsParamsToWrite() {
    return m_hitsParams_to_write;
}

std::unordered_map<std::string, SaveDataParam> &GateToTree::getOpticalParamsToWrite() {
    return m_opticalParams_to_write;
}

std::unordered_map<std::string, SaveDataParam> &GateToTree::getSinglesParamsToWrite() {
    return m_singlesParams_to_write;
}

std::unordered_map<std::string, SaveDataParam> &GateToTree::getCoincidencesParamsToWrite() {
    return m_coincidencesParams_to_write;
}

G4bool GateToTree::getOpticalDataEnabled() const {
    return m_opticalData_enabled;
}

void GateToTree::setOpticalDataEnabled(G4bool mOpticalDataEnabled) {
    m_opticalData_enabled = mOpticalDataEnabled;

    //OK GND 2022 multiSD
    GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

    for (size_t i=0; i< digitizerMgr->m_SDlist.size();i++)
    {
    	addOpticalCollection(digitizerMgr->m_SDlist[i]->GetName());
    }


}

void GateToTree::RecordOpticalData(const G4Event *event) {

	//OK GND 2022
	auto fDM = G4DigiManager::GetDMpointer();
	auto PHC = this->GetOutputMgr()->GetPhantomHitCollection();

    m_nScintillation = 0;
    m_nCrystalOpticalWLS = 0;
    m_nPhantomOpticalWLS = 0;
    m_NumCrystalWLS = 0;
    m_NumPhantomWLS = 0;

    // Looking at Phantom Hit Collection:
    if (PHC) {

        G4int NpHits = PHC->entries();
        m_NameOfProcessInPhantom = "";

        for (G4int iPHit = 0; iPHit < NpHits; iPHit++) {
            auto pHit = (*PHC)[iPHit];
            auto processName = pHit->GetProcess();
//
            if (pHit->GoodForAnalysis() && pHit->GetPDGEncoding() == -22)// looking at optical photons only
            {
                m_NameOfProcessInPhantom = pHit->GetProcess();
//
//                //                   if (processName.find("OpRayleigh") != G4String::npos)  nPhantomOpticalRayleigh++;
//                //                   if (processName.find("OpticalMie") != G4String::npos)  nPhantomOpticalMie++;
//                //                   if (processName.find("OpticalAbsorption") != G4String::npos) {
//                //                          nPhantomOpticalAbsorption++;
//                //                         PhantomAbsorbedPhotonHitPos_X = (*PHC)[iPHit]->GetPos().x();
//                //                          PhantomAbsorbedPhotonHitPos_Y = (*PHC)[iPHit]->GetPos().y();
//                //                          PhantomAbsorbedPhotonHitPos_Z = (*PHC)[iPHit]->GetPos().z();
//                //                   }
//
                if (processName.find("OpticalWLS") != G4String::npos) {
                    m_nPhantomOpticalWLS++;      // Fluorescence counting
                    m_PhantomWLSPos_X = pHit->GetPos().x();
                    m_PhantomWLSPos_Y = pHit->GetPos().y();
                    m_PhantomWLSPos_Z = pHit->GetPos().z();
                }
//
                m_PhantomLastHitPos_X = pHit->GetPos().x();
                m_PhantomLastHitPos_Y = pHit->GetPos().y();
                m_PhantomLastHitPos_Z = pHit->GetPos().z();
                m_PhantomLastHitEnergy = pHit->GetEdep();
            }  // end GoodForAnalysis() and optical photon
        } // end loop over phantom hits
    } // end if PHC


    // Looking at Crystal Hits Collection:

 //OK GND 2022
    for (auto &&m: m_mmanager_optical)
    {
    	auto collectionID = m_optical_to_collectionID.at(m.first);
    	//G4cout<<"GateToTree collectionID "<< collectionID<<G4endl;
    	const GateHitsCollection *CHC =
    			(GateHitsCollection *) (fDM->GetHitsCollection(collectionID));

    	if (!CHC) {
    		//            G4cout << "GateToTree::RecordEndOfEvent no collection = " << m.first << "\n";
    		continue;
        		}

		if (CHC) {

			G4int NbHits = CHC->entries();
		//	G4cout<<"NbHits "<<NbHits<<G4endl;
			m_NameOfProcessInCrystal = "";

			for (G4int iHit = 0; iHit < NbHits; iHit++) {
				auto aHit = (*CHC)[iHit];
				auto processName = aHit->GetProcess();

				if (aHit->GoodForAnalysis()) {
					m_NameOfProcessInCrystal = aHit->GetProcess();

					if (processName.find("Scintillation") != G4String::npos)
						m_nScintillation++;

					if (aHit->GetPDGEncoding() == -22)  // looking at optical photons only
					{
						if (processName.find("OpticalWLS") != G4String::npos)
							m_nCrystalOpticalWLS++;


						m_CrystalLastHitPos_X = aHit->GetGlobalPos().x();
						m_CrystalLastHitPos_Y = aHit->GetGlobalPos().y();
						m_CrystalLastHitPos_Z = aHit->GetGlobalPos().z();
						m_CrystalLastHitEnergy = aHit->GetEdep();
					}
				} // end GoodForAnalysis()
			} // end loop over crystal hits
		} // end if CHC


    // counting the number of Wave Length Shifting = Fluorescence:
    if (m_nCrystalOpticalWLS > 0) m_NumCrystalWLS++;
    if (m_nPhantomOpticalWLS > 0) m_NumPhantomWLS++;

    if (event->GetTrajectoryContainer())
    	m.second.fill();

    }
}

