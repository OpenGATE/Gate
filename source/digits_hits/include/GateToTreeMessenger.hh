/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


//
// Created by mdupont on 17/05/19.
//

#ifndef GATE_GATETOTREEMESSENGER_HH
#define GATE_GATETOTREEMESSENGER_HH

#include <memory>
#include <unordered_map>
#include "GateOutputModuleMessenger.hh"

class GateToTree;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithoutParameter;


class GateToTreeMessenger : public GateOutputModuleMessenger
{
public:
  GateToTreeMessenger(GateToTree *m);

  virtual ~GateToTreeMessenger();

  void SetNewValue(G4UIcommand *icommand, G4String string) override;

private:
  void DeleteMap( std::unordered_map<G4UIcmdWithoutParameter*, G4String>& m );

private:

  std::unique_ptr<G4UIcmdWithAString> m_addFileNameCmd;
  std::unique_ptr<G4UIcmdWithoutParameter> m_enableHitsOutput;
  std::unique_ptr<G4UIcmdWithoutParameter> m_disableHitsOutput;
  std::unique_ptr<G4UIcmdWithoutParameter> m_disableAllHitsBranches;

  std::unique_ptr<G4UIcmdWithoutParameter> m_enableOpticalDataOutput;
  std::unique_ptr<G4UIcmdWithoutParameter> m_disableOpticalDataOutput;
  std::unique_ptr<G4UIcmdWithoutParameter> m_disableOpticalDataBranches;

  std::unique_ptr<G4UIcmdWithAString> m_addCollectionCmd;
  std::unique_ptr<G4UIcmdWithoutParameter> m_disableAllSinglesBranches;
  std::unique_ptr<G4UIcmdWithoutParameter> m_disableAllCoincidencesBranches;
  GateToTree *m_gateToTree;

  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_maphits_cmdParameter_toTreeParameter_disable;
  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_maphits_cmdParameter_toTreeParameter_enable;
  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_mapoptical_cmdParameter_toTreeParameter_disable;
  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_mapoptical_cmdParameter_toTreeParameter_enable;
  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_mapsingles_cmdParameter_toTreeParameter_disable;
  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_mapsingles_cmdParameter_toTreeParameter_enable;
  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_mapscoincidences_cmdParameter_toTreeParameter_disable;
  std::unordered_map<G4UIcmdWithoutParameter*, G4String> m_mapscoincidences_cmdParameter_toTreeParameter_enable;

};


#endif //GATE_GATETOTREEMESSENGER_HH
