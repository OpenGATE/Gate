/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateParameterisedCollimatorMessenger_h
#define GateParameterisedCollimatorMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateVisAttributesMessenger;
class GateParameterisedCollimator;
class G4UIdirectory;

class GateParameterisedCollimatorMessenger: public GateVolumeMessenger
{
public:
  GateParameterisedCollimatorMessenger(GateParameterisedCollimator* itsInserter);
  ~GateParameterisedCollimatorMessenger();

  void SetNewValue(G4UIcommand*, G4String);

  virtual inline GateParameterisedCollimator* GetCollimatorInserter()
  { return (GateParameterisedCollimator*) GetVolumeCreator(); }

private:

  G4String                    name_Geometry;
  G4UIdirectory*              dir_Geometry;


  G4UIcmdWithADoubleAndUnit*  CollimatorDimensionXCmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorDimensionYCmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorFocalDistanceXCmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorFocalDistanceYCmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorHeightCmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorSeptalThicknessCmd;
  G4UIcmdWithADoubleAndUnit*  CollimatorInnerRadiusCmd;

  GateVisAttributesMessenger* visAttributesMessenger;
};

#endif
