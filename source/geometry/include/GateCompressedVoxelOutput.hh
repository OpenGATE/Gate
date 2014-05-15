/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#ifndef GateCompressedVoxelOutput_H
#define GateCompressedVoxelOutput_H

#include "GateVOutputModule.hh"
#include "GateDetectorConstruction.hh"
#include <valarray>

class GateTrajectoryNavigator; 
class GateCompressedVoxelOutputMessenger; 
class GateCompressedVoxelParameterized;
class GateVVolume;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateCompressedVoxelOutput :  public GateVOutputModule
{
public:

  GateCompressedVoxelOutput(const G4String& name, const G4String& phantomName, GateOutputMgr* outputMgr,DigiMode digiMode, GateCompressedVoxelParameterized* inserter);
  virtual ~GateCompressedVoxelOutput();
  const G4String& GiveNameOfFile();

  void RecordBeginOfAcquisition();
  void RecordEndOfAcquisition();
  void RecordBeginOfRun(const G4Run *);
  void RecordEndOfRun(const G4Run *);
  void RecordBeginOfEvent(const G4Event *);
  void RecordEndOfEvent(const G4Event *);
  void RecordStepWithVolume(const GateVVolume * v, const G4Step *);
  void RecordVoxels(GateVGeometryVoxelStore *) {};


  virtual void SetVerboseLevel(G4int val);
  
  //! Get the output file name
  inline  const  G4String& GetFileName()const { return m_fileName; }
  
  //! Set the output file name
  void   SetFileName(const G4String aName)    { m_fileName = aName; }
  void   SetSaveUncertainty(G4bool b) ;


private:


  std::valarray<float>*              m_array;            // the array for collecting energy deposits
  std::valarray<float>*              m_arraySquare;      // the array for square of energy deposits
  std::valarray<unsigned int>*       m_arrayCounts;      // the array for counts

  GateTrajectoryNavigator*           m_trajectoryNavigator;
  GateCompressedVoxelOutputMessenger*          m_outputMessenger;
  GateCompressedVoxelParameterized* m_inserter;

  G4String                           m_fileName;
  G4bool                             m_uncertainty;
  G4String                           m_phantomName;

};

#endif
