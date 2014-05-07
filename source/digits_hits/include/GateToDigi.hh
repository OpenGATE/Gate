/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateToDigi_H
#define GateToDigi_H

 #include "GateVOutputModule.hh"

 class GateOutputModuleMessenger;
class GateDigitizer;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateToDigi :  public GateVOutputModule
{
public:

  GateToDigi(const G4String& name, GateOutputMgr* outputMgr,
      	     DigiMode digiMode);
  virtual ~GateToDigi();
  const G4String& GiveNameOfFile();

  void RecordBeginOfAcquisition();
  void RecordEndOfAcquisition();
  void RecordBeginOfRun(const G4Run *);
  void RecordEndOfRun(const G4Run *);
  void RecordBeginOfEvent(const G4Event *);
  void RecordEndOfEvent(const G4Event *);
  void RecordStepWithVolume(const GateVVolume * v, const G4Step *);
  //! saves the geometry voxel information
  void RecordVoxels(GateVGeometryVoxelStore *) {};


  //! Get the digitizer
  inline GateDigitizer*   GetDigitizer()
      { return m_digitizer; }

private:

  GateOutputModuleMessenger* m_digiMessenger;
      	      	      	      	      	      	      	  //!< crystal-hits into pulses that can be processed
							  //!< by the pulse-processor hits

  GateDigitizer* m_digitizer;
  G4String       m_noFileName;
};

#endif
