/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateToInterfile_H
#define GateToInterfile_H

#include <fstream>
#include "G4Timer.hh"

#include "GateVOutputModule.hh"

class GateToInterfileMessenger;
class GateSPECTHeadSystem;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateToInterfile :  public GateVOutputModule
{
public:
  GateToInterfile(const G4String& name, GateOutputMgr* outputMgr,GateSPECTHeadSystem* itsSystem,DigiMode digiMode);

  virtual ~GateToInterfile();
  const G4String& GiveNameOfFile();

  //! It opens the Interfile files
  void RecordBeginOfAcquisition();
  //! It closes the Interfile files.
  void RecordEndOfAcquisition();

  //! SReset the data array
  void RecordBeginOfRun(const G4Run *);
  //! Saves the latest results
  void RecordEndOfRun(const G4Run *);

  void RecordBeginOfEvent(const G4Event *) {}
  void RecordEndOfEvent(const G4Event *) {}
  void RecordStepWithVolume(const GateVVolume *, const G4Step *) {}


  //! saves the geometry voxel information
  void RecordVoxels(GateVGeometryVoxelStore *) {};

  //! Get the output file name
  const  G4String& GetFileName()             { return m_fileName; };
  //! Set the output file name
  void   SetFileName(const G4String& aName)   { m_fileName = aName; };

  /*! \brief Overload of the base-class' virtual method to print-out a description of the module

	\param indent: the print-out indentation (cosmetic parameter)
  */
  void Describe(size_t indent=0);

  //! Write the general INTERFILE information into the header
  void WriteGeneralInfo();
  //! Write the GATE specific scanner information into the header
  void WriteGateScannerInfo();
  //! Write the GATE specific run information into the header
  void WriteGateRunInfo(G4int runNb);

private:
  GateToInterfileMessenger* m_asciiMessenger;

  //! Pointer to the system, used to get the system information and the projection set
  GateSPECTHeadSystem *m_system;

  G4String m_fileName;


  std::ofstream     	      	m_headerFile; 	      	    //!< Output stream for the header file
  std::ofstream     	      	m_dataFile;   	      	    //!< Output stream for the data file

};

#endif
