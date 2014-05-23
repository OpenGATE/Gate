/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*! \file GateToOpticalRaw.hh
   Created on   2012/07/09  by vesna.cuplov@gmail.com
   Implemented new class GateToOpticalRaw for Optical photons: write result of the projection.
*/


#ifndef GateToOpticalRaw_H
#define GateToOpticalRaw_H

#include <fstream>
#include "G4Timer.hh"

#include "GateVOutputModule.hh"

class GateToOpticalRawMessenger;
class GateOpticalSystem;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateToOpticalRaw :  public GateVOutputModule
{
public:
  GateToOpticalRaw(const G4String& name, GateOutputMgr* outputMgr,GateOpticalSystem* itsSystem,DigiMode digiMode);

  virtual ~GateToOpticalRaw();
  const G4String& GiveNameOfFile();

  //! It opens the raw files
  void RecordBeginOfAcquisition();
  //! It closes the raw files.
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

  //! Write the general RAW information into the header
  void WriteGeneralInfo();

private:
  GateToOpticalRawMessenger* m_asciiMessenger;

  //! Pointer to the system, used to get the system information and the projection set
  GateOpticalSystem *m_system;

  G4String m_fileName;

  std::ofstream     	      	m_headerFile; 	      	    //!< Output stream for the header file
  std::ofstream     	      	m_dataFile;   	      	    //!< Output stream for the data file

};

#endif
