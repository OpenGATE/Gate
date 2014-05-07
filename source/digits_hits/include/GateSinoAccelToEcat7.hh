/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_ECAT7

#ifndef GateSinoAccelToEcat7_H
#define GateSinoAccelToEcat7_H

#include "fstream"
#include "G4Timer.hh"

#include "GateVOutputModule.hh"

#include <stdio.h>

//ECAT7 include file
#include "matrix.h"
#include "machine_indep.h"

class GateSinoAccelToEcat7Messenger;
class GateEcatAccelSystem;
class GateToSinoAccel;
class GateSinogram;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateSinoAccelToEcat7 :  public GateVOutputModule
{
public:
  GateSinoAccelToEcat7(const G4String& name, GateOutputMgr* outputMgr,GateEcatAccelSystem* itsSystem,DigiMode digiMode);
  virtual ~GateSinoAccelToEcat7(); //!< Destructor
  const G4String& GiveNameOfFile();

  //! It opens the ECAT7 file
  void RecordBeginOfAcquisition();
  //! It closes the ECAT7 file.
  void RecordEndOfAcquisition();

  //! Reset the data array
  void RecordBeginOfRun(const G4Run *);
  //! Saves the latest results
  void RecordEndOfRun(const G4Run *);

  void RecordBeginOfEvent(const G4Event *) {}
  void RecordEndOfEvent(const G4Event *) {}
  void RecordStepWithVolume(const GateVVolume * , const G4Step *) {}


  //! saves the geometry voxel information
  void RecordVoxels(GateVGeometryVoxelStore *) {};

  //! Get the output file name
  const  G4String& GetFileName()                { return m_fileName;       };
  //! Set the output file name
  void   SetFileName(const G4String& aName)     { m_fileName = aName;      };
  //! Set azimutal mashing factor
  void SetMashing(const G4int aNumber)          { m_mashing = aNumber;     };
  //! Set span factor
  void SetSpan(const G4int aNumber)             { m_span = aNumber;        };
  //! Set maximum ring difference
  void SetMaxRingDiff(const G4int aNumber)      { m_maxRingDiff = aNumber; };
  //! Set ECAT camera number
  void SetEcatAccelCameraNumber(const G4int aNumber) { m_ecatAccelCameraNumber = aNumber; } ;

  /*! \brief Overload of the base-class' virtual method to print-out a description of the module

	\param indent: the print-out indentation (cosmetic parameter)
  */
  void Describe(size_t indent=0);

  //! Fill the main header
  void FillMainHeader();
  //! Write the GATE specific scanner information into the main header

  //! Fill the sub header
  void FillData();

private:
  GateSinoAccelToEcat7Messenger* m_asciiMessenger;

  //! Pointer to the system, used to get the system information and the sinogram
  GateEcatAccelSystem *m_system;

  G4String m_fileName;

  MatrixFile *m_ptr;
  Main_header *mh;
  Scan3D_subheader *sh;

  G4int                m_mashing;           //! azimutal mashing factor
  G4int                m_span;              //! polar mashing factor
  G4int                m_maxRingDiff;       //! maximum ring difference

  G4int                m_segmentNb;
  G4int                *m_delRingMinSeg;
  G4int                *m_delRingMaxSeg;
  G4int                *m_zMinSeg;
  G4int                *m_zMaxSeg;
  G4int                *m_segment;
  G4int                 m_ecatAccelCameraNumber;

  //G4std::ofstream     	      	m_headerFile; 	      	    //!< Output stream for the header file
  //G4std::ofstream     	      	m_dataFile;   	      	    //!< Output stream for the data file


};

#endif
#endif
