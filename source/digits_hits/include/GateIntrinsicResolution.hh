/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateIntrinsicResolution.cc for more detals
  */


/*! \class  GateIntrinsicResolution
    \brief  GateIntrinsicResolution does some dummy things with input digi
    to create output digi

    - GateIntrinsicResolution - by name.surname@email.com

    \sa GateIntrinsicResolution, GateIntrinsicResolutionMessenger
*/

#ifndef GateIntrinsicResolution_h
#define GateIntrinsicResolution_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateIntrinsicResolutionMessenger.hh"
#include "GateSinglesDigitizer.hh"
#include "GateLevelsFinder.hh"


class GateIntrinsicResolution : public GateVDigitizerModule
{
public:
  
  GateIntrinsicResolution(GateSinglesDigitizer *digitizer, G4String name);
  ~GateIntrinsicResolution();
  
  void Digitize() override;

  // *******implement your methods here
  void SetResolution(const G4double& value ){ m_resolution=value;} ;
  void SetEref(const G4double& value ){ m_Eref=value;} ;
  void SetLightOutput(const G4double& value ){ m_LY=value;} ;
  void SetTransferEff(const G4double& value ){ m_TE=value;} ;
  void SetEdgesFraction (G4double val)       { m_edgesCrosstalkFraction = val;  }
  void SetCornersFraction (G4double val)     { m_cornersCrosstalkFraction = val;  }


  //! Allow to use file(s) as lookout table for quantum efficiency
  void UseFile(G4String aFile);

  //! Apply an unique quantum efficiency for all the channels
  void SetUniqueQE(G4double val) { m_uniqueQE = val; };

  void SetVariance(G4double val) { m_variance = val; };
  
  void CreateTable();

  void CheckVolumeName(G4String val);

  void DescribeMyself(size_t );

protected:
  G4double   m_resolution;
  G4double   m_Eref;
  G4double   m_LY; //Light Yield
  G4double   m_TE; //Transfer efficiency
  G4double   m_QE; //Quantum efficiency
  G4double m_XtalkpCent;
  G4bool isFirstEvent;

  G4double m_variance;
  
  G4double m_uniqueQE;    //!< Value of the quantum efficiency if it's unique
  G4int m_nbFiles;        //!< Number of file(s) used for creating the lookout table
  std::vector<G4String> m_file;  //!< Vector which contains the name(s) of the file(s) for the lookout table

  G4int m_nbCrystals;
      //!< Number of PhysicalVolume copies of the Inserter corresponding @ level 'm_depth-1
  //!< Number of PhysicalVolume copies of the Inserter corresponding @ level 'm_depth-1
  G4int m_level3No;
  //!< Number of PhysicalVolume copies of the Inserter corresponding @ level 'm_depth-2
  G4int m_level2No;
     //!< Number of PhysicalVolume copies of the Inserter corresponding @ volume name 'm_volumeName
  G4int m_level1No;
  G4int m_nbTables;
  GateLevelsFinder* m_levelFinder;
      //!< Number of PhysicalVolume copies of the Inserter corresponding @ volume name 'm_volumeName
  G4int m_i, m_j, m_k;    //!< numero of the volumeID
  G4int m_volumeIDNo;     //!< numero of the volumeID
  size_t m_depth;         //!< Depth of the selected volume in the Inserter
  G4double** m_table;     //!< Lookout table for the quantum efficiency of all channels
  G4String m_volumeName;  //!< Name of the module
  G4int m_testVolume;     //!< equal to 1 if the volume name is valid, 0 else
  G4double m_edgesCrosstalkFraction, m_cornersCrosstalkFraction; //!< Coefficient which connects energy to the resolution


private:
  GateDigi* m_outputDigi;

  GateIntrinsicResolutionMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif







