/*----------------------
  OpenGATE Collaboration

  Didier Benoit <benoit@cppm.in2p3.fr>
  Franca Cassol Brunner <cassol@cppm.in2p3.fr>

  Copyright (C) 2009 imXgam/CNRS, CPPM Marseille

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*!
  \file GateToImageCT.hh

  \brief Class GateToImageCT
  \author Didier Benoit <benoit@cppm.in2p3.fr>
  \author Franca Cassol Brunner <cassol@cppm.in2p3.fr>
*/

#ifndef GATETOIMAGECT_HH
#define GATETOIMAGECT_HH

#include <vector>

#include "GateVOutputModule.hh"
#include "GateApplicationMgr.hh"

class GateVSystem;
class GateImageCT;
class GateToImageCTMessenger;

class GateToImageCT : public GateVOutputModule
{
public:
  //! Public constructor
  GateToImageCT( const G4String&, GateOutputMgr*,
                 GateVSystem*, DigiMode );
  //! Public destructor
  virtual ~GateToImageCT();
  const G4String& GiveNameOfFile();

  void RecordBeginOfAcquisition();
  void RecordEndOfAcquisition();
  void RecordBeginOfRun( const G4Run* );
  void RecordEndOfRun( const G4Run* );
  void RecordBeginOfEvent( const G4Event* );
  void RecordEndOfEvent( const G4Event* );
  void RecordStepWithVolume( const GateVVolume*, const G4Step* );
  void RecordVoxels( GateVGeometryVoxelStore* ){};

public:
  void ModuleGeometry();
  void PixelGeometry( const G4String[], const G4String[] );

public:
  //Setters ans Getters
  void SetFileName( G4String& );
  inline G4String GetFileName()
  { return m_fileName; }

  void SetStartSeed( G4int );
  inline G4int GetStartSeed()
  { return m_seed; }

  void SetVRTFactor( G4int );
  inline G4int GetVRTFactor()
  { return m_vrtFactor; }

  void SetFastPixelXNb( G4int );
  inline G4int GetFastPixelXNb()
  { return m_fastPixelXNb; }

  void SetFastPixelYNb( G4int );
  inline G4int GetFastPixelYNb()
  { return m_fastPixelYNb; }

  void SetFastPixelZNb( G4int );
  inline G4int GetFastPixelZNb()
  { return m_fastPixelZNb; }

  void SetDetectorX( G4double );
  inline G4double GetDetectorX()
  { return m_detectorInX; }

  void SetDetectorY( G4double );
  inline G4double GetDetectorY()
  { return m_detectorInY; }

  void SetSourceDetector( G4double );
  inline G4double GetSourceDetector()
  { return m_sourceDetector; }

  inline void SetVerboseLevel( G4int verbose )
  {
    GateVOutputModule::SetVerboseLevel( verbose );
  }

  inline G4double GetTimeStart()
  { return GateApplicationMgr::GetInstance()->GetTimeStart(); }

  inline G4double GetTimeStop()
  { return GateApplicationMgr::GetInstance()->GetTimeStop(); }

  inline G4double GetFrameDuration()
  { return GateApplicationMgr::GetInstance()->GetTimeSlice(); }

  inline G4double GetTotalDuration()
  { return GetTimeStop() - GetTimeStart();}

  G4int GetFrameID()
  {
    return static_cast<G4int>( GetTimeStart() / GetFrameDuration() );
  }

  //Transform the pixelID of GATE in Reconstruction and/or ImageJ
  size_t InverseMatrixPixel( size_t );

  //return the real value of pixelID
  size_t TransformPixel( size_t, size_t, size_t );

private:
  GateVSystem* m_system;
  GateImageCT* m_gateImageCT;
  GateToImageCTMessenger* m_messenger;

private:
  G4String m_name;
  size_t m_pixelInRaw;
  size_t m_pixelInColumn;
  G4int n_digit;
  G4String m_fileName;
  G4int m_seed;
  G4int m_vrtFactor;
  G4String m_inputDataChannel;
  G4int m_fastPixelXNb;
  G4int m_fastPixelYNb;
  G4int m_fastPixelZNb;
  G4bool m_selfDigi;
  G4double m_detectorInX;
  G4double m_detectorInY;
  G4double m_sourceDetector;

private:
  //characteristics of the CT scanner
  //Modules
  std::vector<G4double> lenghtOfModuleByAxis;
  std::vector<size_t> numberOfModuleByAxis;

  //Clusters
  std::vector<size_t> numberOfClusterByAxis;

  //Pixels
  std::vector<G4double> lenghtOfPixelByAxis;
  std::vector<size_t> numberOfPixelByAxis;
  std::vector<size_t> numberOfPixelByCluster;
};


#endif
