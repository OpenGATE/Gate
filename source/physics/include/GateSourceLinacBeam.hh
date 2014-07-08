/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATESOURCELINACBEAM_H
#define GATESOURCELINACBEAM_H 1

#include <iomanip>   
#include "globals.hh"
#include "G4VPrimaryGenerator.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4PrimaryVertex.hh"
#include "G4ParticleMomentum.hh"
#include "TROOT.h"
#include "TFile.h"
#include "TH1D.h"
#include "TMath.h"
#include "TKey.h"
#include "GateVSource.hh"

class GateSourceLinacBeamMessenger;

//-------------------------------------------------------------------------------------------------
class GateSourceLinacBeam : public GateVSource
{
public:
  GateSourceLinacBeam(G4String name);
  virtual ~GateSourceLinacBeam();
  
  virtual void GeneratePrimaryVertex(G4Event* evt);
  // virtual void Update();
  void SetSourceFromPhaseSpaceFilename(G4String f);
  void SetRmaxFilename(G4String f);
  void SetReferencePosition(G4ThreeVector p);
  
protected:
  GateSourceLinacBeamMessenger* mMessenger;
  G4String mSourceFromPhaseSpaceFilename;
  G4String mRmaxFilename;
  TFile * mPhaseSpaceFile;
  G4ThreeVector mReferencePosition;

  G4int mNumberOfVolume;
  G4int mNbOfRadiusBins;
  G4int mNbOfRadiusBinsForAngle;
  G4int mNbOfEnergyBinsForAngle;
  std::vector<G4String> mVolumeNames;

  typedef std::vector<TH1D*> HistoVector1DType;
  typedef std::vector<HistoVector1DType> HistoVector2DType;
  typedef std::vector<HistoVector2DType> HistoVector3DType;

  TH1D *            mHistoVolume;  // Histo with the probability the the particle come from one volume
  HistoVector1DType mHistoRadius;           // [i] i=volume
  HistoVector2DType mHistoEnergy;           // [i][j] i=vol j=radius -> energy histo
  HistoVector3DType mHistoThetaDirection;   // [i][j] i=vol j=radius k=energy -> angle histo
  HistoVector3DType mHistoPhiDirection;     // [i][j] i=vol j=radisu k=energy -> angle histo

  double GetRmaxFromTime(double time);
  int GetIndexFromTime(double time);
  std::vector<double> mTimeList;
  std::vector<double> mRmaxList;
  
  void PrintHistoInfo(TH1D * h) {
    DD(h->GetName());
    DD(h->GetBinWidth(0));
    DD(h->GetNbinsX());
  }
};

#endif
