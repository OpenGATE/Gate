/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateGammaModelPrimaryParticleInformation_hh
#define GateGammaModelPrimaryParticleInformation_hh
#include <G4VUserPrimaryParticleInformation.hh>
#include <G4ThreeVector.hh>

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Provides additional data about gamma particle - useful for analysis.
 **/
class GateGammaModelPrimaryParticleInformation : public G4VUserPrimaryParticleInformation
{
 public:
  GateGammaModelPrimaryParticleInformation();
  virtual ~GateGammaModelPrimaryParticleInformation();

  virtual void Print() const;

  enum GammaSourceModel
  {
   Unknown = 0,
   Single = 1,
   ParaPositronium = 2,
   OrthoPositronium = 3,
   ParaPositroniumAndPrompt = 4,
   OrthoPositroniumAndPrompt = 5,
   Other = 6
  };

  void setGammaSourceModel( GammaSourceModel model );
  GammaSourceModel getGammaSourceModel() const;

  enum GammaKind
  {
   GammaUnknown = 0,
   GammaSingle = 1,
   GammaPrompt = 2,
   GammaFromParaPositronium = 3,
   GammaFromOrthoPositronium = 4,
   GammaFromOtherModel = 5
  };

  void setGammaKind( GammaKind kind );
  GammaKind getGammaKind() const;

  void setInitialPolarization( const G4ThreeVector& polarization );
  G4ThreeVector getInitialPolarization() const;

 protected:
  GammaSourceModel fGammaSourceModel = GammaSourceModel::Unknown;
  GammaKind fGammaKind = GammaKind::GammaUnknown;
  G4ThreeVector fInitialPolarization = G4ThreeVector( 0, 0, 0 );
};

#endif
