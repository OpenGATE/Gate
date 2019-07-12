/**
 *  @copyright Copyright 2019 The J-PET Gate Authors. All rights reserved.
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  @file GateGammaModelPrimaryParticleInformation.hh
 */
#ifndef GateGammaModelPrimaryParticleInformation_hh
#define GateGammaModelPrimaryParticleInformation_hh
#include <G4VUserPrimaryParticleInformation.hh>
#include <G4ThreeVector.hh>

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  About class: Provides additional data about gamma particle - useful for analysis.
 * */
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

 private:
  GammaSourceModel fGammaSourceModel = GammaSourceModel::Unknown;
  GammaKind fGammaKind = GammaKind::GammaUnknown;
  G4ThreeVector fInitialPolarization = G4ThreeVector( 0, 0, 0 );
};

#endif
