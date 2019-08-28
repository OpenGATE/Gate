/**
 *  @copyright Copyright 2018 The J-PET Gate Authors. All rights reserved.
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
 *  @file GateGammaSourceModelParaPlusPromptDecay.hh
 */

#ifndef GateGammaSourceModelParaPlusPromptDecay_hh
#define GateGammaSourceModelParaPlusPromptDecay_hh

#include "GateGammaSourceModel.hh"
#include "TGenPhaseSpace.h"
#include "GateExtendedVSourceManager.hh"

/**Author: Mateusz Bała
 * Email: bala.mateusz@gmail.com
 * About class: generate two gamma particles from para positronium decay and one from the deexcitation
 */
class GateGammaSourceModelParaPlusPromptDecay : public GateGammaSourceModel
{
 public:
  virtual~GateGammaSourceModelParaPlusPromptDecay();
  /** Each particle is filled with data about momentum.
   * @param: particles - list with initialized particles - without momentum information
   * */
  virtual void GetGammaParticles( std::vector<G4PrimaryParticle*>& particles ) override;
  /** Return model name.
   * @return: model name - it's always simple string
   * */
  virtual G4String GetModelName() const override;
  /** If class object is not initialized this function do this and return pointer.
   * @return: class object pointer
   * */
  static GateGammaSourceModelParaPlusPromptDecay *GetInstance();

 protected:
  GateGammaSourceModelParaPlusPromptDecay();
  static GateGammaSourceModelParaPlusPromptDecay* ptrJPETParaPlusPromptDecayModel;

 protected:
  void AddGammasFromParaPositronium( std::vector<G4PrimaryParticle*>& particles );
  void AddGammaFromDeexcitation( std::vector<G4PrimaryParticle*>& particles );
  G4ThreeVector GetRandomVectorOnSphere();
};

#endif
