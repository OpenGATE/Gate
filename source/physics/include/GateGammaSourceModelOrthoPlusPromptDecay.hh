/**
 *  @copyright Copyright 2017 The J-PET Gate Authors. All rights reserved.
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
 *  @file GateGammaSourceModelOrthoPlusPromptDecay.hh
 */

#ifndef GateGammaSourceModelOrthoPlusPromptDecay_hh
#define GateGammaSourceModelOrthoPlusPromptDecay_hh

#include "GateGammaSourceModel.hh"
#include "TRandom3.h"
#include "GateExtendedVSourceManager.hh"

/**Author: Mateusz Bała
 * Email: bala.mateusz@gmail.com
 * Theorem author: Daria Kamińska ( Eur. Phys. J. C (2016) 76:445 )
 * About class: Provide generation of 3 gamma from ortho positronium decay
 * */
class GateGammaSourceModelOrthoPlusPromptDecay : public GateGammaSourceModel
{
 public:
  /** Destructor
   * */
  virtual ~GateGammaSourceModelOrthoPlusPromptDecay();
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
  static GateGammaSourceModelOrthoPlusPromptDecay* GetInstance();
 private:
  Double_t calculate_mQED(Double_t mass_e, Double_t w1, Double_t w2, Double_t w3) const;
  TRandom3 m_random_gen = TRandom3(0);
 private:
  /** Constructor
   * */
  GateGammaSourceModelOrthoPlusPromptDecay();
  static GateGammaSourceModelOrthoPlusPromptDecay* ptrOrthoPlusPromptDecayModel;

 private:
  void AddGammasFromOrtoPositronium( std::vector<G4PrimaryParticle*>& particles );
  void AddGammaFromDeexcitation( std::vector<G4PrimaryParticle*>& particles );
  G4ThreeVector GetRandomVectorOnSphere();
};
#endif
