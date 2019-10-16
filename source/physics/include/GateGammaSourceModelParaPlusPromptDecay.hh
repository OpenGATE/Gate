/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateGammaSourceModelParaPlusPromptDecay_hh
#define GateGammaSourceModelParaPlusPromptDecay_hh

#include "GateGammaSourceModel.hh"
#include "TGenPhaseSpace.h"
#include "GateExtendedVSourceManager.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: generate two gamma particles from para positronium decay and one from the deexcitation
 **/
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
