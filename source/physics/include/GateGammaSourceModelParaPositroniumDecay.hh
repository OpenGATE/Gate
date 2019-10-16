/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GateGammaSourceModelParaPositroniumDecay_hh
#define GateGammaSourceModelParaPositroniumDecay_hh

#include "GateGammaSourceModel.hh"
#include "TGenPhaseSpace.h"
#include "GateExtendedVSourceManager.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: generate two gamma particles from para Positronium decay
 **/
class GateGammaSourceModelParaPositroniumDecay : public GateGammaSourceModel
{
 public:
  /** Destructor
   * */
  virtual~GateGammaSourceModelParaPositroniumDecay();
  /** Each particle is filled with data about momentum.
   * @param: particles - list with initialized particles - without momentum information
   * */
  virtual void GetGammaParticles(std::vector<G4PrimaryParticle*>& particles) override;
  /** Return model name.
   * @return: model name - it's always simple string
   * */
  virtual G4String GetModelName() const override;
  /** If class object is not initialized this function do this and return pointer.
   * @return: class object pointer
   * */
  static GateGammaSourceModelParaPositroniumDecay *GetInstance();
 protected:
  /** Constructor
   * */
  GateGammaSourceModelParaPositroniumDecay();
  static GateGammaSourceModelParaPositroniumDecay* ptrJPETParaPositroniumDecayModel;
};

#endif
