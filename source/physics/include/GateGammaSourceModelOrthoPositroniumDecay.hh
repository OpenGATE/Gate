/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GateGammaSourceModelOrthoPositroniumDecay_hh
#define GateGammaSourceModelOrthoPositroniumDecay_hh

#include "GateGammaSourceModel.hh"
#include "TRandom3.h"
#include "GateExtendedVSourceManager.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Theorem author: Daria Kamińska ( Eur. Phys. J. C (2016) 76:445 )
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Provide generation of 3 gamma from orto Positronium decay
 **/
class GateGammaSourceModelOrthoPositroniumDecay : public GateGammaSourceModel
{
 public:
 /** Destructor
 * */
 virtual ~GateGammaSourceModelOrthoPositroniumDecay();
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
 static GateGammaSourceModelOrthoPositroniumDecay* GetInstance();

 protected:
  Double_t calculate_mQED(Double_t mass_e, Double_t w1, Double_t w2, Double_t w3) const;
  TRandom3 m_random_gen = TRandom3(0);

 protected:
 /** Constructor
  * */
 GateGammaSourceModelOrthoPositroniumDecay();
 static GateGammaSourceModelOrthoPositroniumDecay* ptrJPETOrtoPositroniumDecayModel;
};
#endif
