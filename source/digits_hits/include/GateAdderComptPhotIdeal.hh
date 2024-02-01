/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  The purpose of this class is to help to create new users digitizer module(DM).

  \class  GateAdderComptPhotIdeal

  Added to GND in November 2022 by olga.kochebina@cea.fr

  Last modification (Adaptation to GND): July 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#ifndef GateAdderComptPhotIdeal_h
#define GateAdderComptPhotIdeal_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"
#include "G4Types.hh"


#include "GateAdderComptPhotIdealMessenger.hh"
#include "GateSinglesDigitizer.hh"
#include "GateVEffectiveEnergyLaw.hh"

class GateAdderComptPhotIdealMessenger;

class GateAdderComptPhotIdeal : public GateVDigitizerModule
{
public:
  
  GateAdderComptPhotIdeal(GateSinglesDigitizer *digitizer, G4String name);
  virtual ~GateAdderComptPhotIdeal();
  
  void Digitize() override;

  void SetEvtRejectionPolicy(G4bool flgval){m_flgRejActPolicy=flgval; };
  void DescribeMyself(size_t);
  GateDigi* CentroidMergeComptPhotIdeal(GateDigi *right, GateDigi *output);

  bool m_flgEvtRej;
  std::vector<G4int> m_lastTrackID;

protected:
  G4double   m_GateAdderComptPhotIdeal;

private:
  GateDigi* m_outputDigi;

  GateAdderComptPhotIdealMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;
  G4bool m_flgRejActPolicy;
  constexpr static double epsilonEnergy=0.00001;

};

#endif








