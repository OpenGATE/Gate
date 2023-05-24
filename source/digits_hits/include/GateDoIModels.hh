/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDoIModels.cc for more detals
  */

/*! \class  GateDoIModels
    \brief  GateDoIModels does some dummy things with input digi
    to create output digi


    \sa GateDoIModels, GateDoIModelsMessenger

    Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com

*/

#ifndef GateDoIModels_h
#define GateDoIModels_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include <iostream>
#include <vector>
#include "globals.hh"

#include "GateDoIModelsMessenger.hh"
#include "GateSinglesDigitizer.hh"
#include "GateVDoILaw.hh"


class GateDoIModels : public GateVDigitizerModule
{
public:

  GateDoIModels(GateSinglesDigitizer *digitizer, G4String name);
  ~GateDoIModels();

  void Digitize() override;

  // *******implement your methods here
  //void setDoIModel(const G4String& );

  //void DummyMethod1(GateDigi *);
  //void DummyMethod2(GateDigi *);

  //void DescribeMyself(size_t );
  virtual void DescribeMyself(size_t indent);

  //! Set the threshold
  void SetDoIAxis( G4ThreeVector val);

  inline void SetDoILaw(GateVDoILaw* law)   { m_DoILaw = law; }

protected:
  // *******implement your parameters here
 // G4String   m_DoIModels;

private:
  GateDigi* m_outputDigi;


  GateDoIModelsMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;

  GateVDoILaw* m_DoILaw;

  G4ThreeVector m_DoIaxis;

  bool flgCorrectAxis;


};

#endif








