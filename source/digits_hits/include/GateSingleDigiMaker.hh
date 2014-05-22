/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSingleDigiMaker_h
#define GateSingleDigiMaker_h 1

#include "globals.hh"

#include "GateVDigiMakerModule.hh"

class GateDigitizer;

/*! \class  GateSingleDigiMaker
    \brief  It processes a pulse-list, generating single digis.

    - GateSingleDigiMaker - by Daniel.Strul@iphe.unil.ch

*/
class GateSingleDigiMaker : public GateVDigiMakerModule
{
public:

  //! Constructor
  GateSingleDigiMaker(GateDigitizer* itsDigitizer,
  		      const G4String& itsInputName,
		      G4bool itsOutputFlag);

  //! Destructor
  ~GateSingleDigiMaker();

  //! Convert a pulse list into a single Digi collection
  void Digitize();

 protected:


};

#endif
