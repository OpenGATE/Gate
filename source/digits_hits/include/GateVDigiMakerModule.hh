/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVDigiMakerModule_h
#define GateVDigiMakerModule_h 1

#include "globals.hh"

#include "GateClockDependent.hh"

class GateDigitizer;

/*! \class  GateVDigiMakerModule
    \brief  It processes a pulse-list, generating single digis.

    - GateVDigiMakerModule - by Daniel.Strul@iphe.unil.ch

*/
class GateVDigiMakerModule : public GateClockDependent
{
public:

  //! Constructor
  GateVDigiMakerModule(GateDigitizer* itsDigitizer,
  		     const G4String& itsInputName);

  //! Destructor
  ~GateVDigiMakerModule();

  //! Convert a pulse list into a single Digi collection
  virtual void Digitize()=0;

  //! Implementation of the pure virtual method declared by the base class
  //! print-out the attributes specific of the attachment list
  virtual void DescribeMyself(size_t indent);

  virtual void SetInputName(const G4String& inputName)
    { m_inputName = inputName; }
  virtual const G4String& GetInputName()
    { return m_inputName; }
  virtual const G4String& GetCollectionName()
    { return m_collectionName; }

 protected:
  GateDigitizer*	 m_digitizer;
  G4String		 m_inputName;
  G4String		 m_collectionName;
};

#endif
