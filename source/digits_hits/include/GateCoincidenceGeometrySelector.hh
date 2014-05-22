/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidenceGeometrySelector_h
#define GateCoincidenceGeometrySelector_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"
#include "GateObjectStore.hh"
#include "GateVCoincidencePulseProcessor.hh"

class GateCoincidenceGeometrySelectorMessenger;


class GateCoincidenceGeometrySelector : public GateVCoincidencePulseProcessor
{
public:



  //! Destructor
  virtual ~GateCoincidenceGeometrySelector() ;


  //! Constructs a new dead time attached to a GateDigitizer
  GateCoincidenceGeometrySelector(GateCoincidencePulseProcessorChain* itsChain,
                                  const G4String& itsName);

public:

  //! Returns the GeometrySelector
  G4double GetMaxS() const {return m_maxS;}
  G4double GetMaxDeltaZ() const {return m_maxDeltaZ;}

  //! Set the GeometrySelector
  void SetMaxS(G4double val) { m_maxS = val;}
  void SetMaxDeltaZ(G4double val) { m_maxDeltaZ = val;}

  //! Implementation of the pure virtual method declared by the base class GateClockDependent
  //! print-out the attributes specific of the GeometrySelector
  virtual void DescribeMyself(size_t indent);

protected:

  /*! Implementation of the pure virtual method declared by the base class GateVCoincidencePulseProcessor*/
  GateCoincidencePulse* ProcessPulse(GateCoincidencePulse* inputPulse,G4int iPulse);



private:
  G4double m_maxS;  //!< contains the rebirth time.
  G4double m_maxDeltaZ;  //!< contains the rebirth time.
  GateCoincidenceGeometrySelectorMessenger *m_messenger;    //!< Messenger
};


#endif
