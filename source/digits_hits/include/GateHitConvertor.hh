/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateHitConvertor_h
#define GateHitConvertor_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateCrystalHit.hh"
#include "GatePulse.hh"
#include "GateClockDependent.hh"

class GateHitConvertorMessenger;

class GateHitConvertor : public GateClockDependent
{
  private:

    GateHitConvertor() ;

  public:
     static GateHitConvertor* GetInstance();
     virtual ~GateHitConvertor();

     virtual GatePulseList* ProcessHits(const GateCrystalHitsCollection* hitCollection);
     virtual void DescribeMyself(size_t indent);

     static  const G4String& GetOutputAlias()
     {return theOutputAlias;}

  private:
    virtual void ProcessOneHit(const GateCrystalHit* hit,GatePulseList* pulseList);

  private:
    GateHitConvertorMessenger *m_messenger;

    static const G4String theOutputAlias;
};


#endif
