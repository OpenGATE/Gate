/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionManual_h
#define GateDistributionManual_h 1

#include "GateVDistributionArray.hh"
#include "GateDistributionManualMessenger.hh"

class GateDistributionManualMessenger;
class GateDistributionManual : public GateVDistributionArray
{
  public:

    //! Constructor
    GateDistributionManual(const G4String& itsName);
    //! Destructor
    virtual ~GateDistributionManual() ;

    virtual void DescribeMyself(size_t indent);
    void AddPoint(G4double x,G4double y);
    void AddPoint(G4double y);
//    void RemovePoint(G4double x);
  private:
    //! private members
    GateDistributionManualMessenger* m_messenger;
};


#endif
