/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionManual.hh"

#include "GateDistributionManualMessenger.hh"
#include <math.h>
#include <CLHEP/Random/RandFlat.h>
#include "GateTools.hh"


GateDistributionManual::GateDistributionManual(const G4String& itsName)
  : GateVDistributionArray(itsName)
{
    m_messenger = new GateDistributionManualMessenger(this,itsName);
}
//___________________________________________________________________
GateDistributionManual::~GateDistributionManual()
{
}
//___________________________________________________________________
void GateDistributionManual::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent)
    	 <<"Size : "         << GetSize()
	 <<G4endl;
}
//___________________________________________________________________
void GateDistributionManual::AddPoint(G4double x,G4double y)
{
    InsertPoint(x,y);
    FillRepartition();
}
//___________________________________________________________________
void GateDistributionManual::AddPoint(G4double y)
{
    InsertPoint(y);
    FillRepartition();
}
