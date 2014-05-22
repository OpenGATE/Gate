/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateClockDependent.hh"

#include "GateClock.hh"
#include "GateTools.hh"

//-------------------------------------------------------------------------------------------------------
// Constructor.
// The flag "itsFlagCanBeDisabled" tells whether the user is allowed to disable this object
// (this flag should be set to false for critical objects, such as the world volume for instance)
 GateClockDependent::GateClockDependent(const G4String& itsName, G4bool canBeDisabled)
: GateNamedObject(itsName), 
  bIsEnabled(true),
  bCanBeDisabled(canBeDisabled)
{
}
//-------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------
// Asks the clock and return the current time 
double GateClockDependent::GetCurrentTime()
{
  GateClock* theClock = GateClock::GetInstance();
  G4double aTime = theClock->GetTime();
  return aTime;
}
//-------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------
// Method overloading GateNamedObject::Describe()
// Print-out a description of the object
void GateClockDependent::Describe(size_t indent)
{
  GateNamedObject::Describe(indent);
  if (CanBeDisabled())
    G4cout << GateTools::Indent(indent) << "Is enabled?         " << ( IsEnabled() ? "Yes" : "No") << G4endl;
}
//-------------------------------------------------------------------------------------------------------

