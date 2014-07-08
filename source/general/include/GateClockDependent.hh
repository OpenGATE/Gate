/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateClockDependent_h
#define GateClockDependent_h 1

#include "globals.hh"
#include "GateNamedObject.hh"

//-------------------------------------------------------------------------------------------------
/*! \class  GateClockDependent
    \brief  A GateClockDependent is a named object that can communicate with the clock
    \brief  to ask for the current time, and that can be enabled or disabled
    
    - GateClockDependent - by Daniel.Strul@iphe.unil.ch 
    
    - The GateClockDependent is based on GateNamedObject and thus has all the
      properties of Gate named objects (a name and a type-name). 
      
    - In addition, it can ask the clock for the current time, and can be
      enabled or diabled
*/      
//-------------------------------------------------------------------------------------------------
class GateClockDependent : public GateNamedObject
{
  public:
    //! Constructor.
    //! The flag "itsFlagCanBeDisabled" tells whether the user is allowed to disable this object
    //! (this flag should be set to false for critical objects, such as the world volume for instance)
    GateClockDependent(const G4String& itsName,G4bool canBeDisabled=true);
    inline virtual  ~GateClockDependent() {}

  public:
    //! Method overloading GateNamedObject::Describe()
    //! Print-out a description of the object
    virtual void Describe(size_t indent=0);
     
    //! Asks the clock and return the current time 
    virtual double GetCurrentTime();

    //! Returns the value of the object enabled/disabled status flag
    inline virtual G4bool IsEnabled() const { return bIsEnabled;}

    //! Enable the object
    inline virtual void Enable(G4bool val) { bIsEnabled = val; }  

    //! Tells whether the user is allowed to disable the object
    inline virtual G4bool CanBeDisabled() const { return bCanBeDisabled;}

   protected:
    //! Flag telling whether the object is enabled (active) or disabled (off)
    G4bool bIsEnabled;

    //! This flag must be set to 1 if the object can be disabled
    //! This is meant to prevent the user from disabling critical objects
    //! such as the world volume
    G4bool bCanBeDisabled;
};

#endif

