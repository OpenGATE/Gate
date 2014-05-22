/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePVParameterisation_H
#define GatePVParameterisation_H 1

#include "globals.hh"
#include "G4VPVParameterisation.hh"


/*! \class  GatePVParameterisation
    \brief  Base-class for GATE physical volume parameterisations
    
    - GatePVParameterisation - by Daniel.Strul@iphe.unil.ch & Giovani.Santin@cern.ch
    
    - This class is meant to allow the use of parameterised volumes in GATE
      
    - It declares a pure virtual method GetNbOfCopies() to return the number of
      copies to be made of the volume. This method must be implemented in concrete
      classes derived from GatePVParameterisation.
      
    - Note that this base-class does not implement the pure virtual method 
      ComputeTransformation() declared by G4VPVParameterisation; this method must be
      implemented in concrete classes derived from GatePVParameterisation.

      \sa GateVParameterisedInserter
*/      
class GatePVParameterisation : public G4VPVParameterisation
{ 
  public:
  
    //! Constructor.
    GatePVParameterisation();
    virtual ~GatePVParameterisation() {}
   
    //! Pure virtual method, to be implemented in concrete classes derived from G4VPVParameterisation
    //! This method must provide the number of copies of the volume
    virtual int GetNbOfCopies()=0;
};

#endif


