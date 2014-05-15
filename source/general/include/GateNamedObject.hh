/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateNamedObject_h
#define GateNamedObject_h 1

#include "globals.hh"

/*! \class GateNamedObject
  \brief  A GateNamedObject is an object with a name and a type-name
    
  - GateNamedObject - by Daniel.Strul@iphe.unil.ch 
    
  - A named object has two main attributes:
  - Its name, which allows to find it in a list (unique for a given object-type)
  - Its type-name, which may be used to retrieve specific types of objects
*/      
class GateNamedObject
{
public:
  /*! \brief Constructor
    \param itsName:       	  the name chosen for this object
  */    
  GateNamedObject(const G4String& itsName)
    : mName(itsName), nVerboseLevel(0)
  {}
  //| Destructor
  virtual ~GateNamedObject() {}

public:
  //! \name Getters and setters
  //@{

  //! Get the object name
  inline virtual const G4String& GetObjectName() const   	{ return mName;}

  //! Get the verbosity 
  inline G4int GetVerbosity() const				{ return nVerboseLevel; }

  //! Set the verbosity of the system. 
  inline virtual void  SetVerbosity(G4int val)		{  nVerboseLevel = val; }

  //@}
     

  /*! \brief Virtual method to print-out a description of the object
    \param indent: the print-out indentation (cosmetic parameter)
  */    
  virtual void Describe(size_t indent=0);
     
protected:
  G4String mName;  	      	      	//!< Name of the object
  G4int nVerboseLevel;     	      	//!< Verbosity level
};

#endif

