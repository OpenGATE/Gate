/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
/*! \file GateImageDeformActorMessenger.hh
    \class GateImageDeformActorMessenger :
    \brief Messenger class of GateImageDeformActor
    \author yannick.lemarechal@univ-brest.fr
	    david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEIMAGEDEFORMMESSENGERMESSENGER_H
#define GATEIMAGEDEFORMMESSENGERMESSENGER_H 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GateActorMessenger.hh"
// #include "GateImageDeformActor.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
// class GateActorMessenger;
class GateImageDeformActor;
//-------------------------------------------------------------------------------------------------
class GateImageDeformActorMessenger: public GateActorMessenger
{
public:
  
    /*!
      \brief GateImageDeformActorMessenger constructor
      \param v GateImageDeformActor associated object
    */
    GateImageDeformActorMessenger(GateImageDeformActor * v);
    
    /*!
      \brief GateImageDeformActor destructor
    */    
    ~GateImageDeformActorMessenger();

    
    /*!
      \fn virtual void SetNewValue(G4UIcommand*, G4String);
      \brief Function from GateActorMessenger
    */
    virtual void SetNewValue(G4UIcommand*, G4String);
    
    /*!
      \fn void BuildCommands(G4String base);
      \brief Function from GateActorMessenger that create "setPDFFile" command
    */
    void BuildCommands(G4String base);
    
    
    GateImageDeformActor *mDeform;

  
protected:
    G4UIcmdWithAString * pName;
    G4UIcmdWithAString * pSetPDFFile;
    G4UIcmdWithABool * mInitialization;
   
};
//-------------------------------------------------------------------------------------------------

#endif
